"""Detached CPU handoff: verify training completion, submit/evaluate once.

Account submission limits are retried once per minute. No training retry, extra
updates or new research stage is authorized by this controller.
"""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import time
import traceback
from tools.research_core import atomic,file_sha,read


def job_state(job):
    result=subprocess.run(['sacct','-n','-P','-j',str(job),'--format=JobIDRaw,State'],capture_output=True,text=True,timeout=30)
    if result.returncode:raise RuntimeError('Accounting unavailable: '+result.stderr.strip())
    for line in result.stdout.splitlines():
        parts=line.split('|')
        if parts[0].strip()==str(job):return parts[1].strip().split()[0].rstrip('+')
    return 'UNKNOWN'


def recover_job(name,start):
    result=subprocess.run(['squeue','-u',os.environ['USER'],'-h','-o','%i|%j'],capture_output=True,text=True,timeout=30)
    if result.returncode:raise RuntimeError('Queue unavailable: '+result.stderr.strip())
    found=[line.split('|')[0].strip() for line in result.stdout.splitlines() if line.split('|')[-1].strip()==name]
    if not found:
        result=subprocess.run(['sacct','-u',os.environ['USER'],'-S',start,'-n','-P',
                               '--name='+name,'--format=JobIDRaw,JobName%80'],capture_output=True,text=True,timeout=30)
        if result.returncode:raise RuntimeError('Recovery accounting unavailable: '+result.stderr.strip())
        found=[line.split('|')[0].strip() for line in result.stdout.splitlines() if line.split('|')[1].strip()==name and line.split('|')[0].strip().isdigit()]
    found=sorted(set(found));assert len(found)<=1,'Ambiguous prior evaluation submissions'
    return found[0] if found else None


def evaluation_command(machine,repo,train,name,config):
    s=machine['Slurm'];root=machine['DataRoot']
    command=['sbatch','--parsable','--no-requeue','--partition='+s['Partition'],'--account='+s['Account'],
        '--qos='+s['Qos'],'--gres='+s['Gres'],'--cpus-per-task='+str(s['Cpus']),
        '--time='+machine['SftTimes']['evaluate'],'--job-name='+name,'--output='+root+'/logs/slurm/%j.log']
    if s.get('Exclude'):command+=['--exclude='+s['Exclude']]
    if s.get('NodeList'):command+=['--nodelist='+s['NodeList']]
    return command+[str(repo/'tools/sft_job.sh'),str(repo),machine['CondaRoot'],machine['CondaEnv'],root,
        'evaluate',config,machine['SftEnvironmentPrefix'],machine['RLinfRoot'],str(train),'',
        machine['BashSetupPath'],machine.get('SftReferenceDir','')]


def main(args):
    repo=Path(args.worktree).resolve();machine=read(args.machine);root=Path(machine['DataRoot'])
    directory=Path(args.directory).resolve();assert directory.is_relative_to(root.resolve()/'research')
    directory.mkdir(parents=True,exist_ok=True)
    lock=(directory/'workflow.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==args.commit
    config_sha=file_sha(repo/args.config);train=root/'runs'/f'job-{args.train_job}'
    state=read(directory/'workflow.json') if (directory/'workflow.json').exists() else dict(
        train_job=str(args.train_job),evaluation_job=None,phase='WAIT_TRAIN',target_commit=args.commit,
        config_sha256=config_sha,comparison_scope='within_native_runtime_only',pid=os.getpid(),
        controller_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        evaluation_name='sft-a-eval-'+str(args.train_job),training_retry_authorized=False,
        recovery_start_date=time.strftime('%Y-%m-%d'),
        evaluation=dict(episodes=80,physical_steps=32000,resets='3300-3319',arms=['native_unadapted','adapted']))
    journal=directory/'evaluation_submitted.json'
    if journal.exists():
        submitted=read(journal);assert submitted['commit']==args.commit and submitted['config_sha256']==config_sha
        state['evaluation_job']=submitted['job_id']
    def save():
        state['updated_unix']=time.time();state['pid']=os.getpid()
        current=root/'runs'/('job-'+str(state.get('evaluation_job') or args.train_job))
        state['current_metrics']=read(current/'latest.json') if (current/'latest.json').exists() else {}
        atomic(directory/'workflow.json',state)
        for job in [str(args.train_job),state.get('evaluation_job')]:
            if job:atomic(root/'runs'/('job-'+str(job))/'SFT_WORKFLOW.json',state)
    try:
        while True:
            try:
                if not state.get('evaluation_job'):
                    status=job_state(args.train_job);state['training_slurm_state']=status
                    if status not in ['COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED']:
                        state['phase']='WAIT_TRAIN';save();time.sleep(60);continue
                    if status!='COMPLETED':
                        state.update(phase='FAILED_TRAIN',error='Training did not complete; no automatic retraining or evaluation');save();return
                    passed=read(train/'SFT_TRAINING_COMPLETE.json');assert passed['passed']
                    assert passed['updates'] in [500,3000]
                    run=read(train/'run.json');assert run['commit']==args.commit and run['config_sha256']==config_sha
                    assert read(train/'comparison_scope.json')['comparison_scope']=='within_native_runtime_only'
                    state['training_completion']=passed
                    existing=recover_job(state['evaluation_name'],state['recovery_start_date'])
                    if existing:
                        state.update(evaluation_job=existing,phase='EVALUATING',recovered_submission=True,submission_uncertain=False);save();continue
                    if state.get('submission_uncertain'):
                        state.update(phase='SUBMISSION_UNCERTAIN',error='No duplicate submission after uncertain scheduler response');save();return
                    command=evaluation_command(machine,repo,args.train_job,state['evaluation_name'],args.config)
                    state.update(phase='SUBMITTING_EVALUATION',submission_uncertain=True);save()
                    submit_env=dict(os.environ);submit_env.pop('CUDA_VISIBLE_DEVICES',None)
                    submitted=subprocess.run(command,cwd=repo,env=submit_env,capture_output=True,text=True,timeout=60)
                    if submitted.returncode:
                        if any(k in submitted.stderr for k in ['MaxSubmitJob','MaxJobCount','job submit limit']):
                            state.update(phase='WAIT_SUBMISSION_CAPACITY',submission_uncertain=False,last_submit_error=submitted.stderr.strip());save();time.sleep(60);continue
                        state.update(phase='SUBMIT_REJECTED',submission_uncertain=False,error=submitted.stderr.strip());save();return
                    job=submitted.stdout.strip().split(';')[0];assert job.isdigit(),submitted.stdout
                    atomic(directory/'evaluation_submitted.json',dict(job_id=job,command=command,commit=args.commit,config_sha256=config_sha))
                    state.update(evaluation_job=job,phase='EVALUATING',submission_uncertain=False);save()
                else:
                    status=job_state(state['evaluation_job']);state['evaluation_slurm_state']=status
                    if status=='COMPLETED':
                        flag=read(root/'runs'/('job-'+state['evaluation_job'])/'SFT_EVAL_COMPLETE.json');assert flag['passed']
                        state.update(phase='COMPLETE',evaluation_completion=flag);save();return
                    if status in ['FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED']:
                        state.update(phase='FAILED_EVALUATION',error='Evaluation needs review; no automatic repeat');save();return
                    state['phase']='EVALUATING';save();time.sleep(60)
            except (subprocess.TimeoutExpired,RuntimeError) as error:
                state['last_scheduler_error']=str(error);save()
                if state.get('submission_uncertain'):
                    state['phase']='RECONCILING_SUBMISSION'
                time.sleep(60)
    except BaseException as error:
        state.update(phase='WORKFLOW_ERROR',error=str(error),traceback=traceback.format_exc());save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worktree',required=True);p.add_argument('--machine',required=True)
    p.add_argument('--commit',required=True);p.add_argument('--train-job',type=int,required=True)
    p.add_argument('--directory',required=True);p.add_argument('--config',default='configs/sft_a_v1.json')
    main(p.parse_args())
