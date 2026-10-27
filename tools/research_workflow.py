"""One-shot login-node handoff; launches only the already authorized first_batch."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

repo=Path(sys.argv[1]);gate_id=sys.argv[2]
cfg=json.loads((repo/'tools/remote.3090.json').read_text(encoding='utf-8-sig'))
data=Path(cfg['DataRoot']);gate=data/'runs'/('job-'+gate_id)
folder=data/'runs'/('research-workflow-'+gate_id);folder.mkdir(exist_ok=True)
def save(name,value):
    f=folder/name;tmp=f.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(f)
if (folder/'submitted.json').exists():sys.exit(0)
expected=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
job_name='dsrl-research-'+gate_id
for attempt in range(720):
    queue=subprocess.check_output(['squeue','-u',os.environ['USER'],'-h','-o','%i|%j'],text=True)
    active=[line.split('|',1) for line in queue.splitlines() if '|' in line]
    existing=[ident for ident,name in active if name==job_name]
    if existing:
        save('submitted.json',dict(job_id=existing[0],recovered=True,commit=expected));break
    if (gate/'FAILED.json').exists():
        save('BLOCKED.json',dict(reason='E0 failed; no full evaluation submitted',gate=gate_id));sys.exit(1)
    if (gate/'SUCCESS.json').exists() and not any(ident==gate_id for ident,_ in active):
        run=json.loads((gate/'run.json').read_text());passed=json.loads((gate/'E0_PASS.json').read_text())
        digest=hashlib.sha256((repo/'configs/research_protocol_v1.json').read_bytes()).hexdigest()
        assert run['commit']==expected and run['protocol_file_sha256']==digest and passed['passed']
        assert json.loads((gate/'SUCCESS.json').read_text())['passed']
        # Accounting DB is reachable from login but not the compute nodes.
        shared=data/'research'/'dsrl-vla-language-stage-v1-20261027'/'budget.json'
        budget=json.loads(shared.read_text())
        accounting={}
        for session in budget['sessions']:
            result=subprocess.run(['sacct','-n','-P','-j',session['job_id'],'--format=JobIDRaw,ElapsedRaw'],capture_output=True,text=True)
            for row in result.stdout.splitlines():
                fields=row.split('|')
                if fields[0].strip()==session['job_id']:
                    seconds=float(fields[1]);session['seconds']=max(session['seconds'],seconds)
                    accounting[session['job_id']]=dict(elapsed_seconds=seconds)
        budget['allocated_gpu_hours_lower_bound']=sum(s['seconds'] for s in budget['sessions'])/3600
        for path in [shared,gate/'budget.json']:
            temp=path.with_suffix('.accounting.tmp');temp.write_text(json.dumps(budget,sort_keys=True,separators=(',',':')));temp.replace(path)
        save('allocation_accounting.json',accounting)
        if any(name.startswith('dsrl-research') for _,name in active):
            time.sleep(30);continue
        slurm=cfg['Slurm']
        command=['sbatch','--parsable','--partition='+slurm['Partition'],'--account='+slurm['Account'],
                 '--qos='+slurm['Qos'],'--gres='+slurm['Gres'],'--cpus-per-task='+str(slurm['Cpus']),
                 '--exclude='+slurm['Exclude'],'--time='+slurm['TrainTime'],'--job-name='+job_name,
                 '--output='+str(data/'logs/slurm/%j.log'),str(repo/'tools/job.sh'),str(repo),cfg['CondaRoot'],
                 cfg['CondaEnv'],str(data),'research','configs/research_protocol_v1.json','0','',gate_id]
        submitted=subprocess.run(command,capture_output=True,text=True)
        save('latest_submission_attempt.json',dict(attempt=attempt,returncode=submitted.returncode,stdout=submitted.stdout,stderr=submitted.stderr))
        if submitted.returncode==0:
            ident=submitted.stdout.strip().split(';')[0]
            assert ident.isdigit(),submitted.stdout
            save('submitted.json',dict(job_id=ident,commit=expected,gate_job=gate_id,protocol_file_sha256=digest,
                                       result=str(data/'runs'/('job-'+ident)),command=command))
            print('FULL_JOB='+ident,flush=True);break
        if not any(s in submitted.stderr for s in ['QOSMaxSubmitJobPerUserLimit','AssocMaxSubmitJobLimit','Socket timed out']):
            save('BLOCKED.json',dict(reason='sbatch rejected',stderr=submitted.stderr));sys.exit(1)
    save('waiting.json',dict(gate=gate_id,attempt=attempt,time=time.time()))
    time.sleep(30)
else:
    save('BLOCKED.json',dict(reason='E0 handoff timeout'));sys.exit(1)
