"""Submit a saved formal job after a smoke test passes and quota is available.

Lightweight login-node launcher: no model, simulation or GPU work. A lock and
durable submission record prevent repeated launches from submitting duplicates.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def check_smoke(plan):
    source = Path(plan['data_root']) / 'runs' / ('job-' + str(plan['smoke_job_id']))
    if (source / 'FAILED.json').exists():
        raise RuntimeError('Smoke failed: ' + (source / 'FAILED.json').read_text())
    if not (source / 'SUCCESS.json').exists():
        return False
    run = json.loads((source / 'run.json').read_text())
    success = json.loads((source / 'SUCCESS.json').read_text())
    digest = hashlib.sha256((Path(plan['repo']) / plan['protocol']).read_bytes()).hexdigest()
    if not (success.get('passed') and run.get('mode') == 'outcome-smoke'
            and run.get('commit') == plan['commit'] and run.get('protocol_file_sha256') == digest):
        raise RuntimeError('Smoke success/version/protocol gate rejected the planned formal job')
    return True


def submit_command(plan):
    resources = plan['slurm']
    return ['sbatch', '--parsable', '--partition=' + resources['Partition'],
            '--account=' + resources['Account'], '--qos=' + resources['Qos'],
            '--gres=' + resources['Gres'], '--cpus-per-task=' + str(resources['Cpus']),
            '--exclude=' + resources['Exclude'], '--time=' + resources['EvalTime'],
            '--job-name=' + plan['formal_job_name'],
            '--output=' + plan['data_root'] + '/logs/slurm/%j.log',
            plan['repo'] + '/tools/job.sh', plan['repo'], plan['conda_root'], plan['conda_env'],
            plan['data_root'], 'outcome', plan['protocol'], str(plan['seed']), '', str(plan['smoke_job_id'])]


def main():
    plan_path = Path(sys.argv[1]).resolve()
    plan = json.loads(plan_path.read_text())
    folder = plan_path.parent
    lock = (folder / 'launcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    submission = folder / 'formal_submission.json'
    if submission.exists():
        return
    actual = subprocess.check_output(['git', '-C', plan['repo'], 'rev-parse', 'HEAD'], text=True).strip()
    if actual != plan['commit']:
        raise RuntimeError('Pinned worktree commit changed')
    started = time.monotonic()
    while time.monotonic() - started < 48 * 3600:
        if not check_smoke(plan):
            state = subprocess.run(['sacct', '-j', str(plan['smoke_job_id']), '-X', '-n',
                                    '--format=State', '-P'], capture_output=True, text=True, timeout=20)
            if any(token in state.stdout for token in ['FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY']):
                raise RuntimeError('Smoke did not succeed: ' + state.stdout.strip())
            status = {'phase': 'waiting_for_smoke', 'smoke_state': state.stdout.strip()}
        else:
            existing = subprocess.check_output(['squeue', '--me', '--name=' + plan['formal_job_name'],
                                                 '--noheader', '--format=%A'], text=True, timeout=20).split()
            if len(existing) > 1:
                raise RuntimeError('Multiple matching formal jobs already exist')
            if existing:
                write_json(submission, {'job_id': existing[0], 'recovered_existing_submission': True,
                                         'smoke_job_id': plan['smoke_job_id'], 'commit': plan['commit']})
                return
            command = submit_command(plan)
            result = subprocess.run(command, capture_output=True, text=True, timeout=45)
            if result.returncode == 0:
                job = result.stdout.strip().split(';')[0]
                if not job.isdigit():
                    raise RuntimeError('Unrecognized sbatch result: ' + result.stdout)
                write_json(submission, {'job_id': job, 'smoke_job_id': plan['smoke_job_id'],
                    'commit': plan['commit'], 'submitted_at_unix': time.time(),
                    'command': command, 'runtime_smoke_gate': True})
                print('FORMAL_JOB_ID=' + job, flush=True)
                return
            if not any(token in result.stderr for token in ['AssocMaxSubmitJobLimit', 'QOSMaxSubmitJob']):
                raise RuntimeError('Formal submission failed: ' + result.stderr)
            status = {'phase': 'waiting_for_submit_quota', 'last_submission_error': result.stderr.strip()}
        write_json(folder / 'launcher_status.json', {**status, 'time_unix': time.time(), 'pid': os.getpid()})
        print(json.dumps(status), flush=True)
        time.sleep(300)
    raise TimeoutError('Smoke/formal launcher exceeded its 48-hour deadline')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        folder = Path(sys.argv[1]).resolve().parent
        write_json(folder / 'STOPPED.json', {'type': type(error).__name__, 'error': str(error),
                                            'time_unix': time.time()})
        raise
