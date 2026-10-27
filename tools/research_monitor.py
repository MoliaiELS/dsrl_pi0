"""Detached CPU monitoring and completion of the frozen research batch."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile

from tools.research_core import atomic, file_sha, read


def watch(output, protocol, postprocess_pid):
    output = Path(output).resolve()
    guard = (output / '.research_monitor.lock').open('a')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(pid=os.getpid(), cpu_only=True, simulation_steps_added=0, vla_calls_added=0,
                 code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())

    def publish(phase, **values):
        state.update(phase=phase, checked_unix=time.time(), **values)
        atomic(output / 'AUTO_WORKFLOW.json', state)
        print(json.dumps(state), flush=True)

    def postprocess_alive():
        cmd = Path('/proc') / str(postprocess_pid) / 'cmdline'
        return cmd.exists() and b'tools.research_postprocess' in cmd.read_bytes()

    try:
        from tools.research_audit import audit
        last_count = -100
        for _ in range(1442):
            success = (output / 'SUCCESS.json').exists()
            if (output / 'FAILED.json').exists() and not success:
                publish('EVALUATION_FAILED', error=read(output / 'FAILED.json'))
                return
            count = len(list((output / 'cases').glob('*.json')))
            if not success and count - last_count >= 100:
                publish('AUDITING_SAVED_TRAJECTORIES', completed_cases=count)
                result = audit(output, read(protocol))
                last_count = result['checked_valid_cases']
            if (output / 'POSTPROCESS_COMPLETE.json').exists():
                publish('EXTRA_CPU_CLOSURE', completed_cases=count)
                from tools.research_diagnostic_summary import summarize
                summarize(output)
                subprocess.run([sys.executable, str(output / 'engineering/account_inference.py'),
                                '--input', str(output)], check=True)
                archive = output.parent / (output.name + '-research-closure.zip')
                temporary = archive.with_suffix('.zip.tmp')
                with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as z:
                    for path in sorted(output.rglob('*')):
                        if path.is_file() and not path.name.endswith(('.tmp', '.lock')):
                            z.write(path, Path(output.name) / path.relative_to(output))
                temporary.replace(archive)
                result = dict(passed=True, archive=str(archive), archive_sha256=file_sha(archive),
                              final_trajectory_audit=read(output / 'closure_check.json')['passed'],
                              reviewed_narrative_pending_next_session=True, added_environment_steps=0, added_vla_calls=0)
                atomic(output / 'EXTRA_POSTPROCESS_COMPLETE.json', result)
                publish('COMPLETE', completed_cases=count, audit_cases=read(output / 'closure_check.json')['checked_valid_cases'],
                        archive=str(archive), human_report_review='pending next conversation')
                return
            if success and not postprocess_alive():
                queue = subprocess.check_output(['squeue', '-u', os.environ['USER'], '-h', '-o', '%i'], text=True)
                if output.name.removeprefix('job-') not in queue.split():
                    publish('RECOVERING_CPU_POSTPROCESS', completed_cases=count)
                    from tools.research_postprocess import finish
                    finish(output, protocol)
                    continue
            audit_result = read(output / 'trajectory_audit.json') if (output / 'trajectory_audit.json').exists() else {}
            publish('WAITING_FINAL_POSTPROCESS' if success else 'MONITORING_EVALUATION', completed_cases=count,
                    audit_cases=audit_result.get('checked_valid_cases'), original_postprocess_alive=postprocess_alive())
            time.sleep(60)
        publish('WATCH_TIMEOUT')
    except BaseException as error:
        publish('CPU_WORKFLOW_ERROR', error=str(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--postprocess-pid', type=int, required=True)
    args = parser.parse_args()
    watch(args.input, args.protocol, args.postprocess_pid)
