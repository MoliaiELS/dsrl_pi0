"""CPU-only accounting, saved-trajectory closure, report and portable archive."""
import argparse
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import zipfile

from tools.research_core import atomic, file_sha, read


def report_html(output):
    output=Path(output)
    text=(output/'REPORT.md').read_text(encoding='utf-8')
    try:
        import markdown
        body=markdown.markdown(text,extensions=['tables','fenced_code'])
    except ImportError:
        try:
            from markdown_it import MarkdownIt
            body=MarkdownIt().enable('table').render(text)
        except ImportError:
            body='<pre>'+html.escape(text)+'</pre>'
    pattern=r'<a href="([^"<>]+\.mp4)">([^<>]+)</a>'
    body=re.sub(pattern,lambda m:'<details><summary>'+m[2]+'</summary><video controls preload="none" src="'+m[1]+'"></video></details>',body)
    style='body{font:16px/1.65 "Microsoft YaHei",Segoe UI,sans-serif;max-width:1300px;margin:40px auto;padding:0 24px;color:#172431}h1,h2{color:#11364c}h2{margin-top:42px;border-bottom:2px solid #dbe6ec}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #d9e1e5;padding:7px;vertical-align:top}th{background:#eef4f6}img{max-width:100%;height:auto}video{width:340px;max-width:100%;margin:8px 0}pre{background:#f1f4f6;padding:15px;overflow:auto;white-space:pre-wrap;word-break:break-word}details{margin:4px}summary{cursor:pointer;color:#176e8c}'
    style+='td,th{overflow-wrap:anywhere}'
    (output/'REPORT.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>DSRL language and stage diagnostic</title><style>'+style+'</style><body>'+body+'</body></html>',encoding='utf-8')


def finish(output,protocol,archive=True):
    output=Path(output).resolve();p=read(protocol)
    assert read(output/'SUCCESS.json')['passed'], 'Evaluation has not finished successfully'
    if (output/'POSTPROCESS_COMPLETE.json').exists():return read(output/'POSTPROCESS_COMPLETE.json')
    budget=read(output/'budget.json');accounting={}
    for session in budget['sessions']:
        result=subprocess.run(['sacct','-n','-P','-j',session['job_id'],'--format=JobIDRaw,ElapsedRaw,State,ExitCode'],capture_output=True,text=True)
        for row in result.stdout.splitlines():
            columns=row.split('|')
            if columns[0].strip()==session['job_id']:
                session['seconds']=max(session['seconds'],float(columns[1]));accounting[session['job_id']]=dict(elapsed_seconds=float(columns[1]),state=columns[2],exit_code=columns[3])
    budget['allocated_gpu_hours_lower_bound']=sum(s['seconds'] for s in budget['sessions'])/3600
    atomic(output/'allocation_accounting.json',accounting)
    atomic(output/'budget.json',budget)
    # Keep the shared ledger consistent after the GPU process has exited.
    shared=Path(os.environ['DSRL_DATA_ROOT'])/'research'/p['protocol_id']/'budget.json'
    atomic(shared,budget)
    for name in ['REPORT.md','decision.json','paired_cluster_statistics.json','summary_by_condition.csv']:
        source=output/name
        saved=output/(source.stem+'.evaluation_code_008a0a2'+source.suffix)
        if source.exists() and not saved.exists():shutil.copy2(source,saved)
    from tools.research_audit import audit
    closure=audit(output,p,require_complete=True,verify_weights=True)
    from tools.research_stats import generate
    decision=generate(output,p)
    report_html(output)
    archive_path=output.parent/(output.name+'-research-evidence.zip')
    if archive:
        temporary=archive_path.with_suffix('.zip.tmp')
        with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
            for path in sorted(output.rglob('*')):
                if path.is_file() and not path.name.endswith('.tmp'):
                    z.write(path,Path(output.name)/path.relative_to(output))
        temporary.replace(archive_path)
    result=dict(passed=True,execution_status=decision['execution_status'],scientific_decision=decision['scientific_decision'],
        checked_valid_cases=closure['checked_valid_cases'],report=str(output/'REPORT.html'),
        archive=str(archive_path) if archive else None,archive_sha256=file_sha(archive_path) if archive else None,
        added_simulation_steps=0,added_vla_calls=0,postprocess_code_sha256=file_sha(__file__))
    atomic(output/'POSTPROCESS_COMPLETE.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',required=True);parser.add_argument('--protocol',required=True)
    parser.add_argument('--wait-job');parser.add_argument('--no-archive',action='store_true')
    args=parser.parse_args();output=Path(args.input)
    if args.wait_job:
        for _ in range(1442):
            if (output/'FAILED.json').exists() and not (output/'SUCCESS.json').exists():raise RuntimeError('Evaluation failed; automatic postprocess stopped')
            queue=subprocess.check_output(['squeue','-u',os.environ['USER'],'-h','-o','%i'],text=True)
            if (output/'SUCCESS.json').exists() and args.wait_job not in queue.split():break
            time.sleep(60)
        else:raise TimeoutError('Evaluation did not finish in postprocess watch window')
    print(json.dumps(finish(output,args.protocol,archive=not args.no_archive),indent=2))
