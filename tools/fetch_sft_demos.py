"""Fetch only the two pinned official demonstrations; verify published LFS SHA."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def download(spec, item, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    name = Path(item['path']).name
    target = output / name
    expected = item['lfs']['oid']
    def digest(path):
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
                h.update(block)
        return h.hexdigest()
    if target.exists():
        assert target.stat().st_size == item['size'] and digest(target) == expected
        return str(target)
    temporary = target.with_suffix('.hdf5.part')
    url = f"https://huggingface.co/datasets/{spec['repo']}/resolve/{spec['revision']}/{item['path']}?download=true"
    for attempt in range(3):
        try:
            start = temporary.stat().st_size if temporary.exists() else 0
            request = urllib.request.Request(url, headers={'Range': f'bytes={start}-'} if start else {})
            with urllib.request.urlopen(request, timeout=60) as response:
                if start and response.status != 206:
                    start = 0
                with temporary.open('ab' if start else 'wb') as stream:
                    current = start
                    reported = start
                    while block := response.read(8 * 1024 ** 2):
                        stream.write(block)
                        current += len(block)
                        if current - reported >= 64 * 1024 ** 2:
                            print(json.dumps(dict(file=name, bytes=current, total=item['size'])), flush=True)
                            reported = current
            assert temporary.stat().st_size == item['size']
            assert digest(temporary) == expected
            temporary.replace(target)
            print(json.dumps(dict(file=name, verified=True, sha256=expected)), flush=True)
            return str(target)
        except Exception as error:
            print(json.dumps(dict(file=name, attempt=attempt + 1, error=str(error))), flush=True)
            if attempt == 2:
                raise
            time.sleep(5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    spec = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda item: download(spec, item, args.output), spec['files']))
    print(json.dumps(dict(passed=True, files=results)), flush=True)
