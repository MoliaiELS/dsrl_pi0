"""Fetch byte-identical official GCS mirrors; verify against canonical S3 ETags."""
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import requests

MIB = 1024 * 1024
PREFIX = 'checkpoints/pi0_libero/'
MIRROR = 'https://openpi-assets.storage.googleapis.com/'


def main():
    root = Path(os.environ['DSRL_DATA_ROOT'])
    target = root / 'checkpoints/pi0_libero'
    if (target / 'download_manifest.json').exists():
        print('Already verified', target, flush=True)
        return
    stage = root / 'checkpoints/pi0_libero-mirror.partial'
    stage.mkdir(parents=True, exist_ok=True)
    chunk_root = root / 'cache/model_chunks/pi0_libero'
    chunk_root.mkdir(parents=True, exist_ok=True)
    listing = root / 'logs/assets/pi0-libero-s3-list.xml'
    doc = ET.fromstring(listing.read_bytes())
    ns = {'s': 'http://s3.amazonaws.com/doc/2006-03-01/'}
    assert doc.findtext('s:IsTruncated', namespaces=ns) == 'false'
    entries = []
    tasks = []
    for item in doc.findall('s:Contents', ns):
        key = item.findtext('s:Key', namespaces=ns)
        size = int(item.findtext('s:Size', namespaces=ns))
        etag = item.findtext('s:ETag', namespaces=ns).strip('"')
        assert key.startswith(PREFIX) and '..' not in key
        entry = {'key': key, 'path': key[len(PREFIX):], 'size': size, 'etag': etag}
        entries.append(entry)
        identifier = hashlib.sha256(key.encode()).hexdigest()
        entry['chunks'] = []
        for start in range(0, size, 64*MIB):
            end = min(size, start + 64*MIB)-1
            path = chunk_root / f'{identifier}-{start}'
            entry['chunks'].append(str(path))
            tasks.append((entry, start, end, path))
    print('DOWNLOAD_START', len(entries), 'files', sum(x['size'] for x in entries),
          'bytes', len(tasks), 'chunks', flush=True)

    def fetch(task):
        entry, start, end, path = task
        if path.exists() and path.stat().st_size == end-start+1:
            return
        for attempt in range(5):
            try:
                headers = {'Range': f'bytes={start}-{end}'}
                with requests.get(MIRROR+entry['key'], headers=headers, stream=True,
                                  timeout=(15, 180)) as response:
                    response.raise_for_status()
                    assert response.status_code == 206, response.status_code
                    assert response.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
                    assert response.headers['x-goog-meta-x-goog-source-etag'].strip('"') == entry['etag']
                    part = path.with_suffix('.part')
                    with part.open('wb') as stream:
                        for block in response.iter_content(MIB):
                            stream.write(block)
                    assert part.stat().st_size == end-start+1
                    part.replace(path)
                    return
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2**attempt)

    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        for count, _ in enumerate(pool.map(fetch, tasks), 1):
            if count % 8 == 0:
                print('CHUNKS', count, '/', len(tasks), 'SECONDS', round(time.monotonic()-started), flush=True)
    for entry in entries:
        output = stage / entry['path']
        output.parent.mkdir(parents=True, exist_ok=True)
        sha, md5, multipart = hashlib.sha256(), hashlib.md5(), []
        multipart_count = int(entry['etag'].split('-')[1]) if '-' in entry['etag'] else 0
        part_size = next((n*MIB for n in [5,8,16,32,64,128,256]
                          if math.ceil(entry['size']/(n*MIB)) == multipart_count), None) if multipart_count else None
        part_md5, part_bytes = hashlib.md5(), 0
        with output.open('wb') as destination:
            for filename in entry['chunks']:
                with Path(filename).open('rb') as source:
                    for block in iter(lambda: source.read(MIB), b''):
                        destination.write(block)
                        sha.update(block)
                        md5.update(block)
                        if part_size:
                            pos = 0
                            while pos < len(block):
                                n = min(len(block)-pos, part_size-part_bytes)
                                part_md5.update(block[pos:pos+n]); part_bytes += n; pos += n
                                if part_bytes == part_size:
                                    multipart.append(part_md5.digest()); part_md5=hashlib.md5(); part_bytes=0
        if part_bytes:
            multipart.append(part_md5.digest())
        assert output.stat().st_size == entry['size']
        if not multipart_count:
            assert md5.hexdigest() == entry['etag'], entry['path']
        else:
            assert part_size, f'Unknown multipart layout: {entry}'
            actual = hashlib.md5(b''.join(multipart)).hexdigest()+f'-{len(multipart)}'
            assert actual == entry['etag'], (entry['path'], actual, entry['etag'])
        entry['sha256'] = sha.hexdigest()
        entry['md5'] = md5.hexdigest()
        del entry['chunks']
        print('VERIFIED_S3_ETAG', entry['path'], entry['size'], flush=True)
    (stage / 'download_manifest.json').write_text(json.dumps({
        'source': 's3://openpi-assets/checkpoints/pi0_libero', 'mirror': MIRROR,
        'canonical_listing_sha256': hashlib.sha256(listing.read_bytes()).hexdigest(),
        'files': entries, 'total_bytes': sum(x['size'] for x in entries),
    }, indent=2))
    assert not target.exists(), target
    stage.rename(target)
    print('DOWNLOAD_COMPLETE', target, 'SECONDS', time.monotonic()-started, flush=True)


if __name__ == '__main__':
    main()
