"""Download the public LIBERO checkpoint without loading any model."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import requests

SOURCE = os.environ.get('DSRL_ASSET_HTTP_SOURCE', 'https://openpi-assets.s3.us-west-2.amazonaws.com')
PREFIX = 'checkpoints/pi0_libero/'


def main():
    data_root = Path(os.environ['DSRL_DATA_ROOT'])
    target = data_root / 'checkpoints/pi0_libero'
    if (target / 'download_manifest.json').exists():
        print('Already verified:', target, flush=True)
        return
    stage = target.with_name(target.name + '.partial')
    stage.mkdir(parents=True, exist_ok=True)
    entries = []
    params = {'list-type': '2', 'prefix': PREFIX, 'max-keys': 1000}
    while True:
        for attempt in range(4):
            try:
                response = requests.get(SOURCE, params=params, timeout=(15, 120))
                response.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        doc = ET.fromstring(response.content)
        ns = {'s': 'http://s3.amazonaws.com/doc/2006-03-01/'}
        for item in doc.findall('s:Contents', ns):
            key = item.findtext('s:Key', namespaces=ns)
            if key.endswith('/'):
                continue
            entries.append({'key': key,
                            'size': int(item.findtext('s:Size', namespaces=ns)),
                            'etag': item.findtext('s:ETag', namespaces=ns).strip('"')})
        if doc.findtext('s:IsTruncated', namespaces=ns) != 'true':
            break
        params['continuation-token'] = doc.findtext('s:NextContinuationToken', namespaces=ns)
    print('Files:', len(entries), 'Bytes:', sum(x['size'] for x in entries), flush=True)

    def download(entry):
        relative = entry['key'][len(PREFIX):]
        path = stage / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(4):
            try:
                if not path.exists() or path.stat().st_size != entry['size']:
                    part = path.with_name(path.name + '.part')
                    offset = part.stat().st_size if part.exists() else 0
                    headers = {'Range': f'bytes={offset}-'} if offset else {}
                    with requests.get(SOURCE + '/' + entry['key'], headers=headers,
                                      stream=True, timeout=(20, 120)) as response:
                        response.raise_for_status()
                        mode = 'ab' if offset and response.status_code == 206 else 'wb'
                        with part.open(mode) as stream:
                            for chunk in response.iter_content(8 * 1024 * 1024):
                                stream.write(chunk)
                    assert part.stat().st_size == entry['size'], relative
                    part.replace(path)
                sha, md5 = hashlib.sha256(), hashlib.md5()
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                        sha.update(block)
                        md5.update(block)
                if len(entry['etag']) == 32:
                    assert md5.hexdigest() == entry['etag'], f'ETag mismatch: {relative}'
                print('VERIFIED', relative, entry['size'], flush=True)
                return dict(entry, path=relative, sha256=sha.hexdigest())
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        raise AssertionError(relative)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        verified = list(pool.map(download, entries))
    (stage / 'download_manifest.json').write_text(json.dumps({
        'source': 's3://openpi-assets/' + PREFIX.rstrip('/'),
        'files': verified, 'total_bytes': sum(x['size'] for x in verified),
        'verified_at_unix': time.time(),
    }, indent=2), encoding='utf-8')
    if target.exists():
        raise FileExistsError(f'Refusing to replace existing {target}')
    stage.rename(target)
    print('DOWNLOAD_COMPLETE', target, flush=True)


if __name__ == '__main__':
    main()
