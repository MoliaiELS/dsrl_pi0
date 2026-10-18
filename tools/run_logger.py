"""Durable local metrics, media, run metadata and process resource samples."""
import csv
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import threading
import time

import numpy as np


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')
    temp.replace(path)


class RunLogger:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.latest = {}
        self.start = time.monotonic()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.monitor, daemon=True)
        self.thread.start()

    def log(self, values, step=None, **kwargs):
        scalars = {}
        for key, value in values.items():
            if hasattr(value, '_path') and value._path:
                source = Path(value._path)
                media = self.directory / 'media'
                media.mkdir(exist_ok=True)
                destination = media / (f'{step}-{key.replace("/", "_")}' + source.suffix)
                shutil.copy2(source, destination)
                scalars[key] = str(destination)
            elif isinstance(value, (str, bool, int, float)) or value is None:
                scalars[key] = value
            else:
                array = np.asarray(value)
                scalars[key] = array.item() if array.ndim == 0 else array.tolist()
        event = {'time_unix': time.time(), 'wall_seconds': time.monotonic() - self.start,
                 'gradient_updates': step, **scalars}
        with (self.directory / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(event, allow_nan=False) + '\n')
        self.latest.update(event)
        write_json(self.directory / 'latest.json', self.latest)
        print('METRIC', json.dumps(event), flush=True)

    def log_histogram(self, name, values, step):
        array = np.asarray(values)
        self.log({name + '/mean': float(array.mean()),
                  name + '/min': float(array.min()), name + '/max': float(array.max())}, step)

    def monitor(self):
        while not self.stop.is_set():
            event = {'time_unix': time.time(), 'pid': os.getpid(),
                     'process_max_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
            try:
                output = subprocess.check_output([
                    'nvidia-smi', '--query-compute-apps=pid,used_gpu_memory,gpu_uuid',
                    '--format=csv,noheader,nounits'], text=True, timeout=10)
                owned = [row for row in csv.reader(output.splitlines())
                         if row and row[0].strip() == str(os.getpid())]
                event['gpus'] = []
                for row in owned:
                    uuid = row[2].strip()
                    gpu = subprocess.check_output([
                        'nvidia-smi', '--id=' + uuid,
                        '--query-gpu=utilization.gpu,memory.total,power.draw,temperature.gpu',
                        '--format=csv,noheader,nounits'], text=True, timeout=10).strip()
                    event['gpus'].append({'uuid': uuid, 'process_memory_mib': row[1].strip(),
                                          'util_totalmem_power_temperature': gpu})
                event['cpu_percent_rss_kib'] = subprocess.check_output(
                    ['ps', '-p', str(os.getpid()), '-o', '%cpu=,rss='], text=True).strip()
            except Exception as error:
                event['monitor_error'] = str(error)
            with (self.directory / 'resources.jsonl').open('a') as stream:
                stream.write(json.dumps(event) + '\n')
            self.stop.wait(30)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=12)
