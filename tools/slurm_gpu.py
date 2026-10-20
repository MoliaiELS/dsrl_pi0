"""Bind this cluster's single-GPU Slurm allocation by verified device identity.

GRES uses File=/dev/nvidia[0-3]. Its allocation index therefore identifies the
device-file minor, which need not equal CUDA's numeric enumeration. No other
GPU is selected and no unmasked CUDA enumeration is performed.
"""
import argparse
import csv
import ctypes
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys


def check_gres(text, hostname):
    for line in text.splitlines():
        fields = dict(token.split('=', 1) for token in shlex.split(line, comments=True) if '=' in token)
        if (fields.get('Name') == 'gpu'
                and fnmatch.fnmatchcase(hostname, fields.get('NodeName', ''))):
            if fields.get('File') == '/dev/nvidia[0-3]':
                return
            raise ValueError('Unsupported GRES device-file mapping: ' + line)
    raise ValueError('No supported GPU GRES mapping for ' + hostname)


def select_gpu(environ, inventory):
    allocation = environ.get('SLURM_JOB_GPUS', '')
    visible = environ.get('CUDA_VISIBLE_DEVICES', '')
    if not environ.get('SLURM_JOB_ID', '').isdigit() or not allocation.isdigit():
        raise ValueError('An explicit single-GPU Slurm allocation is required')
    if not visible or ',' in visible:
        raise ValueError('Expected a single scheduler-visible GPU')
    matches = [gpu for gpu in inventory if gpu['minor_number'] == int(allocation)]
    if len(matches) != 1:
        raise ValueError('Allocation does not resolve to exactly one device file')
    selected = matches[0]
    if not (visible.isdigit() or visible == selected['uuid']):
        raise ValueError('Existing visibility conflicts with the allocated UUID')
    if visible.isdigit() and int(visible) not in (0, int(allocation)):
        raise ValueError('Unexpected numeric scheduler visibility: ' + visible)
    return selected


def inventory():
    output = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,pci.bus_id,memory.free,memory.total',
        '--format=csv,noheader,nounits'], text=True, timeout=15)
    nvml = ctypes.CDLL('libnvidia-ml.so.1')
    assert nvml.nvmlInit_v2() == 0, 'NVML initialization failed'
    devices = []
    try:
        for row in csv.reader(output.splitlines()):
            index, uuid, pci, free, total = [value.strip() for value in row]
            handle = ctypes.c_void_p()
            assert nvml.nvmlDeviceGetHandleByUUID(uuid.encode(), ctypes.byref(handle)) == 0
            minor = ctypes.c_uint()
            assert nvml.nvmlDeviceGetMinorNumber(handle, ctypes.byref(minor)) == 0
            devices.append({'nvml_index': int(index), 'uuid': uuid, 'pci_bus_id': pci,
                            'minor_number': minor.value, 'free_memory_mib': int(free),
                            'total_memory_mib': int(total)})
    finally:
        nvml.nvmlShutdown()
    return devices


def verify_cuda(pci):
    cuda = ctypes.CDLL('libcuda.so.1')
    rc = cuda.cuInit(0)
    if rc != 0:
        raise RuntimeError('Allocated UUID CUDA initialization failed: ' + str(rc))
    count = ctypes.c_int()
    assert cuda.cuDeviceGetCount(ctypes.byref(count)) == 0 and count.value == 1
    device = ctypes.c_int()
    assert cuda.cuDeviceGet(ctypes.byref(device), 0) == 0
    bus = ctypes.create_string_buffer(32)
    assert cuda.cuDeviceGetPCIBusId(bus, len(bus), device) == 0
    actual = bus.value.decode()
    assert actual.lower().lstrip('0') == pci.lower().lstrip('0'), (actual, pci)
    print(json.dumps({'device_count': count.value, 'pci_bus_id': actual}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path)
    ap.add_argument('--verify-pci')
    args = ap.parse_args()
    if args.verify_pci:
        verify_cuda(args.verify_pci)
        return
    if args.output is None:
        ap.error('--output is required')
    args.output.mkdir(parents=True, exist_ok=True)
    record = {'job_id': os.environ.get('SLURM_JOB_ID'), 'hostname': socket.gethostname(),
              'slurm_job_gpus': os.environ.get('SLURM_JOB_GPUS'),
              'scheduler_cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
    try:
        conf = Path(os.environ.get('SLURM_CONF', '/usr/local/etc/slurm.conf')).with_name('gres.conf')
        text = conf.read_text()
        check_gres(text, socket.gethostname())
        selected = select_gpu(os.environ, inventory())
        record.update(selected_gpu=selected, gres_config=str(conf),
                      gres_config_sha256=hashlib.sha256(text.encode()).hexdigest())
        child_env = os.environ.copy()
        child_env['CUDA_VISIBLE_DEVICES'] = selected['uuid']
        checked = subprocess.run([sys.executable, '-m', 'tools.slurm_gpu', '--verify-pci', selected['pci_bus_id']],
                                 env=child_env, capture_output=True, text=True, timeout=30)
        if checked.returncode:
            raise RuntimeError('Allocated-device verification failed: ' + checked.stderr[-3000:])
        record.update(cuda_verification=json.loads(checked.stdout), passed=True,
                      cuda_visible_devices=selected['uuid'])
        exports = {'CUDA_VISIBLE_DEVICES': selected['uuid'],
                   'DSRL_SCHEDULER_CUDA_VISIBLE_DEVICES': record['scheduler_cuda_visible_devices']}
        (args.output / 'gpu_binding.sh').write_text(''.join(
            'export ' + key + '=' + shlex.quote(value) + '\n' for key, value in exports.items()))
        print('GPU_BINDING', json.dumps(record), flush=True)
    except Exception as error:
        record.update(passed=False, type=type(error).__name__, error=str(error))
        (args.output / 'FAILED.json').write_text(json.dumps({**record, 'phase': 'gpu_binding'}, indent=2))
        raise
    finally:
        (args.output / 'gpu_binding.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
