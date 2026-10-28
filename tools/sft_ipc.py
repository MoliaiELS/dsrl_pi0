"""Framed, pickle-free NumPy payloads across the old/new environment boundary."""
from io import BytesIO
import struct
import numpy as np


def write_all(stream,data):
    remaining=memoryview(data)
    while remaining:
        count=stream.write(remaining)
        if not count:raise BrokenPipeError('IPC write did not advance')
        remaining=remaining[count:]


def read_exact(stream,count):
    parts=[]
    while count:
        value=stream.read(count)
        if not value:raise EOFError('IPC stream closed')
        parts.append(value);count-=len(value)
    return b''.join(parts)


def send(stream,fields):
    arrays={k:np.asarray(v) for k,v in fields.items()}
    assert all(v.dtype.kind!='O' for v in arrays.values())
    buffer=BytesIO();np.savez(buffer,**arrays);payload=buffer.getvalue()
    assert len(payload)<=16*1024*1024
    write_all(stream,struct.pack('!Q',len(payload)));write_all(stream,payload)


def receive(stream):
    size=struct.unpack('!Q',read_exact(stream,8))[0]
    assert 0<size<=16*1024*1024
    with np.load(BytesIO(read_exact(stream,size)),allow_pickle=False) as archive:
        return {k:(archive[k].item() if archive[k].ndim==0 and archive[k].dtype.kind in ['U','S'] else archive[k]) for k in archive.files}
