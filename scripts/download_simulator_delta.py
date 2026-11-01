"""Create a fresh verified Agent snapshot using unchanged immutable donors.

Only source reads occur remotely. Each changed file and its checksum are read
from the same bytes; a separate new local snapshot preserves all previous data.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile

from verify_simulator_snapshot import REMOTE

FETCH = r'''
import io,sys,json,hashlib,tarfile,pathlib
root=pathlib.Path('/opt/render/project/src/pipeline/data')
paths=json.load(sys.stdin);manifest={}
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as out:
 for name in paths:
  p=root/name
  if not p.resolve().is_relative_to(root):raise ValueError('Invalid path')
  data=p.read_bytes();manifest[name]=hashlib.sha256(data).hexdigest()
  member=tarfile.TarInfo(name);member.size=len(data);member.mtime=int(p.stat().st_mtime)
  out.addfile(member,io.BytesIO(data))
 data=json.dumps(manifest).encode();member=tarfile.TarInfo('__delta_manifest__.json');member.size=len(data);out.addfile(member,io.BytesIO(data))
'''


def ssh(code, data=None):
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','UpdateHostKeys=no','srv-d6qm9j9aae7s739j7kkg@ssh.oregon.render.com','python -c '+shlex.quote(code)],input=data,capture_output=True,timeout=300)
    if result.returncode:raise RuntimeError(result.stderr.decode(errors='replace'))
    return result.stdout


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base',type=Path,required=True);parser.add_argument('--overlay',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise ValueError('Use a new output directory')
    base_manifest=json.loads((args.base/'simulator_remote_manifest.json').read_text());verification=json.loads((args.overlay/'verification.json').read_text())
    if not verification.get('complete'):raise ValueError('Base overlay is not verified')
    remote=json.loads(ssh(REMOTE));changed=[p for p,h in remote.items() if base_manifest.get(p)!=h]
    print(f'Remote files: {len(remote)}; new/changed files: {len(changed)}',flush=True)
    archive=ssh(FETCH,json.dumps(changed).encode())
    args.output.mkdir(parents=True)
    updated={}
    with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz') as package:
        delta=json.load(package.extractfile('__delta_manifest__.json'))
        for name in changed:
            dest=(args.output/name).resolve()
            if not dest.is_relative_to(args.output.resolve()):raise ValueError('Unsafe path')
            member=package.getmember(name);data=package.extractfile(member).read()
            if hashlib.sha256(data).hexdigest()!=delta[name]:raise ValueError('Invalid changed file')
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data);os.utime(dest,(member.mtime,member.mtime));updated[name]=delta[name]
    remote.update(updated)
    unchanged=[name for name in remote if name not in updated]
    def copy_one(name):
        donor=args.overlay/name
        if not donor.exists():donor=args.base/name
        data=donor.read_bytes()
        if hashlib.sha256(data).hexdigest()!=remote[name]:raise ValueError(f'Donor checksum mismatch: {name}')
        dest=args.output/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data);stat=donor.stat();os.utime(dest,(stat.st_atime,stat.st_mtime))
    with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(copy_one,unchanged))
    report={'complete':True,'captured_at':datetime.now(timezone.utc).isoformat(),'files':len(remote),'changed':len(changed),'unchanged_verified':len(unchanged),'base':str(args.base.resolve()),'consistency':'File list frozen at inventory; changed bytes and checksums captured together. Live source is not transactional.'}
    (args.output/'integrity_manifest.json').write_text(json.dumps(remote),encoding='utf-8');(args.output/'integrity_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
