"""Complete a virtual snapshot in a separate overlay; never alter downloaded data."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("snapshot",type=Path);parser.add_argument("overlay",type=Path);args=parser.parse_args()
    report=json.loads((args.snapshot/'simulator_integrity_verification.json').read_text());manifest=json.loads((args.snapshot/'simulator_remote_manifest.json').read_text())
    if report['local_only']:raise ValueError('Unexpected local files')
    paths=report['remote_only']+report['changed']
    if args.overlay.exists():raise ValueError('Use a new overlay directory')
    command='tar -C /opt/render/project/src/pipeline/data -czf - -- '+' '.join(shlex.quote(p) for p in paths)
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','UpdateHostKeys=no','srv-d6qm9j9aae7s739j7kkg@ssh.oregon.render.com',command],capture_output=True,timeout=180)
    if result.returncode:
        print(result.stderr.decode(errors='replace'));raise SystemExit(result.returncode)
    args.overlay.mkdir(parents=True)
    seen=set()
    with tarfile.open(fileobj=io.BytesIO(result.stdout),mode='r:gz') as archive:
        for member in archive.getmembers():
            if member.name not in paths or not member.isfile():raise ValueError('Unexpected archive member')
            dest=(args.overlay/member.name).resolve()
            if not dest.is_relative_to(args.overlay.resolve()):raise ValueError('Path traversal')
            data=archive.extractfile(member).read()
            if hashlib.sha256(data).hexdigest()!=manifest[member.name]:raise ValueError('Remote changed again; overlay remains incomplete')
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data);os.utime(dest,(member.mtime,member.mtime));seen.add(member.name)
    if seen!=set(paths):raise ValueError('Missing patch file')
    verification={'complete':True,'kind':'verified_virtual_snapshot','source_snapshot':str(args.snapshot.resolve()),'overlay':str(args.overlay.resolve()),'manifest_checked_at':report['checked_at'],'verified_files':report['remote_count'],'overlay_files':paths,'unchanged_verified_files':report['matching']}
    (args.overlay/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8');print(json.dumps(verification,indent=2))
