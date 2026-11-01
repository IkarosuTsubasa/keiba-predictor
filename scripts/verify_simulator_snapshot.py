"""Read-only remote/local SHA-256 reconciliation after a moving-source download."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

REMOTE = r'''
import pathlib,hashlib,json
root=pathlib.Path('/opt/render/project/src/pipeline/data')
out={}
for name in ('agent_predictions','agent_results','agent_backtest_audit'):
 for p in sorted((root/name).rglob('*')):
  if p.is_file(): out[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
print(json.dumps(out))
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("snapshot", type=Path); args = parser.parse_args()
    completed = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "UpdateHostKeys=no", "srv-d6qm9j9aae7s739j7kkg@ssh.oregon.render.com", "python -c " + shlex.quote(REMOTE)], capture_output=True, check=True, timeout=240)
    remote = json.loads(completed.stdout)
    local = {str(p.relative_to(args.snapshot)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest() for d in ('agent_predictions', 'agent_results', 'agent_backtest_audit') for p in (args.snapshot/d).rglob('*') if p.is_file()}
    changed = [k for k in local.keys() & remote.keys() if local[k] != remote[k]]
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "remote_count": len(remote), "local_count": len(local), "local_only": sorted(local.keys()-remote.keys()), "remote_only": sorted(remote.keys()-local.keys()), "changed": sorted(changed), "matching": len(local.keys() & remote.keys())-len(changed), "complete": local == remote}
    (args.snapshot / "simulator_remote_manifest.json").write_text(json.dumps(remote), encoding="utf-8")
    (args.snapshot / "simulator_integrity_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2)); raise SystemExit(0 if report['complete'] else 1)
