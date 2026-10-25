"""Collect public, append-only netkeiba supplements for a frozen turf cohort."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import joblib
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.experiment_central_evidence import read_json
from keiba_llm_agent.fetchers.netkeiba_fetcher import DEFAULT_HEADERS


def verify_existing_response(meta: dict, body_path: Path) -> None:
    if meta.get('status') != 200 or meta.get('error'):
        raise RuntimeError(f'Previous collection failure preserved; inspect before resuming: {body_path}')
    if not body_path.exists() or hashlib.sha256(body_path.read_bytes()).hexdigest() != meta.get('stored_sha256'):
        raise RuntimeError(f'Existing response failed integrity verification: {body_path}')


def collect(prepared: Path, output: Path, stage: str, interval: float) -> None:
    output = output.resolve()
    if not output.is_relative_to(ROOT / 'analysis_outputs'):
        raise ValueError('Supplement output must be inside analysis_outputs')
    payload = joblib.load(prepared)
    sources = {s['race_id']: s for s in payload['manifest']['source_files']}
    races = [r for r in payload['rows'] if r['scope'] == 'central_turf']
    horses = set()
    for row in races:
        data = read_json(Path(sources[row['race_id']]['audit']))['inputs']['race_data']
        horses.update(h['horse_id'] for h in data['horses'] if h.get('horse_id') and h.get('recent_runs'))
    tasks = []
    if stage in ('cards', 'all'):
        tasks += [('cards', r['race_id'], f"https://race.netkeiba.com/race/shutuba.html?race_id={r['race_id']}") for r in races]
    if stage in ('horses', 'all'):
        tasks += [('horses', h, f'https://db.netkeiba.com/horse/result/{h}/') for h in sorted(horses)]
    output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)
    previous = 0.0
    consecutive_errors = 0
    print(f'Frozen turf races={len(races)} historical horses={len(horses)} tasks={len(tasks)}', flush=True)
    for index, (kind, identity, url) in enumerate(tasks):
        directory = output / kind
        directory.mkdir(exist_ok=True)
        body_path = directory / f'{identity}.html.gz'
        meta_path = directory / f'{identity}.json'
        if meta_path.exists():
            meta = read_json(meta_path)
            verify_existing_response(meta, body_path)
            continue
        if body_path.exists():
            raise RuntimeError(f'Uncommitted body preserved; inspect before resuming: {body_path}')
        time.sleep(max(0, interval - (time.monotonic() - previous)))
        previous = time.monotonic()
        meta = {'url': url, 'retrieved_at': datetime.now(timezone.utc).isoformat(),
                'kind': kind, 'identity': identity, 'status': None}
        try:
            response = session.get(url, timeout=20)
            meta['status'] = response.status_code
            meta['content_type'] = response.headers.get('Content-Type')
            meta['final_url'] = response.url
            if response.status_code in (401, 403, 429):
                raise RuntimeError(f'Access/rate restriction {response.status_code}; stopping without retry')
            response.raise_for_status()
            response.encoding = 'euc_jp' if kind == 'horses' else 'utf-8'
            text = response.text
            expected = 'db_h_race_results' if kind == 'horses' else 'RaceData01'
            if expected not in text:
                raise RuntimeError('Expected public data table not found; stopping for inspection')
            raw = text.encode('utf-8')
            stored = gzip.compress(raw, mtime=0)
            with body_path.open('xb') as handle:
                handle.write(stored)
            meta.update(content_sha256=hashlib.sha256(raw).hexdigest(),
                        stored_sha256=hashlib.sha256(stored).hexdigest(), bytes=len(raw))
            consecutive_errors = 0
        except requests.RequestException as exc:
            meta['error'] = str(exc)
            consecutive_errors += 1
        except RuntimeError as exc:
            meta['error'] = str(exc)
            with meta_path.open('x', encoding='utf-8') as handle:
                json.dump(meta, handle, ensure_ascii=False, indent=2)
            raise
        with meta_path.open('x', encoding='utf-8') as handle:
            json.dump(meta, handle, ensure_ascii=False, indent=2)
        if consecutive_errors >= 3:
            raise RuntimeError('Three consecutive request failures; stopping without bulk retries')
        if (index + 1) % 50 == 0 or index + 1 == len(tasks):
            print(f'collected {index + 1}/{len(tasks)} last={kind}/{identity}', flush=True)
    session.close()
    print('Collection complete; original snapshots unchanged.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=('cards', 'horses', 'all'), default='all')
    parser.add_argument('--interval', type=float, default=0.5)
    args = parser.parse_args()
    if args.interval < 0.5:
        parser.error('Minimum request interval is 0.5 seconds')
    collect(args.prepared, args.output, args.stage, args.interval)
