"""Build separate retrospective inputs; never add histories or rewrite frozen facts."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date
import gzip
import hashlib
import json
from pathlib import Path
import sys

import joblib
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.experiment_central_evidence import read_json, digest, save_json
from keiba_llm_agent.parsers.netkeiba_horse_parser import parse_horse_recent_runs
from keiba_llm_agent.parsers.netkeiba_race_parser import parse_netkeiba_shutuba_html, extract_race_id_from_html
from keiba_llm_agent.schemas.race_data import RaceData

DETAIL_FIELDS = ('time_seconds', 'carried_weight', 'body_weight', 'body_weight_change', 'pace_first_3f', 'pace_last_3f')
IDENTITY_FIELDS = ('race_id', 'date', 'course', 'surface', 'distance', 'finish')


def parser_identity():
    paths = ['schemas/race_data.py', 'parsers/netkeiba_horse_parser.py',
             'parsers/netkeiba_race_parser.py', 'parsers/netkeiba_performance_fields.py']
    return hashlib.sha256(''.join(digest(ROOT/'keiba_llm_agent'/p) for p in paths).encode()).hexdigest()


def write_new_gzip(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8')
    with path.open('xb') as handle:
        handle.write(gzip.compress(encoded, mtime=0))


def cache_public_pages(raw, output, kind='all'):
    identity = parser_identity()
    cache = output/'parsed'/identity[:16]
    kinds = ('cards', 'horses') if kind == 'all' else (kind,)
    for category in kinds:
        files = sorted((raw/category).glob('*.json'))
        processed = 0
        for meta_path in files:
            meta = read_json(meta_path)
            body = meta_path.with_suffix('.html.gz')
            if meta.get('status') != 200 or meta.get('error') or not body.exists():
                continue
            target = cache/category/(meta_path.stem+'.json.gz')
            if target.exists():
                continue
            assert digest(body) == meta['stored_sha256']
            html = gzip.decompress(body.read_bytes()).decode('utf-8')
            if category == 'cards':
                detected = extract_race_id_from_html(BeautifulSoup(html, 'html.parser'))
                if detected != meta_path.stem:
                    raise ValueError(f'Card identity mismatch: {meta_path.stem} != {detected}')
                data = parse_netkeiba_shutuba_html(html, meta_path.stem).model_dump()
                for horse in data['horses']:
                    horse['odds'] = horse['popularity'] = None
            else:
                data = [{k: getattr(run, k) for k in IDENTITY_FIELDS+DETAIL_FIELDS}
                        for run in parse_horse_recent_runs(html, include_performance_details=True)]
            write_new_gzip(target, {'data': data, 'source': str(body), 'source_sha256': meta['stored_sha256'],
                                   'retrieved_at': meta['retrieved_at'], 'parser_sha256': identity})
            processed += 1
            if processed % 50 == 0:
                print(f'parsed {category} new={processed}', flush=True)
        print(f'parsed {category} new={processed} available_metadata={len(files)}', flush=True)
    return cache


def compatible_history(original, candidate):
    return all(getattr(original, k) is None or candidate.get(k) == getattr(original, k)
               for k in ('date', 'course', 'surface', 'distance', 'finish'))


def enrich_frozen_race(race, card, histories):
    enriched = race.model_copy(deep=True)
    info = race.race_info
    counts = Counter()
    card_ok = card is not None and all(card['race_info'].get(k) == getattr(info, k)
                                      for k in ('race_id', 'race_date', 'course', 'surface', 'distance'))
    card_horses = {h['horse_no']: h for h in card['horses']} if card_ok else {}
    if card_ok:
        incoming_class = card['race_info'].get('race_class')
        if enriched.race_info.race_class is None:
            enriched.race_info.race_class = incoming_class
            enriched.race_info.race_class_source = card['race_info'].get('race_class_source')
        elif incoming_class is not None and enriched.race_info.race_class != incoming_class:
            counts['existing_class_conflict_preserved'] += 1
        counts['race_class_available'] = int(enriched.race_info.race_class is not None)
    else:
        counts['missing_or_mismatched_card'] += 1
    target = date.fromisoformat(info.race_date)
    for horse in enriched.horses:
        counts['horse_observations'] += 1
        current = card_horses.get(horse.horse_no)
        if current and horse.horse_id and current.get('horse_id') == horse.horse_id:
            for field in ('age', 'sex'):
                if getattr(horse, field) is None:
                    setattr(horse, field, current.get(field))
                elif current.get(field) is not None and getattr(horse, field) != current[field]:
                    counts['existing_'+field+'_conflict_preserved'] += 1
            counts['age_available'] += horse.age is not None
            counts['sex_available'] += horse.sex is not None
        by_id = defaultdict(list); by_context = defaultdict(list)
        for row in histories.get(horse.horse_id, []):
            by_id[row.get('race_id')].append(row)
            by_context[(row.get('date'), row.get('course'), row.get('distance'))].append(row)
        for run in horse.recent_runs:
            if not run.date or run.race_id == info.race_id:
                continue
            try:
                if date.fromisoformat(run.date) >= target:
                    continue
            except ValueError:
                continue
            counts['prior_history_rows'] += 1
            candidates = by_id.get(run.race_id, []) if run.race_id else by_context.get((run.date, run.course, run.distance), [])
            matches = [r for r in candidates if compatible_history(run, r)]
            if len(matches) != 1:
                counts['history_not_uniquely_matched'] += 1
                continue
            counts['history_matched'] += 1
            for field in DETAIL_FIELDS:
                if getattr(run, field) is None and matches[0].get(field) is not None:
                    setattr(run, field, matches[0][field])
                    counts['filled_'+field] += 1
        # Do not infer that absent public history is a new horse or copy late target body weights.
    return enriched, counts


def assemble(prepared, raw, output, cache):
    destination = output/'enriched_races'
    if destination.exists():
        raise ValueError('Existing enriched inputs are preserved; use a fresh output directory')
    payload = joblib.load(prepared)
    sources = {s['race_id']: s for s in payload['manifest']['source_files']}
    histories = {}; lineage = []
    for path in (cache/'horses').glob('*.json.gz'):
        value = read_json(path)
        histories[path.name.split('.')[0]] = value['data']
        lineage.append({k: value[k] for k in ('source', 'source_sha256', 'retrieved_at', 'parser_sha256')})
    counts = Counter(); races = []
    for row in payload['rows']:
        if row['scope'] != 'central_turf':
            continue
        source = sources[row['race_id']]
        assert digest(Path(source['audit'])) == source['audit_sha256']
        race = RaceData.model_validate(read_json(Path(source['audit']))['inputs']['race_data'])
        card_path = cache/'cards'/f"{row['race_id']}.json.gz"
        card = read_json(card_path) if card_path.exists() else None
        if card:
            lineage.append({k: card[k] for k in ('source', 'source_sha256', 'retrieved_at', 'parser_sha256')})
        enriched, coverage = enrich_frozen_race(race, card['data'] if card else None, histories)
        counts.update(coverage)
        target = destination/f"{row['race_id']}.json.gz"
        write_new_gzip(target, enriched.model_dump())
        races.append({'race_id': row['race_id'], 'original_audit': source['audit'],
                      'original_audit_sha256': source['audit_sha256'], 'enriched': str(target),
                      'enriched_sha256': digest(target), 'coverage': dict(coverage)})
    for source in lineage:
        assert digest(Path(source['source'])) == source['source_sha256']
    manifest = {'source_prepared': str(prepared.resolve()), 'source_prepared_sha256': digest(prepared),
                'raw_directory': str(raw), 'parser_sha256': parser_identity(), 'counts': dict(counts),
                'races': races, 'lineage': lineage, 'historical_horses_parsed': len(histories),
                'timing_contract': 'Retrospective stable card age/sex/class and matched prior-run facts; no target outcome, no new histories, no target body weight, no odds/popularity.'}
    save_json(output/'supplement_manifest.json', manifest)
    print(json.dumps({'races': len(races), 'coverage': counts, 'historical_horses': len(histories)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-only', action='store_true')
    parser.add_argument('--kind', choices=('all', 'cards', 'horses'), default='all')
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(ROOT/'analysis_outputs'):
        parser.error('Output must be inside analysis_outputs')
    out.mkdir(parents=True, exist_ok=True)
    cache = cache_public_pages(args.raw.resolve(), out, args.kind)
    if not args.cache_only:
        assemble(args.prepared, args.raw.resolve(), out, cache)
