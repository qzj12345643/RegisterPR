"""Print cache progress without importing Torch or touching every cached file."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
states = [json.loads(path.read_text()) for path in sorted((root / 'exp/register_feature_caching').glob('rank[0-9]*.json'))]
if not states:
    log = root / 'exp/register_feature_caching/run.log'
    tail = ''
    if log.is_file():
        with log.open('rb') as stream:
            stream.seek(max(log.stat().st_size - 4096, 0))
            tail = stream.read().decode(errors='replace')
    waiting = [line for line in tail.splitlines() if 'waiting for GPU memory' in line]
    print('\n'.join(waiting[-4:]) if waiting else 'Workers are initializing; see exp/register_feature_caching/run.log')
else:
    done = sum(s['generated'] + s['skipped'] for s in states)
    expected = json.loads((root / 'outputs/register_readiness/data_audit.json').read_text())['requested_tokens']
    print(json.dumps(dict(
        cached_seen_by_workers=done, expected=expected, percent=round(100*done/expected, 2),
        failed=sum(s['failed'] for s in states), workers_reporting=len(states),
        all_workers_finished=len(states) == states[0]['world_size'] and all(s['complete'] for s in states),
    ), indent=2))
