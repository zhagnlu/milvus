import csv
import json
from pathlib import Path

base = Path('/tmp')
records = list(csv.DictReader((base / 'delete-size-results.csv').open()))
roots = [base / name for name in (
    'delete-size-grid-4MiB', 'delete-size-grid-1MiB', 'delete-size-grid-bulk')]
versions = {'source_sha256': set(), 'kernel_sha256': set(), 'benchmark_sha256': set()}
for root in roots:
    manifest = json.loads((root / 'manifest.json').read_text())
    expected = {(kind, str(rows), str(mib), str(batch), mode)
                for kind in manifest['types'].split(',')
                for rows in manifest['rows']
                for mib in manifest['cell_mib']
                for batch in manifest['batches'] if batch <= rows
                for mode in manifest['modes'].split(',')}
    actual = {(r['type'], r['rows'], r['cell_mib'], r['batch'], r['mode'])
              for r in records if r['run'] == root.name}
    assert actual == expected, (root.name, expected - actual, actual - expected)
    for path in root.glob('*.metadata.txt'):
        for line in path.read_text().splitlines():
            key, sep, value = line.partition('=')
            if sep and key in versions:
                versions[key].add(value)
for key, values in versions.items():
    assert len(values) == 1, (key, values)
for r in records:
    assert int(r['batch']) <= int(r['rows'])
    assert r['shape'] == 'uniform_unique'
    assert float(r['old_total_us']) > 0 and float(r['new_total_us']) > 0
controls = [r for r in records if r['run'] == 'gate-control']
assert {(r['chunks'], r['batch'], r['mode']) for r in controls} == {
    (str(c), str(b), mode) for c in (16, 32)
    for b in (262144, 440000, 1500000) for mode in ('load', 'stream_same')}

def proposed_gate(r):
    c, b = int(r['chunks']), int(r['batch'])
    return (c >= 32 and b >= 128) or (16 <= c < 32 and b >= 1024)

selected = [r for r in records if proposed_gate(r)]
worst = min(selected, key=lambda r: float(r['speedup']))
regressions = [r for r in selected if float(r['speedup']) < 1]
report = {'cases': len(records), 'versions': {k: list(v)[0] for k, v in versions.items()},
          'candidate_gate_cases': len(selected), 'candidate_gate_worst': worst,
          'candidate_gate_regressions': regressions}
(base / 'delete-size-validation.json').write_text(json.dumps(report, indent=2))
print('cases', len(records), 'candidate gate selected', len(selected), 'regressions', len(regressions))
print('candidate worst', json.dumps(worst))
for r in records:
    if r['run'] == 'gate-control':
        print('gate control', r['chunks'], r['batch'], r['mode'], r['old_total_us'],
              r['new_total_us'], r['speedup'], r['round_min_speedup'])
