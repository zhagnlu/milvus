import csv
import glob
import json
from pathlib import Path

rows = []
for path in sorted(glob.glob('/tmp/delete-real-layout-*.csv')):
    for row in csv.DictReader(open(path)):
        row['run'] = Path(path).stem.removeprefix('delete-real-layout-')
        rows.append(row)

def brief(row):
    return {key: row[key] for key in (
        'run', 'type', 'rows', 'chunks', 'shape', 'mode', 'batch',
        'old_total_us', 'new_total_us', 'speedup', 'round_min_speedup')}

def stats(selected):
    if not selected:
        return {}
    return {
        'cases': len(selected),
        'slower_cases': sum(float(r['speedup']) < 1 for r in selected),
        'over_10_percent_slower_cases': sum(float(r['speedup']) < 1/1.1 for r in selected),
        'worst': brief(min(selected, key=lambda r: float(r['speedup']))),
    }

report = {'cases': len(rows), 'gates': {}, 'layouts': {}}
for batch in (128, 256, 512, 1024):
    for chunks in (1, 8, 12, 16, 32):
        selected = [r for r in rows if int(r['batch']) >= batch and int(r['chunks']) >= chunks]
        report['gates'][f'batch>={batch},chunks>={chunks}'] = stats(selected)
for key in sorted({(r['type'], int(r['rows']), int(r['chunks'])) for r in rows}):
    selected = [r for r in rows if (r['type'], int(r['rows']), int(r['chunks'])) == key]
    report['layouts'][str(key)] = {
        str(batch): stats([r for r in selected if int(r['batch']) == batch])
        for batch in sorted({int(r['batch']) for r in selected})
    }
Path('/tmp/delete-layout-analysis.json').write_text(json.dumps(report, indent=2))
with open('/tmp/delete-layout-results.csv', 'w') as output:
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print('cases', len(rows))
for gate, result in report['gates'].items():
    print(gate, json.dumps(result))
