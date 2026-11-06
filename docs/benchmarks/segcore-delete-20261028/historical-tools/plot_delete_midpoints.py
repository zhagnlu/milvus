import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

root = Path('/tmp/delete-size-midpoints-20261027')
original = list(csv.DictReader((root / 'matrix.csv').open()))
def key(r):
    return r['type'], r['rows'], r['mode'], r['batch']
retests = {}
for path in root.glob('retest-*.csv'):
    for row in csv.DictReader(path.open()):
        assert key(row) not in retests
        retests[key(row)] = row
plan = json.loads((root / 'retest-plan.json').read_text())
expected_retests = {(g['type'], g['rows'], g['mode'], b)
                    for g in plan['groups'] for b in g['batches']}
assert set(retests) == expected_retests
row_counts = [100000, 500000, 1000000, 2000000, 8000000, 44000000]
batches = [1024, 2048, 4096, 8192, 16384, 32768, 49152, 65536]
expected = {(t, str(n), m, str(b)) for t in ['int64', 'varchar16']
            for n in row_counts for m in ['stream_same', 'load'] for b in batches}
assert {key(r) for r in original} == expected and len(original) == 192
records = []
for row in original:
    r = dict(row)
    retested = key(row) in retests
    if retested:
        r.update(retests[key(row)])
    r.update(original_speedup=row['speedup'], retested=int(retested),
             rounds=15 if retested else 5)
    records.append(r)
with (root / 'results.csv').open('w') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
versions = {}
for path in root.glob('*.metadata.txt'):
    for line in path.read_text().splitlines():
        k, sep, value = line.partition('=')
        if sep and k in ('source_sha256', 'kernel_sha256', 'benchmark_sha256'):
            versions.setdefault(k, set()).add(value)
assert all(len(v) == 1 for v in versions.values())
(root / 'validation.json').write_text(json.dumps({
    'matrix_cases': len(records), 'retested_cases': len(retests),
    'hashes': {k: list(v)[0] for k,v in versions.items()},
    'retest_rule': plan['criterion']}, indent=2))

lookup = {key(r): r for r in records}
logs = [np.log2(float(r['speedup'])) for r in records]
norm = colors.DivergingNorm(vmin=min(-1., min(logs)), vcenter=0., vmax=max(logs))
fig, axes = plt.subplots(2, 2, figsize=(18, 10))
for i, mode in enumerate(['stream_same', 'load']):
    for j, kind in enumerate(['int64', 'varchar16']):
        ax = axes[i, j]
        cells = [[lookup[(kind, str(n), mode, str(b))] for b in batches] for n in row_counts]
        values = np.array([[np.log2(float(r['speedup'])) for r in line] for line in cells])
        im = ax.imshow(values, cmap='RdBu', norm=norm, aspect='auto')
        ax.set_xticks(range(8))
        ax.set_xticklabels(['1,024','2,048','4,096','8,192','16,384','32,768','49,152','65,536'], fontsize=9)
        ax.set_yticks(range(6))
        ax.set_yticklabels(['{} / C={}'.format(label, line[0]['chunks'])
                            for label,line in zip(['100k','500k','1M','2M','8M','44M'],cells)])
        ax.set_xlabel('Delete pairs (unique PKs)')
        ax.set_ylabel('Segment rows / modeled chunk count')
        ax.set_title('{} | {}'.format('INT64' if kind == 'int64' else 'VARCHAR(16)',
                                     'stream, one timestamp' if mode == 'stream_same' else 'load replay'))
        for y,line in enumerate(cells):
            for x,r in enumerate(line):
                label = '{:.2f}x{}'.format(float(r['speedup']), '*' if r['retested'] else '')
                shade = norm(values[y,x])
                ax.text(x,y,label,ha='center',va='center',fontsize=10,
                        color='white' if shade < .15 or shade > .85 else '#111111')
fig.suptitle('Delete batch midpoints: old/new preparation + lookup runtime', fontsize=18, y=.98)
fig.text(.5,.935,'4 MiB modeled system-column groups | random unique PKs | i7-8700 | forced paths | 2026-10-27',ha='center',fontsize=11)
fig.subplots_adjust(left=.08,right=.9,top=.88,bottom=.12,hspace=.32,wspace=.3)
cax=fig.add_axes([.93,.17,.015,.63])
bar=fig.colorbar(im,cax=cax)
ticks=[.5,1,2,4,8,16,32]
bar.set_ticks(np.log2(ticks)); bar.set_ticklabels(['{}x'.format(t) for t in ticks])
bar.set_label('>1 new faster; <1 new slower')
fig.text(.5,.035,'Median paired ratios, 5 rounds. * = 15-round retest of a noisy near-parity case; original results retained in CSV.\n'
         'Excludes cgo, storage pins, skiplist, snapshots and tSafe. This is not a full apply benchmark.',ha='center',fontsize=10)
fig.savefig(str(root/'heatmap.png'),dpi=160)
fig.savefig(str(root/'heatmap.svg'))
plt.close(fig)
print('validated primary cases:',len(records),'retests:',len(retests))
for mode in ['stream_same','load']:
    print('int64',mode)
    for n in row_counts:
        print(n,' '.join('{:.2f}{}'.format(float(lookup[('int64',str(n),mode,str(b))]['speedup']),
                                         '*' if lookup[('int64',str(n),mode,str(b))]['retested'] else '') for b in batches))
