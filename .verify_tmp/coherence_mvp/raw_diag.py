#!/usr/bin/env python3
import json
import numpy as np

man = json.load(open('corpus_v3/manifest.json'))
PAIRS = {'splice': 'orb_inliers', 'room_swap': 'orb_inliers',
         'reverse_segment': 'flow_flip', 'loop': 'recur_frac',
         'timewarp': 'motion_step'}
OFF = {'orb_inliers': 1, 'recur_frac': 0, 'flow_flip': 30, 'motion_step': 30}

for sig in ('orb_inliers', 'flow_flip', 'motion_step', 'recur_frac'):
    print(f"\n=== {sig} (raw) ===")
    ops = [o for o, s in PAIRS.items() if s == sig]
    cleans, corrupts = [], []
    for c in man['clips']:
        x = np.load(f"results/signals_v3/{c['id']}.npz")[sig]
        if c['operator'] == 'none':
            cleans.append((c['id'], float(x.min()), float(x.max())))
        elif c['operator'] in ops:
            zb = []
            for b in c['breaks_s']:
                i = max(0, min(len(x) - 1, int(b * 30) - OFF[sig]))
                zb.append(round(float(x[i]), 3))
            corrupts.append((c['id'], float(x.min()), float(x.max()), zb))
    print("clean mins :", [round(v[1], 2) for v in cleans])
    print("clean maxes:", [round(v[2], 2) for v in cleans])
    for cid, mn, mx, zb in corrupts:
        print(f"  {cid[:36]:36s} min={mn:8.3f} max={mx:8.3f} raw@brk={zb}")
