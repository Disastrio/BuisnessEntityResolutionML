"""
combined_sweep.py — Candidate recall for blocking vs sparse vs their union.

Usage:
    python scripts/combined_sweep.py [n_s1] [block_cap] [sparse_cap1 sparse_cap2 ...]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.io import load_aligned_sample
from src.normalize import normalize_all_sources
from src.features import restrict_to_core_columns
from src.blocking import build_all_indices, generate_candidates_from_bundle
from src.sparse_retrieval import SparseIndex

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
BCAP = int(sys.argv[2]) if len(sys.argv) > 2 else 100
SCAPS = [int(x) for x in sys.argv[3:]] or [100, 300]

s1, s2, s3, gt = load_aligned_sample(n_s1=N)
s1n, s2n, s3n = normalize_all_sources(s1, s2, s3)
del s1, s2, s3
s1n = restrict_to_core_columns(s1n)
s2n = restrict_to_core_columns(s2n)
s3n = restrict_to_core_columns(s3n)

bundle = build_all_indices(s2n, s3n)
block = generate_candidates_from_bundle(
    s1n, bundle, max_candidates=BCAP, show_progress=False)

targets = pd.concat([s2n, s3n], ignore_index=True)
t = time.time()
index = SparseIndex.build(targets["name_clean"].tolist(),
                          targets["entity_id"].tolist())
print(f"sparse index build {time.time()-t:.0f}s", flush=True)
s1_name = dict(zip(s1n["entity_id"], s1n["name_clean"]))


def recall(sets_fn):
    found = total = 0
    for sid, true in gt.items():
        if not true:
            continue
        total += len(true)
        found += len(true & sets_fn(sid))
    return found, total


bf, bt = recall(lambda sid: block.get(sid, set()))
print(f"BLOCK cap={BCAP} recall={bf/bt:.4f} found={bf}/{bt}", flush=True)

for scap in SCAPS:
    sf, st = recall(lambda sid: set(index.query_topn(s1_name.get(sid, ''), scap)[0]))
    print(f"SPARSE cap={scap} recall={sf/st:.4f} found={sf}/{st}", flush=True)
    uf, ut = recall(lambda sid: block.get(sid, set()) |
                    set(index.query_topn(s1_name.get(sid, ''), scap)[0]))
    print(f"UNION  cap={scap} recall={uf/ut:.4f} found={uf}/{ut}", flush=True)
print("COMBINED_DONE", flush=True)
