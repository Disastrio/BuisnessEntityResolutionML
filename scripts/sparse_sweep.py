"""
sparse_sweep.py — Candidate recall of sparse-dot top-N retrieval.

Usage:
    python scripts/sparse_sweep.py [n_s1] [cap1 cap2 ...]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.io import load_aligned_sample
from src.normalize import normalize_all_sources
from src.features import restrict_to_core_columns
from src.sparse_retrieval import SparseIndex

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
CAPS = [int(x) for x in sys.argv[2:]] or [100, 300, 500]

s1, s2, s3, gt = load_aligned_sample(n_s1=N)
s1n, s2n, s3n = normalize_all_sources(s1, s2, s3)
del s1, s2, s3
s1n = restrict_to_core_columns(s1n)
s2n = restrict_to_core_columns(s2n)
s3n = restrict_to_core_columns(s3n)

targets = pd.concat([s2n, s3n], ignore_index=True)
t0 = time.time()
index = SparseIndex.build(targets["name_clean"].tolist(),
                          targets["entity_id"].tolist())
print(f"sparse index build {time.time()-t0:.0f}s  docs={index.X.shape[0]} "
      f"feats={index.X.shape[1]} nnz={index.X.nnz}", flush=True)

s1_name = dict(zip(s1n["entity_id"], s1n["name_clean"]))

for cap in CAPS:
    t = time.time()
    found = total = 0
    for sid, true in gt.items():
        if not true:
            continue
        total += len(true)
        ids, _ = index.query_topn(s1_name.get(sid, ""), cap)
        found += len(true & set(ids))
    rec = found / total if total else 0.0
    print(f"SPARSE cap={cap} recall={rec:.4f} found={found}/{total} "
          f"secs={time.time()-t:.0f}", flush=True)
print("SPARSE_SWEEP_DONE", flush=True)
