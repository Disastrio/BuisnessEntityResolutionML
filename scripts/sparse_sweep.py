"""
sparse_sweep.py — Candidate recall of sparse top-N retrievers.

Compares three high-recall name retrievers on an aligned sample:
  * SPARSE   — TF-IDF character-n-gram cosine top-N (SparseIndex)
  * BM25     — Okapi BM25 word-n-gram top-N (BM25Index)
  * UNION    — union of both candidate sets (what blocking would use)

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
from src.sparse_retrieval import SparseIndex, BM25Index

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
CAPS = [int(x) for x in sys.argv[2:]] or [100, 300, 500]

s1, s2, s3, gt = load_aligned_sample(n_s1=N)
s1n, s2n, s3n = normalize_all_sources(s1, s2, s3)
del s1, s2, s3
s1n = restrict_to_core_columns(s1n)
s2n = restrict_to_core_columns(s2n)
s3n = restrict_to_core_columns(s3n)

targets = pd.concat([s2n, s3n], ignore_index=True)
names = targets["name_clean"].tolist()
tids = targets["entity_id"].tolist()

t0 = time.time()
sparse = SparseIndex.build(names, tids)
print(f"sparse index build {time.time()-t0:.0f}s  docs={sparse.X.shape[0]} "
      f"feats={sparse.X.shape[1]} nnz={sparse.X.nnz}", flush=True)

t0 = time.time()
bm25 = BM25Index.build(names, tids)
print(f"bm25   index build {time.time()-t0:.0f}s  docs={bm25.X.shape[0]} "
      f"feats={bm25.X.shape[1]} nnz={bm25.X.nnz}", flush=True)

s1_name = dict(zip(s1n["entity_id"], s1n["name_clean"]))
queries = [(sid, true) for sid, true in gt.items() if true]


def sweep(label, retriever):
    for cap in CAPS:
        t = time.time()
        found = total = 0
        for sid, true in queries:
            total += len(true)
            ids, _ = retriever.query_topn(s1_name.get(sid, ""), cap)
            found += len(true & set(ids))
        rec = found / total if total else 0.0
        print(f"{label} cap={cap} recall={rec:.4f} found={found}/{total} "
              f"secs={time.time()-t:.0f}", flush=True)


def sweep_union():
    for cap in CAPS:
        t = time.time()
        found = total = 0
        for sid, true in queries:
            total += len(true)
            sp_ids, _ = sparse.query_topn(s1_name.get(sid, ""), cap)
            bm_ids, _ = bm25.query_topn(s1_name.get(sid, ""), cap)
            found += len(true & (set(sp_ids) | set(bm_ids)))
        rec = found / total if total else 0.0
        print(f"UNION cap={cap} recall={rec:.4f} found={found}/{total} "
              f"secs={time.time()-t:.0f}", flush=True)


sweep("SPARSE", sparse)
sweep("BM25", bm25)
sweep_union()
print("SPARSE_SWEEP_DONE", flush=True)
