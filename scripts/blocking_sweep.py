"""
blocking_sweep.py — Measure candidate recall at several candidate caps.

Loads an aligned sample once, builds the blocking indices once, then evaluates
candidate recall for each cap in CAPS.

Usage:
    python scripts/blocking_sweep.py [n_s1] [cap1 cap2 ...]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.io import load_aligned_sample
from src.normalize import normalize_all_sources
from src.features import restrict_to_core_columns
from src.blocking import (
    build_all_indices, generate_candidates_from_bundle, evaluate_blocking,
)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
CAPS = [int(x) for x in sys.argv[2:]] or [100, 300, 500]

t0 = time.time()
s1, s2, s3, gt = load_aligned_sample(n_s1=N)
s1n, s2n, s3n = normalize_all_sources(s1, s2, s3)
del s1, s2, s3
s1n = restrict_to_core_columns(s1n)
s2n = restrict_to_core_columns(s2n)
s3n = restrict_to_core_columns(s3n)
print(f"loaded in {time.time()-t0:.0f}s: S1={len(s1n)} S2={len(s2n)} S3={len(s3n)}", flush=True)

t = time.time()
bundle = build_all_indices(s2n, s3n)
print(f"index build {time.time()-t:.0f}s", flush=True)

for cap in CAPS:
    t = time.time()
    cands = generate_candidates_from_bundle(
        s1n, bundle, max_candidates=cap, show_progress=False)
    m = evaluate_blocking(cands, gt)
    print(
        f"RESULT cap={cap} recall={m['candidate_recall']} avg={m['avg_candidates_per_s1']} "
        f"max={m['max_candidates_per_s1']} found={m['found_true_matches']}/{m['total_true_matches']} "
        f"secs={time.time()-t:.0f}",
        flush=True,
    )
print("SWEEP_DONE", flush=True)
