"""
tune_threshold.py — Re-tune the decision threshold / singleton barrier for a
given candidate cap, reusing the already-trained model.

Usage:
    python scripts/tune_threshold.py <n_s1> <max_candidates> [n_workers]

Writes models/tuning_<cap>.json with the best threshold/barrier.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MODELS_DIR, FEATURE_WORKERS
from src.io import load_aligned_sample
from src.normalize import normalize_all_sources
from src.features import (
    restrict_to_core_columns, build_feature_matrix, assign_labels,
)
from src.blocking import (
    build_all_indices, generate_candidates_from_bundle, evaluate_blocking,
)
from src.train import load_model
from src.predict import predict_probabilities, conservative_reject_mask
from src.evaluate import (
    threshold_sweep, sweep_singleton_barrier, detailed_evaluation, apply_threshold,
)


def main() -> None:
    n_s1 = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    n_workers = int(sys.argv[3]) if len(sys.argv) > 3 else FEATURE_WORKERS

    t0 = time.time()
    s1, s2, s3, gt = load_aligned_sample(n_s1=n_s1)
    s1n, s2n, s3n = normalize_all_sources(s1, s2, s3, n_workers=n_workers)
    del s1, s2, s3
    s1n = restrict_to_core_columns(s1n)
    s2n = restrict_to_core_columns(s2n)
    s3n = restrict_to_core_columns(s3n)

    bundle = build_all_indices(s2n, s3n)
    cands = generate_candidates_from_bundle(
        s1n, bundle, max_candidates=cap, show_progress=True, n_workers=n_workers)

    bm = evaluate_blocking(cands, gt)
    print(f"BLOCKING cap={cap} candidate_recall={bm['candidate_recall']} "
          f"avg={bm['avg_candidates_per_s1']} max={bm['max_candidates_per_s1']} "
          f"found={bm['found_true_matches']}/{bm['total_true_matches']}", flush=True)

    import pandas as pd
    target = pd.concat([s2n, s3n], ignore_index=True)
    feats, s1_ids, t_ids = build_feature_matrix(
        s1n, target, cands, show_progress=True, n_workers=n_workers)
    labels = assign_labels(s1_ids, t_ids, gt)
    print(f"pairs={len(feats):,} positives={int(labels.sum()):,}", flush=True)

    model, meta = load_model()
    probs = predict_probabilities(model, feats)
    reject = conservative_reject_mask(feats)

    best_t, score_t, _ = threshold_sweep(
        s1_ids, t_ids, probs, gt, n_workers=n_workers)
    best_b, score_b, _ = sweep_singleton_barrier(
        s1_ids, t_ids, probs, gt, threshold=best_t, n_workers=n_workers)

    preds = apply_threshold(
        s1_ids, t_ids, probs, best_t,
        all_s1_ids=set(gt.keys()), barrier=best_b, reject_mask=reject)
    details = detailed_evaluation(gt, preds)

    out = {
        "cap": cap, "n_s1": n_s1,
        "best_threshold": float(best_t), "threshold_macro_f05": float(score_t),
        "best_barrier": float(best_b), "barrier_macro_f05": float(score_b),
        "details": details,
    }
    (MODELS_DIR / f"tuning_{cap}.json").write_text(json.dumps(out, indent=2))
    print("TUNING_RESULT", json.dumps(out), flush=True)
    print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
