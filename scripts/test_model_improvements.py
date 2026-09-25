"""
test_model_improvements.py — Self-contained checks for the E4-E8 model upgrades.

Runs entirely on tiny synthetic data (no challenge dataset needed), so it is
safe to run under tight memory. Exits non-zero if any check fails.

Usage:
    python scripts/test_model_improvements.py
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.normalize import normalize_dataframe
from src.blocking import (
    soundex, build_candidate_indices, _ranked_candidates_for_s1,
    generate_all_candidates,
)
from src.features import FEATURE_NAMES
from src.train import (
    identify_hard_negatives, compute_sample_weights,
    split_by_s1_entity, train_dual_models, predict_dual_probabilities,
    save_dual_models, load_dual_models,
)
from src.evaluate import (
    infer_source, assemble_predictions, threshold_sweep,
    threshold_sweep_by_source, sweep_singleton_barrier, macro_fbeta,
)
from src.predict import (
    conservative_reject_mask, assemble_matches,
    save_matching_results, save_candidate_pairs, validate_output,
)

_PASS = 0
_FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global _PASS, _FAIL
    if condition:
        _PASS += 1
        print(f"  [PASS] {name}")
    else:
        _FAIL += 1
        print(f"  [FAIL] {name} {detail}")


def make_features(rows):
    """Build a 28-column feature frame from partial dicts."""
    frame = pd.DataFrame(rows)
    for col in FEATURE_NAMES:
        if col not in frame.columns:
            frame[col] = 0.0
    return frame[FEATURE_NAMES]


def raw(name, addr, country):
    return {"business_name": name, "business_address": addr, "country": country}


# ═════════════════════════════════════════════════════════════════════════════
# Index 6 — Soundex
# ═════════════════════════════════════════════════════════════════════════════
def test_soundex():
    print("\n[1] Phonetic (Soundex) index")
    check("smith == smyth", soundex("smith") == soundex("smyth"))
    check("muhammed == mohammed", soundex("muhammed") == soundex("mohammed"))
    check("code length is 4", len(soundex("kalyan")) == 4)
    check("empty token -> ''", soundex("") == "")

    s2 = normalize_dataframe(pd.DataFrame([
        raw("smyth enterprises", "1 alpha road xyz 11111", "US"),
    ]).assign(entity_id=["S2-1"]), "S2")
    idx = build_candidate_indices(s2)
    check("build_candidate_indices exposes 'soundex'", "soundex" in idx)

    s1n = normalize_dataframe(pd.DataFrame([
        raw("smith enterprises", "9 beta avenue abc 99999", "US"),
    ]).assign(entity_id=["S1-1"]), "S1")
    # Isolated: only the phonetic index is available.
    only_soundex = {"soundex": idx["soundex"]}
    ranked = _ranked_candidates_for_s1(s1n.iloc[0], only_soundex, rare_tokens=set())
    check("phonetic index retrieves transliteration match",
          "S2-1" in ranked, f"got {ranked}")

    # Deterministic cap.
    big = {"soundex": {"us|sndx|S530": [f"S2-{i:05d}" for i in range(50)]}}
    r1 = _ranked_candidates_for_s1(s1n.iloc[0], big, rare_tokens=set(), max_candidates=5)
    r2 = _ranked_candidates_for_s1(s1n.iloc[0], big, rare_tokens=set(), max_candidates=5)
    check("candidate cap is deterministic", r1 == r2 and len(r1) == 5, f"{r1}")


# ═════════════════════════════════════════════════════════════════════════════
# E4 — Hard negatives & sample weights
# ═════════════════════════════════════════════════════════════════════════════
def test_hard_negatives():
    print("\n[2] E4 - Hard-negative mining")
    feats = make_features([
        {"name_levenshtein": 0.99},                              # positive
        {"name_levenshtein": 0.95},                              # hard (name)
        {"addr_token_jaccard": 0.80},                            # hard (addr)
        {"name_levenshtein": 0.20, "addr_token_jaccard": 0.10},  # easy
    ])
    labels = np.array([1.0, 0.0, 0.0, 0.0])
    hard = identify_hard_negatives(feats, labels)
    check("hard-negative mask", list(hard) == [False, True, True, False], f"{list(hard)}")

    weights = compute_sample_weights(labels, feats, hard_negative_weight=3.0)
    check("hard negatives upweighted",
          list(weights) == [1.0, 3.0, 3.0, 1.0], f"{list(weights)}")

    off = compute_sample_weights(labels, feats, use_hard_negatives=False)
    check("weights disabled -> all ones", np.allclose(off, 1.0))


# ═════════════════════════════════════════════════════════════════════════════
# E5 / E8 — Per-source thresholds, barrier, assembly
# ═════════════════════════════════════════════════════════════════════════════
def test_assembly_and_thresholds():
    print("\n[3] E5/E8 - Assembly, per-source thresholds, singleton barrier")
    s1 = ["S1-A", "S1-A", "S1-B", "S1-C"]
    tgt = ["S2-1", "S3-1", "S2-2", "S3-2"]
    probs = np.array([0.90, 0.80, 0.60, 0.70])
    all_s1 = {"S1-A", "S1-B", "S1-C", "S1-D"}

    check("infer_source", infer_source("S2-1") == "S2" and infer_source("S3-9") == "S3")

    pred = assemble_predictions(s1, tgt, probs, threshold=0.65, all_s1_ids=all_s1)
    check("global threshold assembly",
          pred["S1-A"] == {"S2-1", "S3-1"} and pred["S1-B"] == set()
          and pred["S1-C"] == {"S3-2"} and pred["S1-D"] == set(), f"{pred}")

    pred_src = assemble_predictions(
        s1, tgt, probs, thresholds_by_source={"S2": 0.95, "S3": 0.65},
        all_s1_ids=all_s1)
    check("per-source thresholds",
          pred_src["S1-A"] == {"S3-1"} and pred_src["S1-C"] == {"S3-2"},
          f"{pred_src}")

    pred_bar = assemble_predictions(
        s1, tgt, probs, threshold=0.65, all_s1_ids=all_s1, barrier=0.85)
    check("singleton barrier strips weak entity",
          pred_bar["S1-A"] == {"S2-1", "S3-1"} and pred_bar["S1-C"] == set(),
          f"{pred_bar}")

    reject = np.array([True, False, False, False])
    pred_rej = assemble_predictions(
        s1, tgt, probs, threshold=0.65, all_s1_ids=all_s1, reject_mask=reject)
    check("reject mask removes offending pair",
          pred_rej["S1-A"] == {"S3-1"}, f"{pred_rej}")

    # A source with systematically over-confident false positives should get a
    # higher tuned threshold than the other source.
    gt = {"S1-A": {"S3-0"}}
    s1b, tgtb, probsb = [], [], []
    for i in range(40):
        # S2 candidates: high prob but wrong
        s1b.append("S1-A"); tgtb.append(f"S2-{i}"); probsb.append(0.80)
        # S3 candidates: correct one at 0.70, wrong ones below
        s1b.append("S1-A"); tgtb.append(f"S3-{i}"); probsb.append(0.70 if i == 0 else 0.20)
    thresholds, score, _ = threshold_sweep_by_source(
        s1b, tgtb, np.array(probsb), gt,
        thresholds=[0.3, 0.5, 0.75, 0.85])
    check("per-source tuning raises S2 bar",
          thresholds["S2"] > thresholds["S3"], f"{thresholds}")
    global_t, global_score, _ = threshold_sweep(
        s1b, tgtb, np.array(probsb), gt, thresholds=[0.3, 0.5, 0.75, 0.85])
    check("per-source tuning >= global", score >= global_score - 1e-9,
          f"{score} vs {global_score}")

    # Barrier sweep never scores below baseline (0.0 is in the candidate list).
    barrier, bscore, _ = sweep_singleton_barrier(
        s1b, tgtb, np.array(probsb), gt, threshold=global_t)
    check("barrier sweep >= baseline", bscore >= global_score - 1e-9,
          f"{bscore} vs {global_score}")


# ═════════════════════════════════════════════════════════════════════════════
# E7 — Conservative decision rules
# ═════════════════════════════════════════════════════════════════════════════
def test_rules():
    print("\n[4] E7 - Conservative decision rules")
    feats = make_features([
        # dangerous: near-identical name, no address/postal confirmation
        {"name_levenshtein": 0.96, "addr_token_jaccard": 0.05,
         "addr_postal_match": 0.0, "addr_exact_match": 0.0},
        # confirmed by address
        {"name_levenshtein": 0.96, "addr_token_jaccard": 0.55,
         "addr_postal_match": 0.0, "addr_exact_match": 0.0},
        # confirmed by postal
        {"name_levenshtein": 0.96, "addr_token_jaccard": 0.05,
         "addr_postal_match": 1.0, "addr_exact_match": 0.0},
        # not high-name
        {"name_levenshtein": 0.70, "addr_token_jaccard": 0.05,
         "addr_postal_match": 0.0, "addr_exact_match": 0.0},
    ])
    mask = conservative_reject_mask(feats)
    check("rule rejects only the dangerous pair",
          list(mask) == [True, False, False, False], f"{list(mask)}")


# ═════════════════════════════════════════════════════════════════════════════
# E6 — Dual models, save/load
# ═════════════════════════════════════════════════════════════════════════════
def test_dual_models():
    print("\n[5] E6 - Dual source-specific models")
    rng = np.random.RandomState(0)
    n = 240
    feats = make_features([
        {"name_exact_match": float(rng.rand() > 0.5),
         "name_levenshtein": rng.rand(),
         "addr_token_jaccard": rng.rand()}
        for _ in range(n)
    ])
    labels = (feats["name_levenshtein"].to_numpy()
              + feats["addr_token_jaccard"].to_numpy() > 1.1).astype(float)
    target_ids = ["S2-x" if i % 2 == 0 else "S3-x" for i in range(n)]

    (Xtr, Xva, ytr, yva, s1tr, s1va, ttr, tva, val_set) = split_by_s1_entity(
        [f"S1-{i % 60}" for i in range(n)], target_ids, feats, labels, val_frac=0.25)
    # Guarantee both classes in training.
    if ytr.sum() == 0:
        ytr[:5] = 1.0

    params = {"n_estimators": 25, "learning_rate": 0.15, "num_leaves": 7,
              "min_child_samples": 5}
    models = train_dual_models(Xtr, ytr, ttr, Xva, yva, tva, params=params)
    check("dual models has S2 and S3", set(models) == {"S2", "S3"}, f"{set(models)}")

    probs = predict_dual_probabilities(models, Xva, tva)
    check("dual prediction length", len(probs) == len(Xva))
    check("dual prediction in [0,1]", bool((probs >= 0).all() and (probs <= 1).all()))

    # Save/load in a temporary models dir (never touch real artifacts).
    import src.train as train_mod
    original = train_mod.MODELS_DIR
    with tempfile.TemporaryDirectory() as tmp:
        train_mod.MODELS_DIR = Path(tmp)
        try:
            name = "unit_dual"
            save_dual_models(models, name=name, metadata={"best_threshold": 0.7})
            loaded, meta = load_dual_models(name=name)
            check("dual save/load roundtrip", set(loaded) == {"S2", "S3"},
                  f"{set(loaded)}")
            check("dual metadata roundtrip", meta.get("best_threshold") == 0.7)
        finally:
            train_mod.MODELS_DIR = original


# ═════════════════════════════════════════════════════════════════════════════
# End-to-end blocking + output format + official validator
# ═════════════════════════════════════════════════════════════════════════════
def test_blocking_and_output():
    print("\n[6] Blocking end-to-end + submission format")
    s1 = normalize_dataframe(pd.DataFrame([
        raw("smith enterprises", "9 beta avenue abc 99999", "US"),
        raw("lone wolf llc", "1 nowhere road 55555", "US"),
    ]).assign(entity_id=["S1-1", "S1-2"]), "S1")
    s2 = normalize_dataframe(pd.DataFrame([
        raw("smyth enterprises", "1 alpha road xyz 11111", "US"),
    ]).assign(entity_id=["S2-1"]), "S2")
    s3 = normalize_dataframe(pd.DataFrame([
        raw("smith enterprises", "9 beta avenue abc 99999", "US"),
    ]).assign(entity_id=["S3-1"]), "S3")

    cands = generate_all_candidates(s1, s2, s3, max_candidates=100, show_progress=False)
    check("candidates generated for both S1", set(cands) == {"S1-1", "S1-2"})
    check("true match (S3-1) retrieved", "S3-1" in cands["S1-1"], f"{cands['S1-1']}")

    all_s1 = {"S1-1", "S1-2"}
    matches = assemble_matches(["S1-1"], ["S3-1"], np.array([0.9]), 0.7, all_s1)
    candidates = {"S1-1": {"S3-1", "S2-1"}, "S1-2": set()}
    check("assembly keeps singleton empty", matches["S1-2"] == set())

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mpath = tmp / "matching_results.tsv"
        cpath = tmp / "candidate_pairs.tsv"
        save_matching_results(matches, output_path=mpath)
        save_candidate_pairs(candidates, output_path=cpath)

        raw_text = mpath.read_text(encoding="utf-8")
        check("output is tab-separated with expected header",
              raw_text.splitlines()[0] == "source1_entity_id\tmatched_entity_ids")

        errors = validate_output(
            matches, candidates,
            {"S1-1", "S1-2"}, {"S2-1"}, {"S3-1"})
        check("in-pipeline validate_output clean", errors == [], f"{errors}")

        # Official validator end-to-end on synthetic test dir + outputs.
        test_dir = tmp / "test"
        test_dir.mkdir()
        (test_dir / "test_source1.tsv").write_text(
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            "S1-1\tA\tB\tUS\nS1-2\tC\tD\tUS\n", encoding="utf-8")
        (test_dir / "test_source2.tsv").write_text(
            "entity_id\tbusiness_name\tbusiness_address\tcountry\nS2-1\tA\tB\tUS\n",
            encoding="utf-8")
        (test_dir / "test_source3.tsv").write_text(
            "entity_id\tbusiness_name\tbusiness_address\tcountry\nS3-1\tA\tB\tUS\n",
            encoding="utf-8")
        validator = Path(__file__).resolve().parent.parent / "utils" / "validate_submission.py"
        proc = subprocess.run(
            [sys.executable, str(validator), "--matching", str(mpath),
             "--candidate", str(cpath), "--test-dir", str(test_dir), "--check-ids"],
            capture_output=True, text=True, encoding="utf-8")
        check("official validator PASS", proc.returncode == 0,
              f"rc={proc.returncode}\n{proc.stdout}\n{proc.stderr}")


def test_blocking_recall_fixes():
    print("\n[7] Blocking recall fixes (round-robin + new indexes)")
    df = normalize_dataframe(pd.DataFrame([
        raw("acme corp", "12345 main street springfield 11111", "US"),
    ]).assign(entity_id=["S2-1"]), "S2")
    idx = build_candidate_indices(
        df, rare_tokens=set(), rare_addr_tokens={"springfield"})
    check("single-numeric index built",
          "num_single" in idx and any("12345" in k for k in idx["num_single"]),
          f"{list(idx.get('num_single', {}))[:3]}")
    check("address rare-token index built",
          "addr_rare" in idx and any("springfield" in k for k in idx["addr_rare"]),
          f"{list(idx.get('addr_rare', {}))[:3]}")

    # A huge exact-name block must not starve the relaxed-name block out of the cap.
    s1 = normalize_dataframe(pd.DataFrame([
        raw("alpha beta", "1 road 22222", "US"),
    ]).assign(entity_id=["S1-1"]), "S1")
    big = [f"S2-{i:05d}" for i in range(300)]
    indices = {
        "exact_name": {"us|alpha beta": big},
        "relaxed_name": {"us|rel|alph": ["TARGET"]},
    }
    ranked = _ranked_candidates_for_s1(
        s1.iloc[0], indices, rare_tokens=set(), max_candidates=100)
    check("late block not starved by cap", "TARGET" in ranked, f"size={len(ranked)}")
    check("cap respected with round-robin", len(ranked) == 100, f"{len(ranked)}")


def main():
    print("=" * 70)
    print("  Model Improvement Verification (E4-E8 + Index 6)")
    print("=" * 70)
    test_soundex()
    test_hard_negatives()
    test_assembly_and_thresholds()
    test_rules()
    test_dual_models()
    test_blocking_and_output()
    test_blocking_recall_fixes()

    print("\n" + "=" * 70)
    print(f"  RESULT: {_PASS} passed, {_FAIL} failed")
    print("=" * 70)
    return 1 if _FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
