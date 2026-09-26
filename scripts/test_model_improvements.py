"""
test_model_improvements.py — Self-contained checks for the E4-E8 model upgrades.

Runs entirely on tiny synthetic data (no challenge dataset needed), so it is
safe to run under tight memory. Exits non-zero if any check fails.

Usage:
    python scripts/test_model_improvements.py
"""
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.normalize import normalize_dataframe, normalize_all_sources
from src.blocking import (
    soundex, build_candidate_indices, _ranked_candidates_for_s1,
    generate_all_candidates, build_all_indices, generate_candidates_from_bundle,
    _generate_candidates_parallel,
)
from src.features import (
    FEATURE_NAMES, TargetLookup, restrict_to_core_columns, build_feature_matrix,
    _build_features_parallel, _build_features_spawn, compute_pair_features, token_jaccard,
    char_ngram_jaccard, containment_ratio, token_overlap_count,
    numeric_token_jaccard, numeric_token_overlap, postal_match, length_diff_ratio,
)
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
    TsvListWriter, run_chunked_inference,
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


def test_chunked_inference():
    print("\n[8] Chunked streaming inference")
    s1 = restrict_to_core_columns(normalize_dataframe(pd.DataFrame([
        raw("smith enterprises", "9 beta avenue abc 99999", "US"),
        raw("acme corp", "1 main st 11111", "US"),
        raw("lone wolf", "2 nowhere rd 55555", "US"),
    ]).assign(entity_id=["S1-1", "S1-2", "S1-3"]), "S1"))
    s2 = restrict_to_core_columns(normalize_dataframe(pd.DataFrame([
        raw("smyth enterprises", "1 alpha road xyz 11111", "US"),
    ]).assign(entity_id=["S2-1"]), "S2"))
    s3 = restrict_to_core_columns(normalize_dataframe(pd.DataFrame([
        raw("smith enterprises", "9 beta avenue abc 99999", "US"),
    ]).assign(entity_id=["S3-1"]), "S3"))
    target_df = pd.concat([s2, s3], ignore_index=True)

    # TargetLookup
    tl = TargetLookup(target_df)
    pos = tl.positions(["S3-1", "NOPE"])
    check("TargetLookup resolves ids + missing -> -1",
          pos[0] >= 0 and pos[1] == -1, f"{pos}")
    vals = tl.values(pos[0])
    check("TargetLookup field order",
          vals[0] == "smith enterprise" and vals[7] == "S3", f"{vals[:2] + vals[7:]}")

    # Prebuilt bundle / index reuse
    bundle = build_all_indices(s2, s3)
    cands = generate_candidates_from_bundle(s1, bundle, show_progress=False)
    check("bundle candidate gen covers all S1",
          set(cands) == {"S1-1", "S1-2", "S1-3"})
    check("bundle retrieves true match", "S3-1" in cands["S1-1"], f"{cands['S1-1']}")

    # Feature matrix with prebuilt lookup must match the default path
    f1, si1, ti1 = build_feature_matrix(s1, target_df, cands, show_progress=False)
    f2, si2, ti2 = build_feature_matrix(
        s1, target_df, cands, show_progress=False, target_lookup=tl)
    check("prebuilt lookup yields identical features",
          f1.equals(f2) and si1 == si2 and ti1 == ti2)

    # Streaming inference across multiple chunks
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mpath = tmp / "matching_results.tsv"
        cpath = tmp / "candidate_pairs.tsv"

        def score_fn(feats, tids):
            return np.array([0.9 if t == "S3-1" else 0.1 for t in tids])

        summary = run_chunked_inference(
            s1, s2, s3, score_fn, threshold=0.7, use_rules=False,
            chunk_size=1, matching_path=mpath, candidate_path=cpath,
            show_progress=False)
        check("chunked wrote every S1", summary["s1_written"] == 3, f"{summary}")
        check("chunked superset clean", summary["superset_violations"] == 0)

        ml = mpath.read_text(encoding="utf-8").splitlines()
        check("chunked matching header",
              ml[0] == "source1_entity_id\tmatched_entity_ids")
        d = {ln.split("\t")[0]: ln.split("\t")[1] for ln in ml[1:]}
        check("chunked match + singleton rows",
              d == {"S1-1": "S3-1", "S1-2": "", "S1-3": ""}, f"{d}")

        cl = cpath.read_text(encoding="utf-8").splitlines()
        cd = {ln.split("\t")[0]: (set(ln.split("\t")[1].split(","))
                                  if ln.split("\t")[1] else set())
              for ln in cl[1:]}
        check("chunked candidates superset matches",
              all(({d[k]} if d[k] else set()) <= cd[k] for k in d))

        # TsvListWriter direct format check
        wp = tmp / "w.tsv"
        with TsvListWriter(wp, "matched_entity_ids") as w:
            w.write("S1-1", {"S3-1", "S2-1"})
            w.write("S1-2", set())
        wt = wp.read_text(encoding="utf-8")
        check("TsvListWriter sorted + empty singleton",
              wt == "source1_entity_id\tmatched_entity_ids\n"
                    "S1-1\tS2-1,S3-1\nS1-2\t\n", repr(wt))


def test_parallel_features():
    print("\n[9] Parallel feature building")
    s1 = restrict_to_core_columns(normalize_dataframe(pd.DataFrame([
        raw("smith enterprises", "9 beta avenue abc 99999", "US"),
        raw("acme corp", "1 main st 11111", "US"),
    ]).assign(entity_id=["S1-1", "S1-2"]), "S1"))
    s2 = restrict_to_core_columns(normalize_dataframe(pd.DataFrame([
        raw("smyth enterprises", "1 alpha road xyz 11111", "US"),
    ]).assign(entity_id=["S2-1"]), "S2"))
    s3 = restrict_to_core_columns(normalize_dataframe(pd.DataFrame([
        raw("smith enterprises", "9 beta avenue abc 99999", "US"),
    ]).assign(entity_id=["S3-1"]), "S3"))
    target_df = pd.concat([s2, s3], ignore_index=True)
    bundle = build_all_indices(s2, s3)
    cands = generate_candidates_from_bundle(s1, bundle, show_progress=False)
    tl = TargetLookup(target_df)

    f_ser, si_s, ti_s = build_feature_matrix(
        s1, target_df, cands, show_progress=False, target_lookup=tl)
    if hasattr(os, "fork"):
        f_par, si_p, ti_p = _build_features_parallel(s1, tl, cands, 2)
        pname = "fork"
    else:
        f_par, si_p, ti_p = _build_features_spawn(s1, tl, cands, 2)
        pname = "spawn"
    # Parallel path returns float32; serial returns float64. Compare after casting.
    same_vals = np.array_equal(
        f_par.to_numpy(), f_ser.to_numpy().astype(np.float32))
    check(f"parallel features ({pname}) identical to serial",
          same_vals and si_p == si_s and ti_p == ti_s,
          f"{f_par.shape} vs {f_ser.shape}")

    if hasattr(os, "fork"):
        cands_par = _generate_candidates_parallel(s1, bundle, 100, 2)
        check("parallel candidate gen identical to serial",
              cands_par == cands, f"{cands_par} vs {cands}")

    # Parallel normalization equivalence
    r1 = pd.DataFrame([raw("Acme Pvt. Ltd.", "12 Main St 10001", "US")]).assign(entity_id=["A1"])
    r2 = pd.DataFrame([raw("Beta & Co", "9 Rue de Paris", "FR")]).assign(entity_id=["B1"])
    r3 = pd.DataFrame([raw("Gamma Corp", "Rue 5, Nice", "FR")]).assign(entity_id=["C1"])
    a = normalize_all_sources(r1, r2, r3, n_workers=1)
    b = normalize_all_sources(r1, r2, r3, n_workers=3)
    check("parallel normalize identical to serial",
          all(x.equals(y) for x, y in zip(a, b)))


def test_parallel_sweeps():
    print("\n[10] Parallel threshold / barrier sweeps")
    if not hasattr(os, "fork"):
        check("parallel sweeps skipped (no fork on this OS)", True)
        return
    s1 = ["S1-1", "S1-1", "S1-2", "S1-3"]
    tgt = ["S2-1", "S3-1", "S2-2", "S3-2"]
    probs = np.array([0.9, 0.8, 0.4, 0.6])
    gt = {"S1-1": {"S2-1", "S3-1"}, "S1-2": set(), "S1-3": set()}

    b_s, sc_s, _ = threshold_sweep(s1, tgt, probs, gt, n_workers=1)
    b_p, sc_p, _ = threshold_sweep(s1, tgt, probs, gt, n_workers=2)
    check("parallel threshold sweep == serial", (b_s, sc_s) == (b_p, sc_p),
          f"{(b_s, sc_s)} vs {(b_p, sc_p)}")

    bar_s, bs_s, _ = sweep_singleton_barrier(
        s1, tgt, probs, gt, threshold=b_s, n_workers=1)
    bar_p, bs_p, _ = sweep_singleton_barrier(
        s1, tgt, probs, gt, threshold=b_s, n_workers=2)
    check("parallel barrier sweep == serial", (bar_s, bs_s) == (bar_p, bs_p),
          f"{(bar_s, bs_s)} vs {(bar_p, bs_p)}")


def _reference_features(a, b):
    """Original (pre-optimization) feature logic, used to verify the fast path."""
    (s1_name, s1_addr, s1_country, s1_name_tokens, s1_addr_numeric,
     s1_addr_postal, s1_name_prefix) = a
    (t_name, t_addr, t_country, t_name_tokens, t_addr_numeric,
     t_addr_postal, t_name_prefix, t_source) = b

    name_exact = 1.0 if s1_name == t_name and s1_name else 0.0
    name_lev = fuzz.ratio(s1_name, t_name) / 100.0 if s1_name and t_name else 0.0
    name_jw = distance.JaroWinkler.similarity(s1_name, t_name) if s1_name and t_name else 0.0
    name_tsort = fuzz.token_sort_ratio(s1_name, t_name) / 100.0 if s1_name and t_name else 0.0
    name_tset = fuzz.token_set_ratio(s1_name, t_name) / 100.0 if s1_name and t_name else 0.0
    name_jacc = token_jaccard(s1_name_tokens, t_name_tokens)
    name_ngram = char_ngram_jaccard(s1_name, t_name)
    name_contain = containment_ratio(s1_name_tokens, t_name_tokens)
    name_tok_overlap = float(token_overlap_count(s1_name_tokens, t_name_tokens))
    name_len_d = length_diff_ratio(s1_name, t_name)
    name_pfx_match = 1.0 if s1_name_prefix == t_name_prefix and s1_name_prefix else 0.0
    if s1_name_tokens and t_name_tokens:
        sa, sb = set(s1_name_tokens.split()), set(t_name_tokens.split())
        total = len(sa | sb)
        name_common_ratio = len(sa & sb) / total if total > 0 else 0.0
    else:
        name_common_ratio = 0.0
    addr_exact = 1.0 if s1_addr == t_addr and s1_addr else 0.0
    addr_lev = fuzz.ratio(s1_addr, t_addr) / 100.0 if s1_addr and t_addr else 0.0
    addr_tok_jacc = token_jaccard(s1_addr, t_addr)
    addr_tok_overlap = float(token_overlap_count(s1_addr, t_addr))
    addr_ngram = char_ngram_jaccard(s1_addr, t_addr)
    addr_len_d = length_diff_ratio(s1_addr, t_addr)
    addr_num_jacc = numeric_token_jaccard(s1_addr_numeric, t_addr_numeric)
    addr_num_overlap = float(numeric_token_overlap(s1_addr_numeric, t_addr_numeric))
    addr_post_match = postal_match(s1_addr_postal, t_addr_postal)
    addr_contain = containment_ratio(s1_addr, t_addr)
    same_ctry = 1.0 if s1_country == t_country and s1_country else 0.0
    is_s2 = 1.0 if t_source == 'S2' else 0.0
    both_exact = 1.0 if name_exact and addr_exact else 0.0
    name_high_sim = name_lev > 0.85
    addr_high_sim = addr_lev > 0.70
    name_high_addr_h = 1.0 if name_high_sim and addr_high_sim else 0.0
    has_mis = (s1_addr_numeric and t_addr_numeric
               and not (set(s1_addr_numeric.split()) & set(t_addr_numeric.split())))
    name_high_addr_mis = 1.0 if name_lev > 0.90 and has_mis else 0.0
    name_high_postal = 1.0 if name_high_sim and addr_post_match else 0.0
    return [name_exact, name_lev, name_jw, name_tsort, name_tset, name_jacc, name_ngram,
            name_contain, name_tok_overlap, name_len_d, name_pfx_match, name_common_ratio,
            addr_exact, addr_lev, addr_tok_jacc, addr_tok_overlap, addr_ngram, addr_len_d,
            addr_num_jacc, addr_num_overlap, addr_post_match, addr_contain, same_ctry,
            is_s2, both_exact, name_high_addr_h, name_high_addr_mis, name_high_postal]


def test_feature_equivalence():
    print("\n[11] Optimized features match original logic")
    rng = random.Random(0)
    vocab = ["acme", "private", "limited", "traders", "shivam", "road", "street",
             "nagar", "store", "and", "laboratory", "12", "560001", "main"]

    def rand_text(maxn):
        return " ".join(rng.choice(vocab) for _ in range(rng.randint(0, maxn)))

    def rec():
        name = rand_text(3)
        addr = rand_text(5)
        toks = " ".join(sorted(name.split()))
        num = " ".join(t for t in addr.split() if t.isdigit())
        postal = ",".join(t for t in addr.split() if t.isdigit() and len(t) >= 5)
        pfx = name[:6]
        return name, addr, pfx, toks, num, postal

    mismatches = 0
    for _ in range(300):
        n1, a1, p1, t1, u1, pc1 = rec()
        n2, a2, p2, t2, u2, pc2 = rec()
        c1 = rng.choice(["us", "india", "france", ""])
        c2 = rng.choice(["us", "india", "france", ""])
        src = rng.choice(["S2", "S3"])
        got = compute_pair_features(
            n1, a1, c1, t1, u1, pc1, p1, n2, a2, c2, t2, u2, pc2, p2, src)
        exp = _reference_features(
            (n1, a1, c1, t1, u1, pc1, p1),
            (n2, a2, c2, t2, u2, pc2, p2, src))
        if not np.allclose(got, exp, rtol=0, atol=1e-12):
            mismatches += 1
            if mismatches == 1:
                print("   first mismatch:", list(zip(FEATURE_NAMES, got, exp)))
    check("optimized features identical over 300 random cases", mismatches == 0,
          f"{mismatches} mismatches")


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
    test_chunked_inference()
    test_parallel_features()
    test_parallel_sweeps()
    test_feature_equivalence()

    print("\n" + "=" * 70)
    print(f"  RESULT: {_PASS} passed, {_FAIL} failed")
    print("=" * 70)
    return 1 if _FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
