"""Rescore an existing complete candidate TSV with the latest trained model.

This skips expensive candidate generation while retaining a full S1 row for
every test entity. The existing candidate file must have already been
validated against the current test set.
"""
from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

import pandas as pd

from src.config import ID_COL, MATCHING_OUT, CANDIDATE_OUT
from src.features import TargetLookup, build_feature_matrix, restrict_to_core_columns
from src.io import load_source
from src.normalize import normalize_all_sources
from src.predict import TsvListWriter, assemble_matches, conservative_reject_mask, predict_probabilities
from src.train import load_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, help="Validated full candidate TSV")
    parser.add_argument("--rollback-matching", required=True, help="Prior complete matching TSV to preserve until success")
    parser.add_argument("--chunk-size", type=int, default=10_000)
    parser.add_argument("--resume", action="store_true", help="Resume an existing partial matching file")
    args = parser.parse_args()
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")

    candidate_path = Path(args.candidates)
    rollback_matching = Path(args.rollback_matching)
    matching_path = Path(MATCHING_OUT)
    candidate_out = Path(CANDIDATE_OUT)
    partial_path = matching_path.with_name(matching_path.stem + ".rescored.partial.tsv")
    candidate_out.parent.mkdir(parents=True, exist_ok=True)

    model, metadata = load_model()
    threshold = float(metadata.get("best_threshold", 0.8))
    thresholds_by_source = metadata.get("thresholds_by_source")
    barrier = float(metadata.get("barrier", 0.0) or 0.0)
    use_rules = bool(metadata.get("use_conservative_rules", True))
    print(f"Model threshold={threshold}; barrier={barrier}; source thresholds={thresholds_by_source}", flush=True)

    print("Loading full test sources", flush=True)
    started = time.time()
    s1, s2, s3 = (load_source(i, "test") for i in (1, 2, 3))
    print(f"Loaded S1/S2/S3 rows: {len(s1):,}/{len(s2):,}/{len(s3):,} ({time.time()-started:.1f}s)", flush=True)
    started = time.time()
    s1, s2, s3 = normalize_all_sources(s1, s2, s3, n_workers=1)
    s1 = restrict_to_core_columns(s1)
    s2 = restrict_to_core_columns(s2)
    s3 = restrict_to_core_columns(s3)
    print(f"Normalized test sources ({time.time()-started:.1f}s)", flush=True)

    # Preserve complete files in output until the new matching file is done.
    shutil.copyfile(candidate_path, candidate_out)
    rollback_tmp = matching_path.with_name(matching_path.name + ".rollback.tmp")
    shutil.copyfile(rollback_matching, rollback_tmp)
    os.replace(rollback_tmp, matching_path)
    s1_index = pd.Index(s1[ID_COL].to_numpy())
    target_df = pd.concat([s2, s3], ignore_index=True)
    target_lookup = TargetLookup(target_df)
    del s2, s3
    done = 0
    if args.resume and partial_path.exists():
        with partial_path.open("r", encoding="utf-8") as f:
            done = max(0, sum(1 for _ in f) - 1)
        print(f"Resuming after {done:,} rows", flush=True)
    elif partial_path.exists():
        partial_path.unlink()

    rows_written = done
    with candidate_path.open("r", encoding="utf-8") as candidates_file, \
            TsvListWriter(partial_path, "matched_entity_ids", append=bool(done)) as writer:
        header = candidates_file.readline().rstrip("\r\n").split("\t")
        if header != ["source1_entity_id", "candidate_entity_ids"]:
            raise ValueError(f"Unexpected candidate header: {header}")
        for _ in range(done):
            if not candidates_file.readline():
                raise ValueError("Partial matching file is longer than candidate input")
        batch = []
        chunk_number = 0
        for line in candidates_file:
            s1_id, sep, payload = line.rstrip("\r\n").partition("\t")
            if not sep or not s1_id:
                raise ValueError(f"Malformed candidate row near row {rows_written + len(batch) + 2}")
            cand_ids = set(payload.split(",")) if payload else set()
            batch.append((s1_id, cand_ids))
            if len(batch) < args.chunk_size:
                continue
            rows_written += _score_batch(batch, s1, s1_index, target_df, target_lookup,
                                         model, metadata, threshold, thresholds_by_source,
                                         barrier, use_rules, writer)
            chunk_number += 1
            print(f"Scored {rows_written:,}/{len(s1):,} S1 rows (chunk {chunk_number})", flush=True)
            batch.clear()
        if batch:
            rows_written += _score_batch(batch, s1, s1_index, target_df, target_lookup,
                                         model, metadata, threshold, thresholds_by_source,
                                         barrier, use_rules, writer)
            print(f"Scored {rows_written:,}/{len(s1):,} S1 rows", flush=True)

    if rows_written != len(s1):
        raise ValueError(f"Wrote {rows_written:,} rows but expected {len(s1):,}")
    os.replace(partial_path, matching_path)
    print(f"Complete: {rows_written:,} S1 rows; matching={matching_path}; candidates={candidate_out}", flush=True)
    return 0


def _score_batch(batch, s1, s1_index, target_df, target_lookup, model, metadata,
                 threshold, thresholds_by_source, barrier, use_rules, writer):
    ids = [item[0] for item in batch]
    positions = s1_index.get_indexer(ids)
    if (positions < 0).any():
        missing = [ids[i] for i, pos in enumerate(positions) if pos < 0][:5]
        raise ValueError(f"Candidate input contains unknown S1 IDs: {missing}")
    s1_chunk = s1.iloc[positions].reset_index(drop=True)
    candidate_map = {sid: cands for sid, cands in batch}
    features, pair_s1, pair_targets = build_feature_matrix(
        s1_chunk, target_df, candidate_map, show_progress=False,
        target_lookup=target_lookup, n_workers=1)
    if len(features):
        features = features.astype("float32")
        probabilities = predict_probabilities(model, features)
        reject = conservative_reject_mask(features) if use_rules else None
    else:
        probabilities = []
        reject = None
    matches = assemble_matches(
        pair_s1, pair_targets, probabilities, threshold=threshold,
        all_s1_ids=set(ids), thresholds_by_source=thresholds_by_source,
        barrier=barrier, reject_mask=reject)
    for sid in ids:
        writer.write(sid, matches.get(sid, set()))
    return len(batch)


if __name__ == "__main__":
    raise SystemExit(main())
