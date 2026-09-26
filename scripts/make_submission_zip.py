#!/usr/bin/env python3
"""
make_submission_zip.py — Assemble the final submission archive.

Produces::

    <team>_submission.zip
    ├── output/matching_results.tsv
    ├── output/candidate_pairs.tsv
    ├── code/business_entity_resolution/
    │   ├── src/*.py
    │   ├── README.md
    │   └── requirements.txt
    └── Documentation_template.md

Usage:
    python scripts/make_submission_zip.py --team <team_name>
    python scripts/make_submission_zip.py --team x --allow-missing   # structure check
"""
import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CODE_README = """# Business Entity Resolution — Reproduction Guide

Self-contained implementation of the Amazon ML Challenge 2026 Business Entity
Resolution pipeline (blocking + LightGBM pair classifier, macro-F0.5 tuned).

## Setup
```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Data layout (tab-separated; `sep="\\t"`)
```
dataset/train/{train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv}
dataset/test/{test_source1.tsv, test_source2.tsv, test_source3.tsv}
```

## Reproduce the submission
```bash
# 1. Train (aligned sample; omit --sample for full data on a large machine)
python -m src.pipeline --mode train --sample 120000 --max-candidates 250

# 2. Test inference (chunked/streaming; bounded memory)
python -m src.pipeline --mode predict --max-candidates 20 --chunk-size 25000

# 3. Validate (must print PASS)
python utils/validate_submission.py \\
    --matching output/matching_results.tsv \\
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

## Notes
- All datasets are TSV; read/write with `sep="\\t"` (addresses and ID lists contain commas).
- `country` is treated as an open string (test includes France) — no hard-coded countries.
- Memory is bounded by `--chunk-size` for inference and by `--sample` for training.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Assemble the final submission ZIP.")
    ap.add_argument("--team", default="team", help="Team name used in the zip filename.")
    ap.add_argument("--out", default=None, help="Output zip path (default submissions/<team>_submission.zip).")
    ap.add_argument("--allow-missing", action="store_true",
                    help="Build even if output TSVs are missing (structure check only).")
    args = ap.parse_args()

    matching = ROOT / "output" / "matching_results.tsv"
    candidate = ROOT / "output" / "candidate_pairs.tsv"
    missing = [str(p) for p in (matching, candidate) if not p.exists()]
    if missing and not args.allow_missing:
        raise SystemExit(
            "Missing output file(s): " + ", ".join(missing) +
            "\nRun `python -m src.pipeline --mode predict` first "
            "(or pass --allow-missing to check structure).")

    out_zip = Path(args.out) if args.out else ROOT / "submissions" / f"{args.team}_submission.zip"
    out_zip.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for p in (matching, candidate):
            if p.exists():
                z.write(p, f"output/{p.name}")
        for p in sorted((ROOT / "src").glob("*.py")):
            z.write(p, f"code/business_entity_resolution/src/{p.name}")
        z.write(ROOT / "requirements.txt", "code/business_entity_resolution/requirements.txt")
        z.writestr("code/business_entity_resolution/README.md", CODE_README)
        doc = ROOT / "Documentation_template.md"
        if doc.exists():
            z.write(doc, "Documentation_template.md")

    size_mb = out_zip.stat().st_size / 1e6
    print(f"Wrote {out_zip} ({size_mb:.1f} MB)")
    with zipfile.ZipFile(out_zip) as z:
        for name in z.namelist():
            print(f"  {name}")


if __name__ == "__main__":
    main()
