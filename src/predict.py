"""
predict.py — Inference + submission generation.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from config import ID_COL, TARGET_COL, SUBS_DIR


def make_submission(ids, preds, filename="submission.csv"):
    assert not pd.isnull(preds).any(), "NaN in predictions!"
    sub = pd.DataFrame({ID_COL: ids, TARGET_COL: preds})
    out = SUBS_DIR / filename
    sub.to_csv(out, index=False)
    print(f"Submission saved → {out}  ({len(sub)} rows)")
    return sub
