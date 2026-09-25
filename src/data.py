"""
data.py — Data loaders.
"""
import pandas as pd
from config import DATA_RAW, DATA_PROC, TARGET_COL, ID_COL


def load_raw(train_file="train.csv", test_file="test.csv"):
    train = pd.read_csv(DATA_RAW / train_file)
    test  = pd.read_csv(DATA_RAW / test_file)
    return train, test


def basic_info(df: pd.DataFrame, name: str = "df") -> None:
    print(f"\n{'='*40}\n{name}: {df.shape}")
    print(df.dtypes.value_counts())
    print(f"\nMissing:\n{df.isna().sum()[df.isna().sum() > 0]}")
    print(f"\nDuplicates: {df.duplicated().sum()}")
