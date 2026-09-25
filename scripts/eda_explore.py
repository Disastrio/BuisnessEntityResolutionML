"""
EDA Script — Full data exploration for the Business Entity Resolution challenge.
Outputs an HTML report with all tables, charts, and sample records.
"""
import pandas as pd
import numpy as np
import os, sys, json
from collections import Counter

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_DIR = r"R:\amazon-ML\dataset\train"
TEST_DIR  = r"R:\amazon-ML\dataset\test"
REPORT    = r"R:\amazon-ML\reports\eda_report.html"

os.makedirs(os.path.dirname(REPORT), exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading training data...")
s1_train = pd.read_csv(f"{TRAIN_DIR}/train_source1.tsv", sep="\t")
s2_train = pd.read_csv(f"{TRAIN_DIR}/train_source2.tsv", sep="\t")
s3_train = pd.read_csv(f"{TRAIN_DIR}/train_source3.tsv", sep="\t")
gt_train = pd.read_csv(f"{TRAIN_DIR}/train_ground_truth.tsv", sep="\t")

print("Loading test data...")
s1_test = pd.read_csv(f"{TEST_DIR}/test_source1.tsv", sep="\t")
s2_test = pd.read_csv(f"{TEST_DIR}/test_source2.tsv", sep="\t")
s3_test = pd.read_csv(f"{TEST_DIR}/test_source3.tsv", sep="\t")

# ── Basic Stats ───────────────────────────────────────────────────────────────
html_parts = []

def h(tag, text, **attrs):
    a = " ".join(f'{k}="{v}"' for k, v in attrs.items())
    return f"<{tag} {a}>{text}</{tag}>"

def section(title, content):
    html_parts.append(f"<h2>{title}</h2>\n{content}")

def df_to_html(df, max_rows=20):
    return df.head(max_rows).to_html(index=True, border=1, classes="data-table")

# Header
html_parts.append("""
<html><head>
<meta charset="utf-8">
<title>Entity Resolution — Data Exploration</title>
<style>
body { font-family: 'Segoe UI', sans-serif; max-width: 1200px; margin: 40px auto; background: #0d1117; color: #c9d1d9; padding: 20px; }
h1 { color: #58a6ff; border-bottom: 2px solid #30363d; padding-bottom: 12px; }
h2 { color: #79c0ff; margin-top: 40px; border-left: 4px solid #58a6ff; padding-left: 12px; }
h3 { color: #a5d6ff; }
table.data-table { border-collapse: collapse; width: 100%; margin: 10px 0 20px; font-size: 13px; }
table.data-table th { background: #161b22; color: #58a6ff; padding: 10px; text-align: left; border: 1px solid #30363d; }
table.data-table td { padding: 8px 10px; border: 1px solid #21262d; word-break: break-all; max-width: 400px; }
table.data-table tr:nth-child(even) { background: #161b22; }
table.data-table tr:hover { background: #1c2128; }
.stat-box { display: inline-block; background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px 24px; margin: 8px; min-width: 160px; text-align: center; }
.stat-box .num { font-size: 28px; font-weight: bold; color: #58a6ff; }
.stat-box .label { font-size: 12px; color: #8b949e; margin-top: 4px; }
.highlight { background: #1f2937; border-left: 3px solid #f0883e; padding: 12px 16px; margin: 10px 0; border-radius: 4px; }
.sample-record { background: #161b22; border: 1px solid #30363d; padding: 14px; margin: 8px 0; border-radius: 6px; font-size: 13px; }
.sample-record strong { color: #79c0ff; }
pre { background: #161b22; padding: 12px; border-radius: 6px; overflow-x: auto; font-size: 13px; }
.bar { background: #238636; height: 20px; border-radius: 3px; display: inline-block; margin-right: 8px; }
</style>
</head><body>
<h1>🔍 Entity Resolution — Data Exploration Report</h1>
<p style="color:#8b949e;">Generated for the Amazon ML Hackathon | Business Entity Resolution Challenge</p>
""")

# ── 1. Overview ───────────────────────────────────────────────────────────────
overview_html = '<div style="display: flex; flex-wrap: wrap;">'
for name, df in [("S1 Train", s1_train), ("S2 Train", s2_train), ("S3 Train", s3_train),
                 ("S1 Test", s1_test), ("S2 Test", s2_test), ("S3 Test", s3_test),
                 ("Ground Truth", gt_train)]:
    overview_html += f'''
    <div class="stat-box">
        <div class="num">{len(df):,}</div>
        <div class="label">{name}<br>{df.shape[1]} cols</div>
    </div>'''
overview_html += '</div>'
section("1. Dataset Overview — Row Counts", overview_html)

# ── 2. Columns & Dtypes ──────────────────────────────────────────────────────
cols_html = "<h3>Source Files (S1/S2/S3)</h3>"
cols_df = pd.DataFrame({
    "Column": s1_train.columns,
    "Dtype": s1_train.dtypes.astype(str).values,
    "S1 Train NAs": s1_train.isna().sum().values,
    "S1 Train NA%": (s1_train.isna().mean() * 100).round(2).values,
    "S2 Train NAs": s2_train.isna().sum().values,
    "S2 Train NA%": (s2_train.isna().mean() * 100).round(2).values,
    "S3 Train NAs": s3_train.isna().sum().values,
    "S3 Train NA%": (s3_train.isna().mean() * 100).round(2).values,
})
cols_html += df_to_html(cols_df)

cols_html += "<h3>Ground Truth</h3>"
gt_cols = pd.DataFrame({
    "Column": gt_train.columns,
    "Dtype": gt_train.dtypes.astype(str).values,
    "NAs": gt_train.isna().sum().values,
    "NA%": (gt_train.isna().mean() * 100).round(2).values,
})
cols_html += df_to_html(gt_cols)
section("2. Columns & Missing Values", cols_html)

# ── 3. Sample Records ────────────────────────────────────────────────────────
samples_html = ""
for name, df in [("Source 1 (Train)", s1_train), ("Source 2 (Train)", s2_train), 
                 ("Source 3 (Train)", s3_train)]:
    samples_html += f"<h3>{name} — First 5 Records</h3>"
    samples_html += df_to_html(df, 5)
    samples_html += f"<h3>{name} — Random 5 Records</h3>"
    samples_html += df_to_html(df.sample(5, random_state=42).reset_index(drop=True), 5)

samples_html += "<h3>Ground Truth — First 10</h3>"
samples_html += df_to_html(gt_train, 10)

# Show some examples of multi-match and no-match
gt_train["match_count"] = gt_train["matched_entity_ids"].apply(
    lambda x: len(str(x).split(",")) if pd.notna(x) and str(x).strip() != "" else 0
)
samples_html += "<h3>Ground Truth — Multi-Match Examples (3+ matches)</h3>"
multi = gt_train[gt_train["match_count"] >= 3].head(10)
samples_html += df_to_html(multi)

samples_html += "<h3>Ground Truth — Singleton Examples (0 matches)</h3>"
singles = gt_train[gt_train["match_count"] == 0].head(10)
samples_html += df_to_html(singles)

section("3. Sample Records (See the Actual Data)", samples_html)

# ── 4. Country Distribution ──────────────────────────────────────────────────
country_html = ""
for name, df in [("S1 Train", s1_train), ("S2 Train", s2_train), ("S3 Train", s3_train),
                 ("S1 Test", s1_test), ("S2 Test", s2_test), ("S3 Test", s3_test)]:
    counts = df["country"].value_counts()
    pcts = (df["country"].value_counts(normalize=True) * 100).round(2)
    tbl = pd.DataFrame({"Count": counts, "Percentage": pcts})
    country_html += f"<h3>{name}</h3>"
    country_html += df_to_html(tbl)

section("4. Country Distribution (⚠️ France in Test!)", country_html)

# ── 5. Ground Truth Analysis ─────────────────────────────────────────────────
gt_html = ""

# Match count distribution
match_dist = gt_train["match_count"].value_counts().sort_index()
total_s1 = len(gt_train)
singletons = (gt_train["match_count"] == 0).sum()
has_match = (gt_train["match_count"] > 0).sum()

gt_html += f'''
<div style="display: flex; flex-wrap: wrap;">
    <div class="stat-box"><div class="num">{total_s1:,}</div><div class="label">Total S1 Entities</div></div>
    <div class="stat-box"><div class="num">{singletons:,}</div><div class="label">Singletons (0 matches)<br>{singletons/total_s1*100:.1f}%</div></div>
    <div class="stat-box"><div class="num">{has_match:,}</div><div class="label">With Matches<br>{has_match/total_s1*100:.1f}%</div></div>
    <div class="stat-box"><div class="num">{gt_train["match_count"].max()}</div><div class="label">Max Matches per S1</div></div>
    <div class="stat-box"><div class="num">{gt_train["match_count"].mean():.2f}</div><div class="label">Avg Matches per S1</div></div>
</div>
'''

gt_html += "<h3>Match Count Distribution</h3>"
gt_html += "<table class='data-table'><tr><th># Matches</th><th>Count</th><th>%</th><th>Visual</th></tr>"
for mc, cnt in match_dist.items():
    pct = cnt / total_s1 * 100
    bar_w = max(1, int(pct * 5))
    gt_html += f"<tr><td>{mc}</td><td>{cnt:,}</td><td>{pct:.2f}%</td><td><div class='bar' style='width:{bar_w}px'></div></td></tr>"
gt_html += "</table>"

# Source breakdown in matches
all_match_ids = []
for ids_str in gt_train["matched_entity_ids"].dropna():
    ids_str = str(ids_str).strip()
    if ids_str:
        all_match_ids.extend([x.strip() for x in ids_str.split(",")])

s2_matches = [x for x in all_match_ids if x.startswith("S2-")]
s3_matches = [x for x in all_match_ids if x.startswith("S3-")]

gt_html += f'''
<div class="highlight">
<strong>Match Source Breakdown:</strong><br>
Total matched IDs: <strong>{len(all_match_ids):,}</strong><br>
From S2: <strong>{len(s2_matches):,}</strong> ({len(s2_matches)/max(len(all_match_ids),1)*100:.1f}%)<br>
From S3: <strong>{len(s3_matches):,}</strong> ({len(s3_matches)/max(len(all_match_ids),1)*100:.1f}%)
</div>'''

section("5. Ground Truth Analysis (Critical for Strategy)", gt_html)

# ── 6. Text Length Analysis ───────────────────────────────────────────────────
len_html = ""
for name, df in [("S1 Train", s1_train), ("S2 Train", s2_train), ("S3 Train", s3_train)]:
    stats = pd.DataFrame({
        "Field": ["business_name", "business_address"],
        "Min Len": [df["business_name"].str.len().min(), df["business_address"].str.len().min()],
        "Max Len": [df["business_name"].str.len().max(), df["business_address"].str.len().max()],
        "Mean Len": [df["business_name"].str.len().mean().round(1), df["business_address"].str.len().mean().round(1)],
        "Median Len": [df["business_name"].str.len().median(), df["business_address"].str.len().median()],
        "Std Len": [df["business_name"].str.len().std().round(1), df["business_address"].str.len().std().round(1)],
        "Empty/NaN": [df["business_name"].isna().sum() + (df["business_name"].str.strip() == "").sum(),
                      df["business_address"].isna().sum() + (df["business_address"].str.strip() == "").sum() if df["business_address"].notna().any() else "N/A"],
    })
    len_html += f"<h3>{name}</h3>"
    len_html += df_to_html(stats)

section("6. Text Length Analysis", len_html)

# ── 7. Uniqueness & Duplicates ────────────────────────────────────────────────
dup_html = ""
for name, df in [("S1 Train", s1_train), ("S2 Train", s2_train), ("S3 Train", s3_train),
                 ("S1 Test", s1_test), ("S2 Test", s2_test), ("S3 Test", s3_test)]:
    n_total = len(df)
    n_unique_names = df["business_name"].nunique()
    n_unique_addr = df["business_address"].nunique()
    n_unique_pairs = df[["business_name", "business_address"]].drop_duplicates().shape[0]
    dup_html += f'''
    <h3>{name}</h3>
    <table class="data-table">
    <tr><th>Measure</th><th>Count</th><th>% Unique</th></tr>
    <tr><td>Total rows</td><td>{n_total:,}</td><td>—</td></tr>
    <tr><td>Unique entity_ids</td><td>{df["entity_id"].nunique():,}</td><td>{df["entity_id"].nunique()/n_total*100:.2f}%</td></tr>
    <tr><td>Unique business_names</td><td>{n_unique_names:,}</td><td>{n_unique_names/n_total*100:.2f}%</td></tr>
    <tr><td>Unique business_addresses</td><td>{n_unique_addr:,}</td><td>{n_unique_addr/n_total*100:.2f}%</td></tr>
    <tr><td>Unique (name, address) pairs</td><td>{n_unique_pairs:,}</td><td>{n_unique_pairs/n_total*100:.2f}%</td></tr>
    </table>'''
section("7. Uniqueness & Duplicate Analysis", dup_html)

# ── 8. Most Common Business Names ─────────────────────────────────────────────
common_html = ""
for name, df in [("S1 Train", s1_train), ("S2 Train", s2_train), ("S3 Train", s3_train)]:
    top = df["business_name"].value_counts().head(15)
    tbl = pd.DataFrame({"Name": top.index, "Count": top.values})
    common_html += f"<h3>{name} — Top 15 Most Common Names</h3>"
    common_html += df_to_html(tbl)
section("8. Most Common Business Names (⚠️ High False-Positive Risk)", common_html)

# ── 9. Train vs Test Comparison ───────────────────────────────────────────────
compare_html = ""
compare_data = pd.DataFrame({
    "": ["Source 1", "Source 2", "Source 3"],
    "Train Rows": [f"{len(s1_train):,}", f"{len(s2_train):,}", f"{len(s3_train):,}"],
    "Test Rows": [f"{len(s1_test):,}", f"{len(s2_test):,}", f"{len(s3_test):,}"],
    "Train Countries": [str(s1_train["country"].unique().tolist()), str(s2_train["country"].unique().tolist()), str(s3_train["country"].unique().tolist())],
    "Test Countries": [str(s1_test["country"].unique().tolist()), str(s2_test["country"].unique().tolist()), str(s3_test["country"].unique().tolist())],
})
compare_html += df_to_html(compare_data)

# Name overlap
s1_names_tr = set(s1_train["business_name"].str.lower().str.strip())
s2_names_tr = set(s2_train["business_name"].str.lower().str.strip())
s3_names_tr = set(s3_train["business_name"].str.lower().str.strip())
s1_names_te = set(s1_test["business_name"].str.lower().str.strip())

compare_html += f'''
<div class="highlight">
<strong>Name Overlaps (case-insensitive):</strong><br>
S1 Train ∩ S2 Train: <strong>{len(s1_names_tr & s2_names_tr):,}</strong> names in common<br>
S1 Train ∩ S3 Train: <strong>{len(s1_names_tr & s3_names_tr):,}</strong> names in common<br>
S1 Train ∩ S1 Test: <strong>{len(s1_names_tr & s1_names_te):,}</strong> names in common (train→test leak check)
</div>'''

section("9. Train vs Test Comparison", compare_html)

# ── 10. Sample Matching Pairs ─────────────────────────────────────────────────
pair_html = ""
# Show 10 real matching examples so human can see what matches look like
s2_map = s2_train.set_index("entity_id")
s3_map = s3_train.set_index("entity_id")
s1_map = s1_train.set_index("entity_id")

shown = 0
for _, row in gt_train[gt_train["match_count"] > 0].head(50).iterrows():
    if shown >= 10:
        break
    s1_id = row["source1_entity_id"]
    if s1_id not in s1_map.index:
        continue
    s1_rec = s1_map.loc[s1_id]
    match_ids = [x.strip() for x in str(row["matched_entity_ids"]).split(",") if x.strip()]
    
    pair_html += f'<div class="sample-record">'
    pair_html += f'<strong>S1 [{s1_id}]:</strong> {s1_rec.get("business_name", "?")} | {s1_rec.get("business_address", "?")} | {s1_rec.get("country", "?")}<br>'
    for mid in match_ids[:3]:  # show up to 3 matches
        if mid.startswith("S2-") and mid in s2_map.index:
            mrec = s2_map.loc[mid]
        elif mid.startswith("S3-") and mid in s3_map.index:
            mrec = s3_map.loc[mid]
        else:
            continue
        pair_html += f'&nbsp;&nbsp;↳ <strong style="color:#3fb950">{mid}:</strong> {mrec.get("business_name", "?")} | {mrec.get("business_address", "?")} | {mrec.get("country", "?")}<br>'
    pair_html += '</div>'
    shown += 1

section("10. Real Matching Pairs (What Matches Look Like)", pair_html)

# ── Footer ────────────────────────────────────────────────────────────────────
html_parts.append("</body></html>")

# Write report
full_html = "\n".join(html_parts)
with open(REPORT, "w", encoding="utf-8") as f:
    f.write(full_html)

print(f"\n[OK] Report saved to: {REPORT}")
print(f"     Open in browser to view.")

# ── Console Summary ───────────────────────────────────────────────────────────
print("\n" + "="*60)
print("QUICK SUMMARY")
print("="*60)
print(f"S1 Train: {len(s1_train):>10,} rows")
print(f"S2 Train: {len(s2_train):>10,} rows")
print(f"S3 Train: {len(s3_train):>10,} rows")
print(f"S1 Test:  {len(s1_test):>10,} rows")
print(f"S2 Test:  {len(s2_test):>10,} rows")
print(f"S3 Test:  {len(s3_test):>10,} rows")
print(f"GT rows:  {len(gt_train):>10,} rows")
print(f"\nSingletons: {singletons:,} / {total_s1:,} ({singletons/total_s1*100:.1f}%)")
print(f"With matches: {has_match:,} / {total_s1:,} ({has_match/total_s1*100:.1f}%)")
print(f"Max matches per S1: {gt_train['match_count'].max()}")
print(f"Avg matches per S1: {gt_train['match_count'].mean():.2f}")
print(f"\nTotal matched IDs: {len(all_match_ids):,}")
print(f"  From S2: {len(s2_matches):,}")
print(f"  From S3: {len(s3_matches):,}")
print(f"\nTrain countries: {s1_train['country'].unique().tolist()}")
print(f"Test countries:  {s1_test['country'].unique().tolist()}")
print(f"\nColumns: {s1_train.columns.tolist()}")
