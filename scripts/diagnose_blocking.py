"""
diagnose_blocking.py — Explain why the blocker misses true matches.

Loads a streaming aligned sample, runs the blocker, and for every missed true
pair reports which blocking keys *should* have connected the pair (and therefore
were dropped by the candidate cap), versus pairs that share no key at all.

Usage:
    python scripts/diagnose_blocking.py [n_s1]
"""
import re
import sys
import collections
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rapidfuzz import fuzz

from src.io import load_aligned_sample
from src.normalize import normalize_all_sources
from src.blocking import (
    generate_all_candidates, evaluate_blocking, soundex, _first_name_token,
)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
MAXC = 100

s1, s2, s3, gt = load_aligned_sample(n_s1=N)
s1n, s2n, s3n = normalize_all_sources(s1, s2, s3)
cands = generate_all_candidates(s1n, s2n, s3n, max_candidates=MAXC, show_progress=False)
metrics = evaluate_blocking(cands, gt)
print("\nBLOCKING METRICS:", {k: metrics[k] for k in (
    'candidate_recall', 'found_true_matches', 'total_true_matches',
    'avg_candidates_per_s1', 'max_candidates_per_s1', 'gate_passed')})

lookup = {}
for df in (s2n, s3n):
    for _, r in df.iterrows():
        lookup[r['entity_id']] = r
s1look = s1n.set_index('entity_id')

reasons = collections.Counter()
examples_share_key = []
examples_none = []
n_missed = 0
for sid, true in gt.items():
    if not true:
        continue
    got = cands.get(sid, set())
    missed = true - got
    if not missed:
        continue
    a = s1look.loc[sid]
    for tid in missed:
        b = lookup.get(tid)
        if b is None:
            continue
        n_missed += 1
        r = []
        if a['country_clean'] != b['country_clean']:
            r.append('country_diff')
        ca = re.sub(r'^the\s+', '', a['name_clean'])
        cb = re.sub(r'^the\s+', '', b['name_clean'])
        if ca and ca == cb:
            r.append('exact_name')
        if len(ca) >= 3 and len(cb) >= 3 and ca[:6] == cb[:6]:
            r.append('prefix6')
        pa = {x for x in str(a['addr_postal']).split(',') if x}
        pb = {x for x in str(b['addr_postal']).split(',') if x}
        if pa & pb:
            r.append('postal')
        ta = _first_name_token(a['name_clean'])
        tb = _first_name_token(b['name_clean'])
        if ta and tb and soundex(ta) == soundex(tb):
            r.append('soundex')
        if len(ca) >= 3 and len(cb) >= 3 and ca[:4] == cb[:4]:
            r.append('relaxed4')
        na = set(str(a['addr_numeric']).split())
        nb = set(str(b['addr_numeric']).split())
        if len(na & nb) >= 2:
            r.append('numerics2')
        shared = set(str(a['name_tokens']).split()) & set(str(b['name_tokens']).split())
        if shared:
            r.append('shared_name_token')
        if len(na & nb) >= 1:
            r.append('numerics1')
        if not r:
            r.append('NONE')
        for reason in r:
            reasons[reason] += 1

        rec = (sid, tid, a['name_clean'][:40], b['name_clean'][:40],
               a['country_clean'], b['country_clean'],
               round(fuzz.ratio(a['name_clean'], b['name_clean'])))
        if 'NONE' in r:
            if len(examples_none) < 20:
                examples_none.append(rec)
        elif len(r) == 1 and r[0] == 'country_diff':
            if len(examples_share_key) < 10:
                examples_share_key.append(rec)

print(f"\nMISSED TRUE PAIRS: {n_missed}")
print("REASONS (a pair can have several):")
for reason, count in reasons.most_common():
    print(f"  {reason:22s} {count}")
print("\nEXAMPLES — no shared key at all:")
for e in examples_none:
    print("  ", e)
print("\nEXAMPLES — only country differs:")
for e in examples_share_key:
    print("  ", e)
