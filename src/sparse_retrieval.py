"""
sparse_retrieval.py — Sparse TF-IDF top-N candidate retrieval.

Represents each business name as a sparse TF-IDF vector over character n-grams
(character n-grams are typo/word-order robust). Candidates for an S1 record are
retrieved via an inverted postings index (features shared with the query), then
ranked by sparse dot product (cosine), returning the top-N targets.

This is the "sparse-dot-topN" retriever: recall comes from the inverted index,
precision/ordering from the dot product.
"""
from typing import List, Optional, Tuple

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


class SparseIndex:
    """TF-IDF character-n-gram index with inverted postings + dot-product top-N."""

    def __init__(self, vectorizer: TfidfVectorizer, X: sparse.csr_matrix,
                 postings: List[np.ndarray], ids: List[str]):
        self.vectorizer = vectorizer
        self.X = X                      # CSR (n_docs, V)
        self.postings = postings        # feature_id -> np.array(doc rows)
        self.ids = ids                  # row -> entity_id

    # ── Build ─────────────────────────────────────────────────────────────────
    @classmethod
    def build(cls, texts: List[str], ids: List[str],
              ngram_range: Tuple[int, int] = (3, 5),
              min_df: int = 2, max_features: int = 500_000) -> "SparseIndex":
        vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=ngram_range,
            min_df=min_df, max_features=max_features, dtype=np.float32)
        X = vec.fit_transform(texts).tocsr()
        X.sort_indices()
        Xc = X.tocsc()
        postings = [Xc.indices[Xc.indptr[j]:Xc.indptr[j + 1]]
                    for j in range(Xc.shape[1])]
        return cls(vec, X, postings, ids)

    # ── Query ─────────────────────────────────────────────────────────────────
    def query_topn(self, text: str, top_n: int,
                   max_post: int = 2000) -> Tuple[List[str], np.ndarray]:
        """
        Return the top-N target ids by cosine (dot) score for one query text.

        `max_post` bounds the number of postings pulled per query feature, so a
        single very common n-gram cannot blow up the candidate union.
        """
        q = self.vectorizer.transform([text]).tocsr()
        if q.nnz == 0:
            return [], np.empty(0, dtype=np.float32)

        feat_ids = q.indices
        buckets = [self.postings[f][:max_post]
                   for f in feat_ids if len(self.postings[f])]
        if not buckets:
            return [], np.empty(0, dtype=np.float32)
        cand = np.unique(np.concatenate(buckets))
        scores = np.asarray(self.X[cand].dot(q.T).todense()).ravel()

        k = min(top_n, len(cand))
        part = np.argpartition(-scores, k - 1)[:k]
        part = part[np.argsort(-scores[part])]
        rows = cand[part]
        return [self.ids[r] for r in rows], scores[part]


class BM25Index:
    """
    Okapi BM25 top-N retrieval over word n-grams (typo / token-recall signal).

    Unlike the TF-IDF cosine `SparseIndex` above, BM25 saturates term frequency
    and normalises by document length, so a single distinctive name token can
    pull a candidate that prefix/soundex blocks miss. Built with a count
    vectorizer + a compact CSC postings list, so no extra dependency is needed.

    Typical use is as an extra *high-recall* blocking pass unioned with the
    inverted-index blocks, or as a candidate top-up when a query is under-filled.
    """

    def __init__(self, vectorizer, X: sparse.csr_matrix,
                 postings: List[np.ndarray], ids: List[str],
                 idf: np.ndarray, doc_len: np.ndarray, avgdl: float,
                 k1: float = 1.5, b: float = 0.75):
        self.vectorizer = vectorizer
        self.X = X                      # CSR (n_docs, V) raw term counts
        self.postings = postings        # feature_id -> np.array(doc rows)
        self.ids = ids                  # row -> entity_id
        self.idf = idf                  # (V,)
        self.doc_len = doc_len          # (n_docs,)
        self.avgdl = float(avgdl) if avgdl > 0 else 1.0
        self.k1 = float(k1)
        self.b = float(b)

    # ── Build ─────────────────────────────────────────────────────────────────
    @classmethod
    def build(cls, texts: List[str], ids: List[str],
              ngram_range: Tuple[int, int] = (1, 2),
              min_df: int = 2, max_features: int = 500_000,
              k1: float = 1.5, b: float = 0.75) -> "BM25Index":
        vec = CountVectorizer(
            analyzer="word", token_pattern=r"(?u)\b\w+\b",
            ngram_range=ngram_range, min_df=min_df, max_features=max_features,
            dtype=np.float32)
        X = vec.fit_transform(texts).tocsr()
        X.sort_indices()

        n_docs = X.shape[0]
        df = np.asarray((X > 0).sum(axis=0)).ravel().astype(np.float64)
        idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

        doc_len = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
        avgdl = float(doc_len.mean()) if n_docs else 0.0

        Xc = X.tocsc()
        postings = [Xc.indices[Xc.indptr[j]:Xc.indptr[j + 1]]
                    for j in range(Xc.shape[1])]
        return cls(vec, X, postings, ids, idf, doc_len, avgdl, k1=k1, b=b)

    # ── Query ─────────────────────────────────────────────────────────────────
    def query_topn(self, text: str, top_n: int,
                   max_post: int = 2000) -> Tuple[List[str], np.ndarray]:
        """Return the top-N target ids by BM25 score for one query string."""
        if not text or not str(text).strip() or self.X.shape[0] == 0:
            return [], np.empty(0, dtype=np.float32)

        q = self.vectorizer.transform([text]).tocsr()
        if q.nnz == 0:
            return [], np.empty(0, dtype=np.float32)

        feat_ids = q.indices
        buckets = [self.postings[f][:max_post]
                   for f in feat_ids if len(self.postings[f])]
        if not buckets:
            return [], np.empty(0, dtype=np.float32)
        cand = np.unique(np.concatenate(buckets))

        # Score only the candidate rows, and only over the query features.
        sub = self.X[cand][:, feat_ids].tocsr()
        norm = self.k1 * (1.0 - self.b + self.b * self.doc_len[cand] / self.avgdl)
        coef = self.idf[feat_ids] * (self.k1 + 1.0)

        rows = np.repeat(np.arange(sub.shape[0]), np.diff(sub.indptr))
        contrib = sub.data * coef[sub.indices] / (sub.data + norm[rows])
        scores = np.bincount(rows, weights=contrib,
                             minlength=sub.shape[0]).astype(np.float64)

        k = min(top_n, len(cand))
        part = np.argpartition(-scores, k - 1)[:k]
        part = part[np.argsort(-scores[part], kind="stable")]
        rows = cand[part]
        return [self.ids[r] for r in rows], scores[part].astype(np.float32)

