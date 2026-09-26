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
from sklearn.feature_extraction.text import TfidfVectorizer


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
