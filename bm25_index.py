from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer


@dataclass
class BM25Index:
    k1: float = 1.5
    b: float = 0.75

    def fit(self, tokenized_docs: Sequence[Sequence[str]]) -> "BM25Index":
        self.vectorizer = CountVectorizer(
            analyzer=lambda tokens: tokens,
            lowercase=False,
        )
        X = self.vectorizer.fit_transform(tokenized_docs).tocsr()
        n_docs, n_terms = X.shape

        doc_len = np.asarray(X.sum(axis=1)).ravel()
        avgdl = doc_len.mean() if n_docs > 0 else 1.0
        avgdl = avgdl if avgdl > 0 else 1.0

        df = np.diff((X > 0).tocsc().indptr).astype(np.float64)
        idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
        idf = np.maximum(idf, 0.0)

        row_lengths = np.diff(X.indptr)
        row_idx = np.repeat(np.arange(n_docs), row_lengths)
        denom_extra = self.k1 * (1 - self.b + self.b * doc_len[row_idx] / avgdl)
        tf = X.data.astype(np.float64)
        tf_sat = tf * (self.k1 + 1) / (tf + denom_extra)

        W = csr_matrix((tf_sat, X.indices, X.indptr), shape=X.shape)
        W = W.multiply(idf.reshape(1, -1)).tocsr()

        self.W = W
        self.n_docs = n_docs
        self.idf_ = idf
        return self

    def _query_vector(self, tokens: Sequence[str]) -> csr_matrix:
        vec = self.vectorizer.transform([tokens])
        vec.data[:] = 1.0
        return vec.tocsr()

    def score(self, tokens: Sequence[str]) -> np.ndarray:
        if len(tokens) == 0:
            return np.zeros(self.n_docs, dtype=np.float64)
        q_vec = self._query_vector(tokens)
        scores = self.W.dot(q_vec.T)
        return np.asarray(scores.todense()).ravel()

    def topk(self, tokens: Sequence[str], k: int) -> Tuple[np.ndarray, np.ndarray]:
        scores = self.score(tokens)
        k = min(k, len(scores))
        if k == 0:
            return np.array([], dtype=int), np.array([], dtype=float)
        part = np.argpartition(-scores, k - 1)[:k]
        order = part[np.argsort(-scores[part])]
        return order, scores[order]

    def batch_topk(
        self, list_of_tokens: Sequence[Sequence[str]], k: int, show_progress: bool = True
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        iterator = list_of_tokens
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(list_of_tokens, desc="BM25 retrieval")
            except ImportError:
                pass
        return [self.topk(tokens, k) for tokens in iterator]