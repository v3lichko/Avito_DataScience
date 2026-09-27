from typing import List, Tuple

import numpy as np


def brute_force_topk(
    query_emb: np.ndarray,
    item_emb: np.ndarray,
    k: int,
    query_chunk_size: int = 256,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    n_queries = query_emb.shape[0]
    n_items = item_emb.shape[0]
    k = min(k, n_items)
    results: List[Tuple[np.ndarray, np.ndarray]] = []

    item_emb_t = item_emb.T.astype(np.float32)

    for start in range(0, n_queries, query_chunk_size):
        chunk = query_emb[start:start + query_chunk_size].astype(np.float32)
        sims = chunk @ item_emb_t
        part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        row_idx = np.arange(sims.shape[0])[:, None]
        part_scores = sims[row_idx, part]
        order_within = np.argsort(-part_scores, axis=1)
        sorted_idx = part[row_idx, order_within]
        sorted_scores = part_scores[row_idx, order_within]
        for i in range(chunk.shape[0]):
            results.append((sorted_idx[i], sorted_scores[i]))
    return results


def faiss_topk(
    query_emb: np.ndarray,
    item_emb: np.ndarray,
    k: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    import faiss
    d = item_emb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(np.ascontiguousarray(item_emb.astype(np.float32)))
    scores, idx = index.search(np.ascontiguousarray(query_emb.astype(np.float32)), k)
    return [(idx[i], scores[i]) for i in range(query_emb.shape[0])]


def search_topk(
    query_emb: np.ndarray,
    item_emb: np.ndarray,
    k: int,
    use_faiss: bool = True,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    return faiss_topk(query_emb, item_emb, k)