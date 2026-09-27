from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from text_utils import build_item_text, build_query_text


def make_local_split(
    train_df: pd.DataFrame,
    n_eval_queries: int = 3000,
    corpus_size: int = 150_000,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, List[str], List[str], List[int]]:
    rng = np.random.default_rng(random_state)
    n = len(train_df)
    n_eval_queries = min(n_eval_queries, n // 2)

    all_idx = rng.permutation(n)
    eval_idx = all_idx[:n_eval_queries]
    pool_idx = all_idx[n_eval_queries:]

    corpus_size = min(corpus_size, len(pool_idx) + n_eval_queries)
    n_extra = max(corpus_size - n_eval_queries, 0)
    extra_idx = pool_idx[:n_extra]

    corpus_idx = np.concatenate([eval_idx, extra_idx])
    corpus_idx = rng.permutation(corpus_idx)

    eval_queries_df = train_df.iloc[eval_idx].reset_index(drop=True)
    corpus_df = train_df.iloc[corpus_idx].reset_index(drop=True)

    corpus_item_texts = [build_item_text(r) for _, r in corpus_df.iterrows()]
    eval_query_texts = [build_query_text(r) for _, r in eval_queries_df.iterrows()]

    pos_in_corpus = {orig_idx: pos for pos, orig_idx in enumerate(corpus_idx)}
    true_item_positions = [pos_in_corpus[orig_idx] for orig_idx in eval_idx]

    return eval_queries_df, corpus_item_texts, eval_query_texts, true_item_positions


def recall_at_k(
    ranked_candidate_lists: Sequence[Sequence[int]],
    true_item_positions: Sequence[int],
) -> float:
    hits = 0
    for cand, true_pos in zip(ranked_candidate_lists, true_item_positions):
        if true_pos in set(cand):
            hits += 1
    return hits / max(len(true_item_positions), 1)


def build_recall_report(name: str, value: float) -> str:
    return f"[{name}] локальный Recall@50 = {value:.4f}"