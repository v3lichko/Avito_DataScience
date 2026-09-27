import re
from typing import Dict, List, Sequence

import pandas as pd

ITEM_ID_RE = re.compile(r"^[0-9a-f]{16}$")
QUERY_ID_LEN = 16


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[int]],
    rrf_k: int = 60,
) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, item_idx in enumerate(ranked):
            scores[item_idx] = scores.get(item_idx, 0.0) + 1.0 / (rrf_k + rank + 1)
    return scores


def fuse_candidates_for_query(
    ranked_lists: Sequence[Sequence[int]],
    top_n: int = 50,
    rrf_k: int = 60,
) -> List[int]:
    scores = reciprocal_rank_fusion(ranked_lists, rrf_k=rrf_k)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [idx for idx, _ in ordered[:top_n]]


def build_answer_dataframe(
    query_ids: Sequence[str],
    fused_item_indices_per_query: Sequence[Sequence[int]],
    item_ids: Sequence[str],
) -> pd.DataFrame:
    answers = []
    for indices in fused_item_indices_per_query:
        ids = [item_ids[i] for i in indices]
        answers.append(" ".join(ids))
    return pd.DataFrame({"query_id": list(query_ids), "answer": answers})


def validate_answer_df(
    answer_df: pd.DataFrame,
    expected_query_ids: Sequence[str],
    valid_item_ids: Sequence[str],
    max_items_per_row: int = 50,
) -> List[str]:
    problems: List[str] = []
    expected_set = set(expected_query_ids)
    got_set = set(answer_df["query_id"])

    missing = expected_set - got_set
    extra = got_set - expected_set
    dup_counts = answer_df["query_id"].value_counts()
    dups = dup_counts[dup_counts > 1]
    bad_qid_len = [q for q in answer_df["query_id"].unique() if len(str(q)) != QUERY_ID_LEN]
    valid_item_set = set(valid_item_ids)
    bad_rows = 0
    for _, row in answer_df.iterrows():
        items = str(row["answer"]).split() if pd.notna(row["answer"]) and row["answer"] != "" else []
        if len(items) > max_items_per_row:
            problems.append(f"query_id={row['query_id']}: больше {max_items_per_row} item_id ({len(items)})")
            bad_rows += 1
        if len(set(items)) != len(items):
            problems.append(f"query_id={row['query_id']}: повторяющиеся item_id в строке")
            bad_rows += 1
        bad_ids = [it for it in items if not ITEM_ID_RE.match(it) or it not in valid_item_set]
        if bad_ids:
            problems.append(
                f"query_id={row['query_id']}: item_id не найдены в корпусе или неверного формата: {bad_ids[:3]}"
            )
            bad_rows += 1
            break

    return problems


def write_answer_csv(answer_df: pd.DataFrame, path: str) -> None:
    answer_df.to_csv(path, index=False, encoding="utf-8")