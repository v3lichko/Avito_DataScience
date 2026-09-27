import re
from typing import Iterable, List, Optional

import pandas as pd

RU_STOPWORDS = {
    "и", "а", "но", "да", "или", "либо", "то", "не", "ни", "же", "ли",
    "бы", "б", "ж", "уж", "вот", "будто", "чтоб", "чтобы", "как будто",
    "в", "во", "на", "с", "со", "к", "ко", "у", "о", "об", "обо", "от",
    "ото", "до", "из", "изо", "из-за", "из-под", "за", "над", "надо",
    "под", "подо", "по", "при", "про", "через", "без", "для", "перед",
    "передо", "между",
    "я", "ты", "он", "она", "оно", "они", "мы", "вы", "себя", "себе",
    "мой", "моя", "моё", "мои", "твой", "твоя", "твоё", "твои", "его",
    "её", "ее", "их", "наш", "наша", "наше", "наши", "ваш", "ваша",
    "ваше", "ваши", "свой", "своя", "своё", "свою", "этот", "эта",
    "это", "эти", "этого", "этой", "этом", "тот", "та", "те", "того",
    "том", "такой", "такая", "такое", "такие", "кто", "что", "какой",
    "какая", "какое", "какие", "который", "которая", "которое",
    "которые", "весь", "вся", "всё", "все", "всего", "всех", "сам",
    "сама", "само", "сами", "меня", "мне", "тебя", "тебе", "него",
    "нему", "нее", "ней", "них", "ним", "нём", "нам", "вам", "вас",
    "нас", "им",
    "тут", "там", "тогда", "теперь", "сейчас", "потом", "здесь",
    "везде", "куда", "откуда", "зачем", "почему", "как", "когда",
    "где", "уже", "еще", "ещё", "снова", "опять", "вдруг", "сразу",
    "почти", "совсем", "чуть", "больше", "меньше", "более", "менее",
    "лучше", "хуже", "очень", "слишком", "всегда", "никогда",
    "иногда", "разве", "неужели", "конечно", "наверное", "может",
    "можно", "нельзя", "надо", "нужно", "хоть", "хотя", "впрочем",
    "наконец", "потому", "поэтому", "затем",
    "быть", "был", "была", "было", "были", "будет", "будут", "есть",
    "один", "одна", "одно", "два", "две", "три", "нет",
    "чего", "чем", "чему", "раз", "разом", "также", "тоже", "ничего",
    "нибудь", "либо-нибудь", "кое-что", "пожалуйста",
}

_TOKEN_RE = re.compile(r"[0-9a-zA-Zа-яёА-ЯЁ]+", re.UNICODE)


def normalize_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return text.lower()


def tokenize(text: Optional[str], drop_stopwords: bool = True,
             min_len: int = 2) -> List[str]:
    norm = normalize_text(text)
    tokens = _TOKEN_RE.findall(norm)
    if min_len > 1:
        tokens = [t for t in tokens if len(t) >= min_len]
    if drop_stopwords:
        tokens = [t for t in tokens if t not in RU_STOPWORDS]
    return tokens


def _safe_str(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def build_query_text(row: pd.Series) -> str:
    parts = [
        _safe_str(row.get("search_query")),
        _safe_str(row.get("search_infm_params_text")),
    ]
    category = _safe_str(row.get("search_category"))
    if category:
        parts.append(f"категория {category}")
    return " [SEP] ".join(p for p in parts if p)


def build_item_text(row: pd.Series) -> str:
    parts = [
        _safe_str(row.get("item_title_raw")),
        _safe_str(row.get("item_description_raw")),
        _safe_str(row.get("item_infm_params_text")),
    ]
    cat = _safe_str(row.get("item_category_id"))
    microcat = _safe_str(row.get("item_microcat_id"))
    if cat:
        parts.append(f"категория {cat}")
    if microcat:
        parts.append(f"подкатегория {microcat}")
    return " [SEP] ".join(p for p in parts if p)


def build_query_texts(df: pd.DataFrame) -> List[str]:
    return [build_query_text(row) for _, row in df.iterrows()]


def build_item_texts(df: pd.DataFrame) -> List[str]:
    return [build_item_text(row) for _, row in df.iterrows()]


def tokenize_all(texts: Iterable[str], **kwargs) -> List[List[str]]:
    return [tokenize(t, **kwargs) for t in texts]