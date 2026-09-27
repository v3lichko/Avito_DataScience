from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


DEFAULT_MODEL_NAME = "cointegrated/rubert-tiny2"


@dataclass
class TwoTowerConfig:
    model_name: str = DEFAULT_MODEL_NAME
    projection_dim: int = 128
    max_length: int = 64
    temperature: float = 0.05
    n_hard_negatives: int = 4
    batch_size: int = 32
    epochs: int = 3
    lr: float = 2e-5
    device: str = "cpu"


def _require_torch():
    try:
        import torch
        import transformers
    except ImportError as e:
        raise ImportError(
            "Для нейросетевого энкодера нужны torch и transformers: "
            "pip install torch transformers"
        ) from e


class TwoTowerModel:
    def __init__(self, config: TwoTowerConfig):
        _require_torch()
        import torch
        import torch.nn as nn
        from transformers import AutoModel, AutoTokenizer

        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.backbone = AutoModel.from_pretrained(config.model_name)
        hidden_size = self.backbone.config.hidden_size
        self.projection = nn.Linear(hidden_size, config.projection_dim)
        self.device = torch.device(config.device)
        self.backbone.to(self.device)
        self.projection.to(self.device)

    def parameters(self):
        return list(self.backbone.parameters()) + list(self.projection.parameters())

    def train(self):
        self.backbone.train()
        self.projection.train()

    def eval(self):
        self.backbone.eval()
        self.projection.eval()

    def _tokenize(self, texts: Sequence[str]):
        return self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        ).to(self.device)

    def forward(self, texts: Sequence[str]):
        import torch.nn.functional as F

        batch = self._tokenize(texts)
        out = self.backbone(**batch)
        token_emb = out.last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).float()
        summed = (token_emb * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts
        projected = self.projection(pooled)
        return F.normalize(projected, p=2, dim=1)

    def encode(self, texts: Sequence[str], batch_size: int = 128, show_progress: bool = True) -> np.ndarray:
        import torch

        self.eval()
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(list(iterator), desc="Encoding")
            except ImportError:
                pass
        out_chunks = []
        with torch.no_grad():
            for start in iterator:
                chunk = texts[start:start + batch_size]
                emb = self.forward(chunk)
                out_chunks.append(emb.cpu().numpy())
        return np.concatenate(out_chunks, axis=0).astype(np.float32)


def mine_hard_negative_item_indices(
    bm25_index,
    query_tokens: Sequence[Sequence[str]],
    positive_item_indices: Sequence[int],
    n_hard: int,
    pool_size: int = 50,
) -> List[List[int]]:
    hard_negs: List[List[int]] = []
    results = bm25_index.batch_topk(query_tokens, k=pool_size, show_progress=False)
    for (idx, _), pos in zip(results, positive_item_indices):
        candidates = [i for i in idx if i != pos]
        if len(candidates) < n_hard:
            extra_needed = n_hard - len(candidates)
            rng = np.random.default_rng(abs(hash(tuple(idx))) % (2**32))
            random_fill = rng.integers(0, bm25_index.n_docs, size=extra_needed * 3)
            for r in random_fill:
                if r != pos and r not in candidates:
                    candidates.append(int(r))
                if len(candidates) >= n_hard:
                    break
        hard_negs.append(candidates[:n_hard])
    return hard_negs


def train_two_tower(
    model: TwoTowerModel,
    query_texts: Sequence[str],
    item_texts_for_positive: Sequence[str],
    all_item_texts: Sequence[str],
    hard_negative_indices: Sequence[Sequence[int]],
    config: TwoTowerConfig,
) -> TwoTowerModel:
    import torch
    import torch.nn.functional as F
    from torch.optim import AdamW

    n = len(query_texts)
    optimizer = AdamW(model.parameters(), lr=config.lr)
    model.train()

    order = np.arange(n)
    for epoch in range(config.epochs):
        np.random.shuffle(order)
        running_loss, n_batches = 0.0, 0
        for start in range(0, n, config.batch_size):
            batch_idx = order[start:start + config.batch_size]
            bsz = len(batch_idx)
            if bsz < 2:
                continue

            q_texts = [query_texts[i] for i in batch_idx]
            pos_texts = [item_texts_for_positive[i] for i in batch_idx]
            hard_texts: List[str] = []
            for i in batch_idx:
                for hn_idx in hard_negative_indices[i]:
                    hard_texts.append(all_item_texts[hn_idx])

            q_emb = model.forward(q_texts)
            pos_emb = model.forward(pos_texts)
            candidates = pos_emb
            if hard_texts:
                hard_emb = model.forward(hard_texts)
                candidates = torch.cat([pos_emb, hard_emb], dim=0)

            logits = q_emb @ candidates.T / config.temperature
            labels = torch.arange(bsz, device=model.device)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        avg_loss = running_loss / max(n_batches, 1)
        print(f"[epoch {epoch + 1}/{config.epochs}] средний loss = {avg_loss:.4f}")

    return model