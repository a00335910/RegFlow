"""Local sentence-transformers embedder (BGE-M3 by default).

BGE-M3 is multilingual and required for cross-jurisdiction work (EU FR/DE, UK EN, US EN).
Loaded lazily — first call triggers HuggingFace download (~2GB for bge-m3).

Once the model is cached locally, we run in offline mode — sentence-transformers
otherwise tries to HEAD https://huggingface.co/... on every cold start to check for
model updates, which fails (and crashes the embedder) if DNS or HF is unreachable.
Set HF_HUB_OFFLINE=0 explicitly to force a freshness check when a model changes.
"""

from __future__ import annotations

import os

# Default to offline mode so the embedder never blocks on HuggingFace connectivity.
# `setdefault` means an explicit HF_HUB_OFFLINE=0 in the environment still wins.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from regflow.common.logging import get_logger
from regflow.common.settings import get_settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

log = get_logger(__name__)


class Embedder:
    # Cap input length so BGE-M3 never has to chunk internally — chunked-and-pooled
    # embedding on CPU is the #1 reason a single long article can take minutes.
    # 6000 chars ~= 1500 tokens, well under BGE-M3's 8192-token context.
    _MAX_INPUT_CHARS = 6000

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        s = get_settings().embeddings
        log.info("embedder.loading", model=s.model_name, device=s.device)
        self._model: SentenceTransformer = SentenceTransformer(s.model_name, device=s.device)
        self._batch_size = s.batch_size
        self._normalize = s.normalize
        self.dimension = s.dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        truncated = [t[: self._MAX_INPUT_CHARS] for t in texts]
        vectors: np.ndarray = self._model.encode(
            truncated,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder()
