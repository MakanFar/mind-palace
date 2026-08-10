"""Pluggable embedding backends. Local by default; stub in tests."""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

import numpy as np

DTYPE = np.float32


class EmbedderError(RuntimeError):
    """Raised for an unknown or unusable embedder configuration."""


@runtime_checkable
class Embedder(Protocol):
    model_id: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class StubEmbedder:
    """Deterministic hash-derived vectors. Never touches the network."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self.model_id = f"stub-{dim}"

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=DTYPE)
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], "big")
            generator = np.random.default_rng(seed)
            vector = generator.standard_normal(self.dim).astype(DTYPE)
            rows.append(vector / np.linalg.norm(vector))
        return np.vstack(rows)


class LocalEmbedder:
    """On-device embeddings via fastembed. Downloads the model on first use."""

    def __init__(self, model_name: str) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise EmbedderError(
                "fastembed is not installed; run `uv sync` to restore dependencies"
            ) from exc
        self._model = TextEmbedding(model_name=model_name)
        self.model_id = model_name
        probe = np.array(list(self._model.embed(["dimension probe"]))[0])
        self.dim = int(probe.size)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=DTYPE)
        vectors = np.vstack([np.asarray(v, dtype=DTYPE) for v in self._model.embed(texts)])
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return vectors / norms


def get_embedder(spec: dict) -> Embedder:
    kind = spec.get("kind")
    if kind == "stub":
        return StubEmbedder(dim=int(spec.get("dim", 64)))
    if kind == "local":
        return LocalEmbedder(spec["model"])
    if kind == "cloud":
        raise EmbedderError(
            "the cloud embedder is not implemented in v0; every capture and note "
            "body would be sent to a third party, so it needs an explicit build"
        )
    raise EmbedderError(f"unknown embedder kind {kind!r}")
