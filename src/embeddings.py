"""Extração de representações textuais — TF-IDF e Sentence Transformers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class BaseEmbedder(ABC):
    name: str

    @abstractmethod
    def fit_transform(self, texts: list[str]) -> np.ndarray:
        ...

    @abstractmethod
    def transform(self, texts: list[str]) -> np.ndarray:
        ...

    def get_params(self) -> dict[str, Any]:
        return {"name": self.name}


class TfidfEmbedder(BaseEmbedder):
    def __init__(self, max_features: int = 5000, ngram_range: tuple[int, int] = (1, 2)):
        self.name = "tfidf"
        self.max_features = max_features
        self.ngram_range = ngram_range
        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
        )

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self._vectorizer.fit_transform(texts).toarray()

    def transform(self, texts: list[str]) -> np.ndarray:
        return self._vectorizer.transform(texts).toarray()

    def get_params(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "max_features": self.max_features,
            "ngram_range": self.ngram_range,
        }


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "paraphrase-MiniLM-L6-v2"):
        self.name = "sentence_transformer"
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        model = self._load_model()
        return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    def transform(self, texts: list[str]) -> np.ndarray:
        return self.fit_transform(texts)

    def get_params(self) -> dict[str, Any]:
        return {"name": self.name, "model_name": self.model_name}


def get_embedder(name: str, **kwargs) -> BaseEmbedder:
    embedders = {
        "tfidf": TfidfEmbedder,
        "sentence_transformer": SentenceTransformerEmbedder,
    }
    if name not in embedders:
        raise ValueError(f"Embedder desconhecido: {name}")
    return embedders[name](**kwargs)


def list_embedders() -> list[str]:
    return ["tfidf", "sentence_transformer"]
