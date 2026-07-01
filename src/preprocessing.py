"""Pré-processamento textual configurável."""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def _ensure_nltk_data() -> None:
    for resource in ("stopwords", "wordnet", "omw-1.4"):
        try:
            if resource == "stopwords":
                stopwords.words("english")
            elif resource == "wordnet":
                nltk.data.find("corpora/wordnet")
            else:
                nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download(resource, quiet=True)


@dataclass
class PreprocessingConfig:
    """Configuração de etapas de limpeza textual."""

    lowercase: bool = True
    remove_punctuation: bool = True
    remove_stopwords: bool = True
    lemmatize: bool = False
    min_token_length: int = 2
    language: str = "english"
    name: str = "default"

    @classmethod
    def minimal(cls) -> "PreprocessingConfig":
        return cls(
            lowercase=True,
            remove_punctuation=False,
            remove_stopwords=False,
            lemmatize=False,
            name="minimal",
        )

    @classmethod
    def aggressive(cls) -> "PreprocessingConfig":
        return cls(
            lowercase=True,
            remove_punctuation=True,
            remove_stopwords=True,
            lemmatize=True,
            name="aggressive",
        )


@dataclass
class TextPreprocessor:
    config: PreprocessingConfig = field(default_factory=PreprocessingConfig)

    def __post_init__(self) -> None:
        _ensure_nltk_data()
        self._stopwords = set(stopwords.words(self.config.language))
        self._lemmatizer = WordNetLemmatizer()
        self._punct_table = str.maketrans("", "", string.punctuation)

    def clean(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""
        result = text
        if self.config.lowercase:
            result = result.lower()
        if self.config.remove_punctuation:
            result = result.translate(self._punct_table)
        result = re.sub(r"\s+", " ", result).strip()
        tokens = result.split()
        if self.config.remove_stopwords:
            tokens = [t for t in tokens if t not in self._stopwords]
        if self.config.lemmatize:
            tokens = [self._lemmatizer.lemmatize(t) for t in tokens]
        if self.config.min_token_length > 1:
            tokens = [t for t in tokens if len(t) >= self.config.min_token_length]
        return " ".join(tokens)

    def transform_series(self, texts) -> list[str]:
        return [self.clean(t) for t in texts]


def get_preprocessing_variants() -> list[PreprocessingConfig]:
    """Variantes para análise de sensibilidade do pré-processamento."""
    return [
        PreprocessingConfig(name="default"),
        PreprocessingConfig.minimal(),
        PreprocessingConfig.aggressive(),
    ]
