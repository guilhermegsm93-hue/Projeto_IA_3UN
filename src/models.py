"""Algoritmos de classificação e clustering."""

from __future__ import annotations

from typing import Any

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC


def get_classifier(name: str, random_state: int = 42) -> Any:
    """Retorna classificador sklearn pelo nome."""
    classifiers = {
        "naive_bayes": MultinomialNB(),
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=random_state
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=random_state, n_jobs=-1
        ),
        "svm": LinearSVC(random_state=random_state, max_iter=2000),
    }
    if name not in classifiers:
        raise ValueError(f"Classificador desconhecido: {name}. Opções: {list(classifiers)}")
    return classifiers[name]


def get_classifier_params(name: str) -> dict[str, Any]:
    params = {
        "naive_bayes": {"alpha": 1.0},
        "logistic_regression": {"max_iter": 1000, "solver": "lbfgs"},
        "random_forest": {"n_estimators": 100},
        "svm": {"max_iter": 2000},
    }
    return {"algorithm": name, **params.get(name, {})}


def list_classifiers() -> list[str]:
    return ["naive_bayes", "logistic_regression", "random_forest"]


def build_scaled_pipeline(classifier: Any) -> Pipeline:
    """Pipeline com normalização para embeddings densos."""
    return Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", classifier),
    ])


def get_clusterer(n_clusters: int, random_state: int = 42) -> KMeans:
    return KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)


def encode_labels(y) -> tuple[Any, LabelEncoder]:
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(y)
    return encoded, encoder
