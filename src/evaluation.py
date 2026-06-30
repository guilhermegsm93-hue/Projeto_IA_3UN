"""Motor de experimentos e representação intermediária (logs JSON)."""

from __future__ import annotations

import json
import pickle
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split

from src import __version__
from src.config import (
    ARTIFACTS_DIR,
    LOGS_FILE,
    MODELS_DIR,
    RANDOM_STATE,
    RUNS_DIR,
    TEST_SIZE,
)
from src.embeddings import BaseEmbedder, get_embedder
from src.ingestion import load_dataset
from src.models import (
    build_scaled_pipeline,
    encode_labels,
    get_classifier,
    get_classifier_params,
    get_clusterer,
)
from src.preprocessing import PreprocessingConfig, TextPreprocessor


def _ensure_dirs() -> None:
    for d in (RUNS_DIR, ARTIFACTS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _load_logs() -> list[dict]:
    if LOGS_FILE.exists():
        with LOGS_FILE.open(encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_logs(logs: list[dict]) -> None:
    _ensure_dirs()
    with LOGS_FILE.open("w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)


def _needs_scaling(embedder_name: str) -> bool:
    return embedder_name == "sentence_transformer"


def _collect_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    texts: list[str],
    label_encoder,
    max_errors: int = 20,
) -> list[dict]:
    errors = []
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        if true != pred:
            errors.append({
                "text": texts[i],
                "true_label": label_encoder.inverse_transform([true])[0],
                "predicted_label": label_encoder.inverse_transform([pred])[0],
            })
        if len(errors) >= max_errors:
            break
    return errors


def run_classification_experiment(
    dataset_name: str,
    embedder_name: str,
    classifier_name: str,
    preprocessing: PreprocessingConfig | None = None,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Executa um experimento de classificação e retorna registro estruturado."""
    _ensure_dirs()
    preprocessing = preprocessing or PreprocessingConfig()
    experiment_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now(timezone.utc).isoformat()

    df = load_dataset(dataset_name)
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=RANDOM_STATE).reset_index(drop=True)

    preprocessor = TextPreprocessor(preprocessing)
    texts_clean = preprocessor.transform_series(df["texto_bruto"].tolist())
    y_raw = df["alvo"].values
    y, label_encoder = encode_labels(y_raw)

    split_kwargs: dict = {
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
    }
    # stratify falha quando alguma classe tem poucas amostras
    _, counts = np.unique(y, return_counts=True)
    if counts.min() >= 2:
        split_kwargs["stratify"] = y

    X_train_text, X_test_text, y_train, y_test, raw_train, raw_test = train_test_split(
        texts_clean,
        y,
        df["texto_bruto"].tolist(),
        **split_kwargs,
    )

    embedder: BaseEmbedder = get_embedder(embedder_name)
    X_train = embedder.fit_transform(X_train_text)
    X_test = embedder.transform(X_test_text)

    classifier = get_classifier(classifier_name)
    if _needs_scaling(embedder_name):
        if classifier_name == "naive_bayes":
            # MultinomialNB exige features não-negativas; embeddings densos usam LR.
            classifier = get_classifier("logistic_regression")
            classifier_name = "logistic_regression"
        pipeline = build_scaled_pipeline(classifier)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        model_obj = pipeline
    else:
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        model_obj = classifier

    accuracy = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="weighted"))
    report = classification_report(
        y_test, y_pred, target_names=list(label_encoder.classes_), output_dict=True
    )
    errors = _collect_errors(y_test, y_pred, raw_test, label_encoder)

    model_path = MODELS_DIR / f"{experiment_id}.pkl"
    artifact = {
        "model": model_obj,
        "embedder_name": embedder_name,
        "embedder": embedder if embedder_name == "tfidf" else None,
        "embedder_model_name": getattr(embedder, "model_name", None),
        "label_encoder": label_encoder,
        "preprocessing": preprocessing,
        "classifier_name": classifier_name,
        "dataset_name": dataset_name,
    }
    with model_path.open("wb") as f:
        pickle.dump(artifact, f)

    record = {
        "experiment_id": experiment_id,
        "timestamp": timestamp,
        "framework_version": __version__,
        "task": "classification",
        "dataset": dataset_name,
        "n_samples": len(df),
        "n_classes": len(label_encoder.classes_),
        "classes": list(label_encoder.classes_),
        "preprocessing": {
            "name": preprocessing.name,
            "lowercase": preprocessing.lowercase,
            "remove_punctuation": preprocessing.remove_punctuation,
            "remove_stopwords": preprocessing.remove_stopwords,
            "lemmatize": preprocessing.lemmatize,
        },
        "embedding": embedder.get_params(),
        "algorithm": get_classifier_params(classifier_name),
        "metrics": {
            "accuracy": accuracy,
            "f1_weighted": f1,
            "classification_report": report,
        },
        "errors_sample": errors,
        "artifacts": {
            "model_path": str(model_path.relative_to(RUNS_DIR.parent)),
            "confusion_matrix_key": f"{experiment_id}_cm",
        },
    }
    return record


def run_clustering_experiment(
    dataset_name: str,
    embedder_name: str,
    preprocessing: PreprocessingConfig | None = None,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Executa experimento de clustering (K-Means) com silhouette score."""
    _ensure_dirs()
    preprocessing = preprocessing or PreprocessingConfig()
    experiment_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now(timezone.utc).isoformat()

    df = load_dataset(dataset_name)
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=RANDOM_STATE).reset_index(drop=True)

    preprocessor = TextPreprocessor(preprocessing)
    texts_clean = preprocessor.transform_series(df["texto_bruto"].tolist())
    y_raw = df["alvo"].values
    _, label_encoder = encode_labels(y_raw)
    n_clusters = len(label_encoder.classes_)

    embedder = get_embedder(embedder_name)
    X = embedder.fit_transform(texts_clean)

    clusterer = get_clusterer(n_clusters=n_clusters)
    labels = clusterer.fit_predict(X)

    sil = float(silhouette_score(X, labels)) if n_clusters > 1 and len(set(labels)) > 1 else 0.0

    record = {
        "experiment_id": experiment_id,
        "timestamp": timestamp,
        "framework_version": __version__,
        "task": "clustering",
        "dataset": dataset_name,
        "n_samples": len(df),
        "n_clusters": n_clusters,
        "preprocessing": {"name": preprocessing.name},
        "embedding": embedder.get_params(),
        "algorithm": {"name": "kmeans", "n_clusters": n_clusters},
        "metrics": {"silhouette_score": sil},
        "cluster_preview": _cluster_preview(texts_clean, labels, label_encoder, n=3),
    }
    return record


def _cluster_preview(
    texts: list[str],
    labels: np.ndarray,
    label_encoder,
    n: int = 3,
) -> dict[str, list[str]]:
    preview: dict[str, list[str]] = {}
    for cluster_id in sorted(set(labels)):
        indices = np.where(labels == cluster_id)[0][:n]
        preview[f"cluster_{cluster_id}"] = [texts[i][:100] for i in indices]
    return preview


def run_benchmark(
    datasets: list[str] | None = None,
    embedders: list[str] | None = None,
    classifiers: list[str] | None = None,
    include_clustering: bool = True,
    preprocessing_variants: list[PreprocessingConfig] | None = None,
    max_samples: int | None = 600,
) -> list[dict]:
    """Orquestrador: cruza bases × embeddings × algoritmos e persiste logs."""
    from src.models import list_classifiers
    from src.ingestion import list_available_datasets

    datasets = datasets or list(list_available_datasets().keys())
    embedders = embedders or ["tfidf", "sentence_transformer"]
    classifiers = classifiers or list_classifiers()
    preprocessing_variants = preprocessing_variants or [PreprocessingConfig()]

    logs = _load_logs()
    new_records: list[dict] = []

    for dataset in datasets:
        for prep in preprocessing_variants:
            for embedder in embedders:
                for clf in classifiers:
                    try:
                        record = run_classification_experiment(
                            dataset_name=dataset,
                            embedder_name=embedder,
                            classifier_name=clf,
                            preprocessing=prep,
                            max_samples=max_samples,
                        )
                        new_records.append(record)
                        logs.append(record)
                    except Exception as exc:
                        err_record = {
                            "task": "classification",
                            "dataset": dataset,
                            "embedding": embedder,
                            "algorithm": clf,
                            "error": str(exc),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                        logs.append(err_record)
                        new_records.append(err_record)

                if include_clustering:
                    try:
                        record = run_clustering_experiment(
                            dataset_name=dataset,
                            embedder_name=embedder,
                            preprocessing=prep,
                            max_samples=max_samples,
                        )
                        new_records.append(record)
                        logs.append(record)
                    except Exception as exc:
                        logs.append({
                            "task": "clustering",
                            "dataset": dataset,
                            "embedding": embedder,
                            "error": str(exc),
                        })

    _save_logs(logs)
    return new_records


def get_best_experiment(task: str = "classification", dataset: str | None = None) -> dict | None:
    """Retorna experimento com melhor métrica principal, opcionalmente filtrado por base."""
    logs = _load_logs()
    candidates = [r for r in logs if r.get("task") == task and "metrics" in r]
    if dataset:
        candidates = [r for r in candidates if r.get("dataset") == dataset]
    if not candidates:
        return None
    if task == "classification":
        return max(candidates, key=lambda r: r["metrics"].get("f1_weighted", 0))
    return max(candidates, key=lambda r: r["metrics"].get("silhouette_score", 0))


def load_champion_model(dataset: str | None = None) -> dict | None:
    """Carrega artefato do melhor experimento de classificação."""
    best = get_best_experiment("classification", dataset=dataset)
    if not best or "artifacts" not in best:
        # fallback: qualquer campeão disponível
        best = get_best_experiment("classification")
    if not best or "artifacts" not in best:
        return None
    model_path = RUNS_DIR.parent / best["artifacts"]["model_path"]
    if not model_path.exists():
        return None
    with model_path.open("rb") as f:
        artifact = pickle.load(f)
    artifact["_experiment"] = best
    return artifact


def predict_text(text: str, artifact: dict | None = None, dataset: str | None = None) -> dict[str, Any]:
    """Prediz classe para texto usando modelo campeão."""
    artifact = artifact or load_champion_model(dataset=dataset)
    if artifact is None:
        raise RuntimeError("Nenhum modelo treinado. Execute run_benchmark() primeiro.")

    preprocessor = TextPreprocessor(artifact["preprocessing"])
    clean = preprocessor.clean(text)
    embedder_name = artifact["embedder_name"]

    if embedder_name == "tfidf":
        embedder = artifact["embedder"]
        if embedder is None:
            embedder = get_embedder("tfidf")
        vector = embedder.transform([clean])
    else:
        from sentence_transformers import SentenceTransformer

        model_name = artifact.get("embedder_model_name") or "paraphrase-MiniLM-L6-v2"
        model = SentenceTransformer(model_name)
        vector = model.encode([clean], convert_to_numpy=True)

    model = artifact["model"]
    pred_idx = model.predict(vector)[0]
    label = artifact["label_encoder"].inverse_transform([pred_idx])[0]
    return {"text": text, "prediction": label, "preprocessed": clean}
