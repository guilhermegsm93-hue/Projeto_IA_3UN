"""Visualizações a partir da representação intermediária (logs JSON)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.config import ARTIFACTS_DIR, LOGS_FILE, RUNS_DIR


def _ensure_artifacts() -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR


def load_experiment_logs() -> list[dict]:
    if not LOGS_FILE.exists():
        return []
    with LOGS_FILE.open(encoding="utf-8") as f:
        return json.load(f)


def logs_to_dataframe(logs: list[dict] | None = None) -> pd.DataFrame:
    """Converte logs em tabela comparativa."""
    logs = logs or load_experiment_logs()
    rows = []
    for r in logs:
        if "metrics" not in r or r.get("task") != "classification":
            continue
        rows.append({
            "experiment_id": r.get("experiment_id"),
            "dataset": r.get("dataset"),
            "embedding": r.get("embedding", {}).get("name"),
            "algorithm": r.get("algorithm", {}).get("algorithm"),
            "preprocessing": r.get("preprocessing", {}).get("name"),
            "accuracy": r["metrics"].get("accuracy"),
            "f1_weighted": r["metrics"].get("f1_weighted"),
        })
    return pd.DataFrame(rows)


def plot_comparison_bar(output_name: str = "comparison_bar.png") -> Path | None:
    """Gráfico de barras comparando F1 por configuração."""
    df = logs_to_dataframe()
    if df.empty:
        return None
    _ensure_artifacts()
    df["label"] = df["dataset"] + " | " + df["embedding"] + " | " + df["algorithm"]
    plt.figure(figsize=(12, max(4, len(df) * 0.35)))
    sns.barplot(data=df, y="label", x="f1_weighted", hue="dataset", dodge=False)
    plt.xlabel("F1-Score (weighted)")
    plt.ylabel("Configuração experimental")
    plt.title("Comparação de experimentos — Classificação")
    plt.tight_layout()
    path = ARTIFACTS_DIR / output_name
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def plot_heatmap_by_dataset(output_name: str = "heatmap_f1.png") -> Path | None:
    """Heatmap embedding × algoritmo por base."""
    df = logs_to_dataframe()
    if df.empty:
        return None
    _ensure_artifacts()
    fig, axes = plt.subplots(1, len(df["dataset"].unique()), figsize=(6 * len(df["dataset"].unique()), 5))
    if len(df["dataset"].unique()) == 1:
        axes = [axes]
    for ax, (dataset, sub) in zip(axes, df.groupby("dataset")):
        pivot = sub.pivot_table(
            index="embedding", columns="algorithm", values="f1_weighted", aggfunc="max"
        )
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax, robust=True)
        ax.set_title(f"Dataset: {dataset}")
    plt.suptitle("F1-Score: Embedding × Algoritmo")
    plt.tight_layout()
    path = ARTIFACTS_DIR / output_name
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def plot_confusion_from_report(
    experiment_id: str,
    output_name: str | None = None,
) -> Path | None:
    """Gera matriz de confusão a partir do classification_report salvo."""
    logs = load_experiment_logs()
    record = next((r for r in logs if r.get("experiment_id") == experiment_id), None)
    if not record or "metrics" not in record:
        return None
    report = record["metrics"].get("classification_report", {})
    classes = [c for c in report if c not in ("accuracy", "macro avg", "weighted avg")]
    if not classes:
        return None

    # Reconstrói matriz aproximada a partir de precision/recall/support
    n = len(classes)
    cm = np.zeros((n, n))
    for i, cls in enumerate(classes):
        support = report[cls].get("support", 0)
        recall = report[cls].get("recall", 0)
        cm[i, i] = support * recall
        # Erros distribuídos na linha (aproximação visual)
        errors = support - cm[i, i]
        if errors > 0 and n > 1:
            for j in range(n):
                if j != i:
                    cm[i, j] = errors / (n - 1)

    _ensure_artifacts()
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".0f", xticklabels=classes, yticklabels=classes, cmap="Blues")
    plt.xlabel("Previsto")
    plt.ylabel("Verdadeiro")
    plt.title(f"Matriz de confusão — {experiment_id}")
    plt.tight_layout()
    out = output_name or f"confusion_{experiment_id}.png"
    path = ARTIFACTS_DIR / out
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def plot_clustering_silhouette(output_name: str = "clustering_silhouette.png") -> Path | None:
    """Compara silhouette score dos experimentos de clustering."""
    logs = load_experiment_logs()
    rows = [
        {
            "dataset": r["dataset"],
            "embedding": r.get("embedding", {}).get("name"),
            "silhouette": r["metrics"].get("silhouette_score"),
        }
        for r in logs
        if r.get("task") == "clustering" and "metrics" in r
    ]
    if not rows:
        return None
    df = pd.DataFrame(rows)
    _ensure_artifacts()
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="embedding", y="silhouette", hue="dataset")
    plt.title("Silhouette Score — Clustering K-Means")
    plt.tight_layout()
    path = ARTIFACTS_DIR / output_name
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def generate_all_visualizations() -> list[str]:
    """Gera todos os gráficos a partir dos logs existentes."""
    paths = []
    for fn in (
        plot_comparison_bar,
        plot_heatmap_by_dataset,
        plot_clustering_silhouette,
    ):
        result = fn()
        if result:
            paths.append(str(result))

    logs = load_experiment_logs()
    best = max(
        (r for r in logs if r.get("task") == "classification" and "metrics" in r),
        key=lambda r: r["metrics"].get("f1_weighted", 0),
        default=None,
    )
    if best:
        cm_path = plot_confusion_from_report(best["experiment_id"])
        if cm_path:
            paths.append(str(cm_path))

    # Salva tabela comparativa CSV
    df = logs_to_dataframe()
    if not df.empty:
        _ensure_artifacts()
        csv_path = ARTIFACTS_DIR / "comparison_table.csv"
        df.to_csv(csv_path, index=False)
        paths.append(str(csv_path))

    return paths
