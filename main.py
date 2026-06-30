"""
Ponto de entrada do framework experimental de NLP — Tema 10.

Executa benchmark completo: 2 bases × 2 embeddings × 3 classificadores + clustering,
persiste logs JSON e gera visualizações.
"""

from __future__ import annotations

import argparse
import json

from src.config import LOGS_FILE
from src.evaluation import get_best_experiment, run_benchmark
from src.llm_service import explain_errors_batch
from src.visualization import generate_all_visualizations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Framework experimental de NLP — ELE 606 Tema 10"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=600,
        help="Limite de amostras por base (default: 600)",
    )
    parser.add_argument(
        "--skip-clustering",
        action="store_true",
        help="Não executar experimentos de clustering",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Modo rápido: apenas TF-IDF e Naive Bayes",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Framework Experimental de NLP — Tema 10")
    print("=" * 60)

    if args.quick:
        embedders = ["tfidf"]
        classifiers = ["naive_bayes"]
    else:
        embedders = ["tfidf", "sentence_transformer"]
        classifiers = ["naive_bayes", "logistic_regression", "random_forest"]

    print("\n[1/3] Executando benchmark experimental...")
    records = run_benchmark(
        datasets=["20newsgroups", "reviews","noticias"],
        embedders=embedders,
        classifiers=classifiers,
        include_clustering=not args.skip_clustering,
        max_samples=args.max_samples,
    )
    print(f"  -> {len(records)} experimentos registrados em {LOGS_FILE}")

    print("\n[2/3] Gerando visualizações...")
    viz_paths = generate_all_visualizations()
    for p in viz_paths:
        print(f"  -> {p}")

    print("\n[3/3] Melhor experimento de classificação:")
    best = get_best_experiment("classification")
    if best:
        print(json.dumps({
            "experiment_id": best["experiment_id"],
            "dataset": best["dataset"],
            "embedding": best["embedding"]["name"],
            "algorithm": best["algorithm"]["algorithm"],
            "f1_weighted": best["metrics"]["f1_weighted"],
            "accuracy": best["metrics"]["accuracy"],
        }, indent=2, ensure_ascii=False))

        if best.get("errors_sample"):
            print("\nAnálise de erros (via API LLM, se configurada):")
            explanation = explain_errors_batch(best["errors_sample"])
            print(explanation)
    else:
        print("  Nenhum experimento concluído com sucesso.")

    print("\n" + "=" * 60)
    print("Concluído. Inicie a API: uvicorn app:app --reload")
    print("=" * 60)


if __name__ == "__main__":
    main()
