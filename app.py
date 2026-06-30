"""API FastAPI — serviço do framework experimental de NLP."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import LOGS_FILE
from src.evaluation import get_best_experiment, load_champion_model, predict_text, run_benchmark
from src.ingestion import list_available_datasets
from src.llm_service import explain_prediction
from src.visualization import generate_all_visualizations, load_experiment_logs, logs_to_dataframe

app = FastAPI(
    title="Framework Experimental de NLP — Tema 10",
    description=(
        "API para benchmark reprodutível, predição textual e explicação via LLM. "
        "ELE 606 — UFRN."
    ),
    version="1.0.0",
)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Texto a classificar")
    explain: bool = Field(True, description="Incluir explicação via API externa")
    dataset: Optional[str] = Field(
        None,
        description="Base cujo modelo campeão será usado (ex: 20newsgroups, reviews)",
    )


class BenchmarkRequest(BaseModel):
    max_samples: int = Field(600, ge=50, le=5000)
    include_clustering: bool = True


class PredictResponse(BaseModel):
    text: str
    prediction: str
    preprocessed: str
    explanation: Optional[str] = None
    champion_config: Optional[dict] = None


@app.get("/")
def root():
    return {
        "service": "NLP Experimental Framework",
        "tema": 10,
        "endpoints": ["/datasets", "/experiments", "/run_benchmark", "/predict", "/health"],
    }


@app.get("/health")
def health():
    champion = get_best_experiment("classification")
    return {
        "status": "ok",
        "logs_exist": LOGS_FILE.exists(),
        "n_experiments": len(load_experiment_logs()),
        "champion_loaded": load_champion_model() is not None,
        "best_f1": champion["metrics"]["f1_weighted"] if champion else None,
    }


@app.get("/datasets")
def datasets():
    return list_available_datasets()


@app.get("/experiments")
def experiments():
    logs = load_experiment_logs()
    df = logs_to_dataframe(logs)
    return {
        "total": len(logs),
        "classification_summary": df.to_dict(orient="records") if not df.empty else [],
        "best": get_best_experiment("classification"),
    }


@app.post("/run_benchmark")
def trigger_benchmark(body: BenchmarkRequest):
    records = run_benchmark(
        max_samples=body.max_samples,
        include_clustering=body.include_clustering,
    )
    viz = generate_all_visualizations()
    best = get_best_experiment("classification")
    return {
        "experiments_run": len(records),
        "logs_file": str(LOGS_FILE),
        "visualizations": viz,
        "best_experiment": best,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    try:
        result = predict_text(body.text, dataset=body.dataset)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    explanation = None
    if body.explain:
        explanation = explain_prediction(body.text, result["prediction"])

    champion = get_best_experiment("classification", dataset=body.dataset)
    champion_config = None
    if champion:
        champion_config = {
            "experiment_id": champion["experiment_id"],
            "dataset": champion["dataset"],
            "embedding": champion["embedding"]["name"],
            "algorithm": champion["algorithm"]["algorithm"],
            "f1_weighted": champion["metrics"]["f1_weighted"],
        }

    return PredictResponse(
        text=result["text"],
        prediction=result["prediction"],
        preprocessed=result["preprocessed"],
        explanation=explanation,
        champion_config=champion_config,
    )
