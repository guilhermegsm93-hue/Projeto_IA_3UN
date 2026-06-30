"""Configurações globais do framework experimental."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RUNS_DIR = ROOT_DIR / "runs"
ARTIFACTS_DIR = RUNS_DIR / "artifacts"
MODELS_DIR = RUNS_DIR / "models"
LOGS_FILE = RUNS_DIR / "experiment_logs.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Sentence Transformer leve para execução local
SENTENCE_MODEL_NAME = "paraphrase-MiniLM-L6-v2"

# Subconjunto do 20 Newsgroups (4 categorias para experimentos mais rápidos)
NEWSGROUPS_CATEGORIES = [
    "sci.med",
    "sci.space",
    "rec.sport.baseball",
    "rec.sport.hockey",
]

DATASET_NAMES = ("20newsgroups", "reviews")
