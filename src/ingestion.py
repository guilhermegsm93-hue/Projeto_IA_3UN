"""Módulo de ingestão — padroniza bases textuais distintas."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_20newsgroups

from src.config import DATA_DIR, NEWSGROUPS_CATEGORIES, RANDOM_STATE


def _ensure_data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def load_20newsgroups(max_samples: int | None = 800) -> pd.DataFrame:
    """Carrega subconjunto do 20 Newsgroups (base 1)."""
    data = fetch_20newsgroups(
        subset="train",
        categories=NEWSGROUPS_CATEGORIES,
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    df = pd.DataFrame({"texto_bruto": data.data, "alvo": data.target})
    df["alvo"] = df["alvo"].map(lambda i: data.target_names[i])
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=RANDOM_STATE).reset_index(drop=True)
    return df


def _create_sample_reviews_csv(path: Path) -> None:
    """Gera CSV de reviews sintéticas quando a base externa não existe."""
    samples = [
        ("Este produto é excelente, superou minhas expectativas.", "positivo"),
        ("Péssima qualidade, quebrou no primeiro dia de uso.", "negativo"),
        ("Entrega rápida e embalagem impecável, recomendo.", "positivo"),
        ("Não vale o preço pago, decepcionante.", "negativo"),
        ("Atendimento ao cliente muito bom e produto funcional.", "positivo"),
        ("Veio com defeito e o suporte não resolveu.", "negativo"),
        ("Design bonito e fácil de usar no dia a dia.", "positivo"),
        ("Material frágil, não recomendo para uso intenso.", "negativo"),
        ("Custo-benefício ótimo para quem busca praticidade.", "positivo"),
        ("Demorou demais para chegar e veio errado.", "negativo"),
        ("Funciona perfeitamente, estou satisfeito com a compra.", "positivo"),
        ("Manual confuso e instalação complicada.", "negativo"),
        ("Melhor compra que fiz este ano, qualidade top.", "positivo"),
        ("Parou de funcionar após uma semana.", "negativo"),
        ("Leve, compacto e eficiente para viagens.", "positivo"),
        ("Cheiro forte de plástico ao abrir a caixa.", "negativo"),
        ("Bateria dura o dia inteiro sem problemas.", "positivo"),
        ("Tela pequena demais para o que prometiam.", "negativo"),
        ("Software intuitivo e atualizações frequentes.", "positivo"),
        ("Ruído excessivo durante o funcionamento.", "negativo"),
    ]
    # Expande variações para ter volume mínimo de treino
    rows: list[tuple[str, str]] = []
    prefixes = ["", "Na minha opinião, ", "Honestamente, ", "Comprei recentemente: "]
    for text, label in samples:
        for prefix in prefixes:
            rows.append((prefix + text, label))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["texto_bruto", "alvo"])
        writer.writerows(rows)


def load_reviews(csv_name: str = "reviews.csv") -> pd.DataFrame:
    """Carrega base de reviews (base 2) de CSV em data/ ou gera amostra."""
    _ensure_data_dir()
    path = DATA_DIR / csv_name
    if not path.exists():
        _create_sample_reviews_csv(path)
    df = pd.read_csv(path)
    required = {"texto_bruto", "alvo"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV deve conter colunas {required}, encontrado: {list(df.columns)}")
    return df[["texto_bruto", "alvo"]].dropna().reset_index(drop=True)


def load_news_excel(xlsx_name: str = "noticias.xlsx") -> pd.DataFrame | None:
    """Carrega planilha de notícias em 6 classes, se disponível."""
    path = DATA_DIR / xlsx_name
    if not path.exists():
        return None
    df = pd.read_excel(path)
    text_col = next(
        (c for c in df.columns if "texto" in c.lower() or "text" in c.lower()),
        None,
    )
    label_col = next(
        (c for c in df.columns if c.lower() in ("classe", "alvo", "label", "categoria")),
        None,
    )
    if text_col is None or label_col is None:
        raise ValueError(
            f"Planilha {xlsx_name} deve ter coluna de texto e de classe. "
            f"Colunas: {list(df.columns)}"
        )
    return df.rename(columns={text_col: "texto_bruto", label_col: "alvo"})[
        ["texto_bruto", "alvo"]
    ].dropna().reset_index(drop=True)


def load_dataset(name: str) -> pd.DataFrame:
    """Interface unificada de carga de bases."""
    loaders = {
        "20newsgroups": load_20newsgroups,
        "reviews": load_reviews,
        "noticias": load_news_excel,
    }
    if name not in loaders:
        raise ValueError(f"Dataset desconhecido: {name}. Opções: {list(loaders)}")
    result = loaders[name]()
    if result is None:
        raise FileNotFoundError(
            f"Base '{name}' não encontrada em {DATA_DIR}. "
            "Coloque o arquivo ou use outro nome de dataset."
        )
    return result


def list_available_datasets() -> dict[str, str]:
    """Retorna datasets disponíveis e descrição."""
    available = {
        "20newsgroups": "20 Newsgroups (4 categorias científicas/esportivas)",
        "reviews": "Reviews de produtos (positivo/negativo)",
    }
    if (DATA_DIR / "noticias.xlsx").exists():
        available["noticias"] = "Planilha de notícias em 6 classes"
    return available
