"""Serviço de explicação via API externa (Gemini ou OpenAI)."""

from __future__ import annotations

import os
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()


def _gemini_explain(text: str, prediction: str, api_key: str) -> str:
    prompt = (
        f"Você é um especialista em NLP. Um classificador rotulou o texto abaixo como '{prediction}'.\n"
        f"Analise criticamente: se a categoria fizer sentido, explique brevemente o porquê (em até 2 frases).\n"
        f"Se a categoria estiver visivelmente errada, brevemente aponte o erro e determine uma categoria hipotética, criada por você, que melhor se adequaria.\n\n"
        f"Texto: {text[:1500]}"
    )
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 1024},
    }
    with httpx.Client(timeout=30.0) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


def _openai_explain(text: str, prediction: str, api_key: str) -> str:
    prompt = (
        f"O modelo classificou este texto como '{prediction}'. "
        f"Explique brevemente em português o porquê.\n\nTexto: {text[:1500]}"
    )
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0.3,
    }
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def explain_prediction(
    text: str,
    prediction: str,
    provider: Optional[str] = None,
) -> str:
    """Gera explicação via LLM externa ou fallback local."""
    provider = (provider or os.getenv("LLM_PROVIDER", "gemini")).lower()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                return _openai_explain(text, prediction, api_key)
            except Exception as exc:
                return _fallback_explanation(prediction, str(exc))
    else:
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key and api_key != "sua_chave_aqui":
            try:
                return _gemini_explain(text, prediction, api_key)
            except Exception as exc:
                return _fallback_explanation(prediction, str(exc))

    return _fallback_explanation(prediction, "API key não configurada")


def _fallback_explanation(prediction: str, reason: str) -> str:
    return (
        f"Classificação: {prediction}. "
        f"(Explicação via LLM indisponível: {reason}. "
        "Configure GEMINI_API_KEY ou OPENAI_API_KEY no arquivo .env.)"
    )


def explain_errors_batch(
    errors: list[dict],
    max_items: int = 5,
) -> str:
    """Resume erros de classificação para análise qualitativa."""
    if not errors:
        return "Nenhum erro encontrado na amostra analisada."
    sample = errors[:max_items]
    lines = ["Principais erros de classificação:"]
    for i, err in enumerate(sample, 1):
        lines.append(
            f"{i}. Verdadeiro={err['true_label']}, Previsto={err['predicted_label']}: "
            f"{err['text'][:120]}..."
        )
    summary = "\n".join(lines)
    return explain_prediction(
        summary,
        "análise de erros",
    )
