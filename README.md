# Framework Experimental de NLP — Tema 10 (ELE 606)

Framework modular e reprodutível para experimentação comparativa em NLP, com representação intermediária em JSON, visualizações automáticas e API FastAPI com explicação via LLM.

## Pergunta central

Como organizar um framework experimental de NLP que seja **comparável**, **documentado** e **reutilizável**?

## Requisitos atendidos

| Requisito (Unidade 3) | Implementação |
|----------------------|---------------|
| 2 bases textuais | `20newsgroups` + `reviews` (CSV em `data/`) |
| 2 embeddings | TF-IDF + Sentence Transformers (`paraphrase-MiniLM-L6-v2`) |
| 3+ algoritmos | Naive Bayes, Regressão Logística, Random Forest + K-Means (clustering) |
| Pré-processamento | Módulo configurável com variantes (default, minimal, aggressive) |
| Representação intermediária | `runs/experiment_logs.json` + modelos em `runs/models/` |
| Métricas e visualizações | Accuracy, F1, Silhouette; gráficos em `runs/artifacts/` |
| API externa (LLM) | Explicação de predições via Gemini ou OpenAI |
| API local (FastAPI) | Rotas `/predict`, `/run_benchmark`, `/experiments` |

## Estrutura do projeto

```
├── data/                  # Bases textuais (reviews.csv gerado automaticamente)
├── src/
│   ├── ingestion.py       # Carga padronizada (texto_bruto, alvo)
│   ├── preprocessing.py   # Limpeza configurável
│   ├── embeddings.py      # TF-IDF e Sentence Transformers
│   ├── models.py          # Classificadores e clustering
│   ├── evaluation.py      # Motor de experimentos e logs JSON
│   ├── visualization.py   # Gráficos a partir dos logs
│   └── llm_service.py     # Integração Gemini/OpenAI
├── runs/
│   ├── experiment_logs.json
│   ├── artifacts/         # Gráficos e tabelas
│   └── models/            # Modelos campeões serializados
├── main.py                # Executa benchmark completo
├── app.py                 # Servidor FastAPI
├── requirements.txt
└── .env.example
```

## Instalação

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Configure a chave da API (opcional, para explicações LLM):

```bash
copy .env.example .env
# Edite .env e insira GEMINI_API_KEY
```

## Execução

### 1. Benchmark experimental (linha de comando)

```bash
python main.py
```

Modo rápido (sem Sentence Transformers):

```bash
python main.py --quick --max-samples 200
```

### 2. API FastAPI

```bash
uvicorn app:app --reload
```

Acesse a documentação interativa: http://127.0.0.1:8000/docs

#### Exemplos de chamadas

```bash
# Disparar benchmark via API
curl -X POST http://127.0.0.1:8000/run_benchmark -H "Content-Type: application/json" -d "{\"max_samples\": 400}"

# Classificar texto com explicação
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"text\": \"The spacecraft launched successfully into orbit.\", \"explain\": true}"
```

## Bases de dados

- **20 Newsgroups** (4 categorias): baixada automaticamente via scikit-learn.
- **Reviews**: `data/reviews.csv` é criado na primeira execução. Substitua por uma base real do Kaggle mantendo colunas `texto_bruto` e `alvo`.
- **Notícias (opcional)**: coloque `data/noticias.xlsx` (planilha de 6 classes) para usar como base alternativa.

## Representação intermediária

Cada experimento gera um registro JSON em `runs/experiment_logs.json` com:

- ID, timestamp e versão do framework
- Parâmetros (dataset, pré-processamento, embedding, algoritmo)
- Métricas (accuracy, F1 ou silhouette)
- Amostra de erros de classificação
- Caminho do modelo treinado

## Limitações conhecidas

- Sentence Transformers exige download do modelo na primeira execução (~90 MB).
- Explicações LLM dependem de chave de API e conexão com internet.
- A base de reviews sintética é pequena; substitua por dados reais para resultados mais robustos.

## Disciplina

Projeto desenvolvido para ELE 606 — Tópicos em Inteligência Artificial, UFRN, 2026.1.
