# Simple RAG Pipeline

[![Eval Loop](https://github.com/cloudlesson95-arch/Simple_RAG/actions/workflows/evaluate.yml/badge.svg)](https://github.com/cloudlesson95-arch/Simple_RAG/actions/workflows/evaluate.yml)

An agentic Retrieval-Augmented Generation pipeline built deliberately from scratch. 
Every component was added incrementally with a clear engineering rationale and proper commit history.

## What This Project Demonstrates

| Skill | Implementation |
|---|---|
| **RAG fundamentals** | Document chunking, embedding, vector search, context-stuffed LLM prompting |
| **Agentic routing** | Pydantic-based structured output to route queries to the correct data source |
| **Classical ML routing** | Replacing LLM router with Logistic Regression (needs retrieval) & Nearest Centroid (source selection) |
| **Self-correction** | Automatic query rephrasing when retrieval returns insufficient context |
| **Evaluation pipeline** | LLM-as-judge scoring with Precision@k metrics, JSON baselines, and SQLite run history |
| **Containerization** | Dockerfile with baked-in vector index and ML models for reproducible deployments |
| **CI/CD** | GitHub Actions pipeline with automated pass/fail gates and Step Summaries |
| **Centralized config** | All hyperparameters (chunk size, k, models, routing method) in one file — not hardcoded |
| **Structured logging** | Python `logging` module replacing all `print()` statements |

## Architecture

```
User Query
    │
    ▼
┌──────────┐     ┌────────────────┐
│  Router  │────▶│ Route Decision │
│ (LLM)   │     │  (Pydantic)    │
└──────────┘     └───────┬────────┘
                         │
            ┌────────────┼────────────┐
            ▼            ▼            ▼
     ┌────────────┐ ┌──────────┐ ┌────────┐
     │ Vector DB  │ │ Vector DB│ │  LLM   │
     │ fictional  │ │ cat-facts│ │ direct │
     │ _text.txt  │ │   .txt   │ │ answer │
     └─────┬──────┘ └────┬─────┘ └───┬────┘
           │              │           │
           ▼              ▼           ▼
     ┌─────────────────────────────────────┐
     │         Answer LLM (Groq)           │
     │  + Self-correction loop (2 retries) │
      └─────────────────────────────────────┘
```

The router uses **Pydantic structured output** to make a typed decision about which data source to query — or whether to skip retrieval entirely (e.g., for math questions). If the first retrieval returns "I don't know," the agent automatically rephrases the query and retries up to 2 times.

## Classical ML Routing vs. LLM Router

In addition to the LLM router, the pipeline implements a **zero-cost, sub-millisecond classical ML router** using two models trained on document embeddings:

1. **Decision Point 1 ("Needs retrieval?"):** A **Logistic Regression** binary classifier (`src/classifier.py`) trained on query embeddings. Predicts whether a query requires vector search vs. direct LLM answer (greetings, math, general chitchat).
2. **Decision Point 2 ("Which source?"):** A **Nearest Centroid** classifier (`src/clustering.py`). Computes the mean embedding vector for each source document and routes query embeddings to the closest source centroid via Euclidean distance.

Switch between routing methods via environment variable or [`src/config.py`](src/config.py): `ROUTING_METHOD = "llm"` or `"classical"`.

### Performance Comparison

| Metric | LLM Router (`ROUTING_METHOD="llm"`) | Classical ML Router (`ROUTING_METHOD="classical"`) |
|---|---|---|
| **Precision@4 Score** | **90.0%** (9/10) | **90.0%** (9/10) |
| **Routing Latency** | ~500–800 ms (API call) | **< 1 ms** (local NumPy math) |
| **Token Cost per Query** | ~150 prompt tokens | **$0.00** (Zero API calls) |
| **Determinism** | Non-deterministic | **100% Deterministic** |
| **Offline Capability** | Requires internet/API key | **Fully offline** |

*Key takeaway:* For a fixed corpus with known topics, classical ML achieves **identical accuracy** to an LLM router while eliminating 100% of routing latency and API cost.

## Workflow Orchestration (N8N & FastAPI)

The core Python agent is wrapped in a **FastAPI** REST server (`src/api.py`). This allows **N8N** (a visual workflow orchestrator) to trigger the RAG pipeline via webhooks. 

```
Webhook ──▶ N8N HTTP Node ──▶ FastAPI POST /query ──▶ Python RAG Agent ──▶ Response
```
This architecture keeps the complex Pydantic logic in Python, while enabling non-engineers to connect the RAG system to Slack, emails, or CRMs visually in N8N.

## Project Structure

```
Simple_RAG/
├── src/
│   ├── api.py              # FastAPI REST server exposing RAG agent
│   ├── app.py              # Single entry point (CLI: index, query, evaluate, history, serve)
│   ├── classifier.py       # Logistic Regression classifier for retrieval necessity
│   ├── clustering.py       # Nearest Centroid classifier & t-SNE visualization
│   ├── config.py            # All hyperparameters and settings
│   ├── eval_db.py           # SQLite database wrapper for tracking evaluation history
│   ├── evaluator.py         # Evaluation pipeline with LLM-as-judge & result logger
│   ├── logging_config.py    # Centralized logging setup
│   ├── rag_agent.py         # Router (LLM / Classical) + self-correction agent
│   ├── semantic_cache.py    # Sub-millisecond vector similarity caching
│   ├── utils.py             # LLM factory (Groq / Gemini)
│   └── vectorstore.py       # Chunking, embedding, ChromaDB operations
├── clusters/                # Saved ML models & t-SNE visualization PNG
├── s_cache/                 # Persisted semantic cache store
├── data/                    # Source documents (3 files, different scales)
├── baseline/
│   └── questions.json       # 10 structured test questions with expected context & answers
├── n8n/
│   └── workflow.json        # Exported N8N visual workflow pipeline
├── .github/workflows/
│   └── evaluate.yml         # CI pipeline — runs eval and logs step summaries
├── Dockerfile               # Containerization with baked-in vector index & ML models
├── docker-compose.yml       # Orchestrates FastAPI + N8N containers
├── .dockerignore
├── .env.example             # Template for required API keys
├── requirements.txt
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.10+
- API keys for [Groq](https://console.groq.com/) or/and [Google AI Studio](https://aistudio.google.com/apikey)

### Local Setup

```bash
# Clone and enter the project
git clone https://github.com/YOUR_USERNAME/Simple_RAG.git
cd Simple_RAG

# Create virtual environment
python -m venv .venv
.venv/Scripts/Activate.ps1   # Windows PowerShell
# source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Configure API keys (copy the template and add your keys)
cp .env.example .env
# Edit .env and add your GROQ_API_KEY and GOOGLE_API_KEY
```

### Build the Vector Index

```bash
python -m src.app index
```

This loads 3 documents from `data/`, chunks them (500 tokens, 50 overlap), embeds them using `all-MiniLM-L6-v2` (local, no API needed), and stores them in ChromaDB.

### Query the System

```bash
python -m src.app query "What is a group of cats called?"
# Output: Clowder.

python -m src.app query "What is 2345 * 849?"
# Output: 2345 × 849 = 1,990,905
# (Router detects this needs no retrieval — answers directly)
```

### Run Evaluation & View History

```bash
# Run evaluation suite
python -m src.app evaluate

# View historical evaluation runs
python -m src.app history
```

Runs 10 test questions (including 3 adversarial) from `baseline/questions.json`. An LLM judge evaluates each answer and persists the run metrics to `baseline/eval_history.db`. The script exits with code `1` if Precision@k drops below 80%.

## Docker & Orchestration

The project uses Docker and Docker Compose for fully reproducible deployments. The vector index is pre-computed during the image build step, so no external database is needed.

### Running with Docker Compose (N8N + FastAPI)

To spin up both the RAG API and the N8N orchestrator:

```bash
docker compose up --build
```
- **FastAPI backend**: `http://localhost:8000/docs`
- **N8N UI**: `http://localhost:5678` (import `n8n/workflow.json` here to test the webhook flow).

### Running Standalone Image

```bash
# Build the image (downloads HuggingFace model + builds index)
docker build -t simple-rag .

# Run evaluation inside the container
docker run --rm --env-file .env simple-rag python -m src.app evaluate

# Run a query
docker run --rm --env-file .env simple-rag python -m src.app query "What is a group of cats called?"
```

> **Note:** The `.env` file is never copied into the Docker image (excluded via `.dockerignore`). API keys are injected at runtime via `--env-file`, keeping secrets out of the image layer history.

## CI/CD Pipeline

Every push to `main` triggers a GitHub Actions workflow that:

1. Builds the Docker image on a clean Ubuntu server
2. Runs the full evaluation suite inside the container
3. **Generates a formatted Markdown summary card** in GitHub Action Step Summaries
4. **Blocks the merge** if Precision@k drops below 80%

API keys are stored as GitHub Repository Secrets — never in code or in the Docker image.

## Test Corpus

Three data sources at deliberately different scales to test different failure modes:

| Source | Size | Purpose |
|---|---|---|
| `fictional_text.txt` | 6 lines | Verify exact retrieval on a tiny, controlled text |
| `cat-facts.txt` | ~150 lines | Test chunk boundary effects on medium-length content |
| `pydantic.llms-full.txt` | 4 MB | Real-world retrieval on actual technical documentation |

Test questions in `baseline/questions.json` include **adversarial cases**: questions requiring information from multiple chunks, questions with answers not in the corpus, and questions phrased differently from the source text.

## Configuration

All hyperparameters are centralized in [`src/config.py`](src/config.py):

| Parameter | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 500 | Characters per text chunk |
| `CHUNK_OVERLAP` | 50 | Overlap between chunks |
| `K_RETRIEVAL` | 4 | Top-k results for vector search |
| `MAX_RETRIES` | 2 | Self-correction rephrasing attempts |
| `MAIN_LLM_MODEL` | `groq` | LLM provider for routing and answering |
| `EVAL_LLM_MODEL` | `groq` | LLM provider for judge evaluation |
| `EMBEDDING_LOCAL_MODEL` | `all-MiniLM-L6-v2` | Local embedding model (~80MB) |
| `ROUTING_METHOD` | `classical` | Routing engine (`llm` or `classical`) |
| `ENABLE_SEMANTIC_CACHE` | `True` | Toggle sub-millisecond vector similarity caching |
| `CACHE_SIMILARITY_THRESHOLD` | `0.95` | Cosine similarity threshold for semantic cache hit |
| `SEMANTIC_CACHE_DIR` | `s_cache` | Directory where semantic cache is persisted |
| `EVAL_QUESTIONS_PATH` | `baseline/questions.json` | Path to benchmark JSON test cases |
| `EVAL_DB_PATH` | `baseline/eval_history.db` | Path to SQLite evaluation history database |

## Development Journey

This project was built incrementally across 6 phases. The commit history reflects each deliberate step:

1. **Phase 0 — Corpus & Questions:** Chose 3 documents at different scales. Wrote test questions (including adversarial) before any code.
2. **Phase 1 — Naive RAG:** Basic chunking → embedding → retrieval → LLM answer. Identified failure modes.
3. **Phase 2 — Retrieval Fixes:** Added evaluation metrics (Precision@k). Diagnosed and fixed retrieval quality issues.
4. **Phase 3 — Agentic Layer:** Added Pydantic-based router for source selection, self-correction loop for query rephrasing, and direct-answer path for non-retrieval questions.
5. **Phase 4 — MLOps:** Centralized config, structured logging, Docker containerization, GitHub Actions CI with automated eval gate.
6. **Phase 5 — Orchestration (N8N):** Built a FastAPI backend to expose the agent and orchestrated it with N8N for visual webhook execution.
7. **Phase 6 — Classical ML Routing & Semantic Caching:** Replaced LLM router with Logistic Regression & Nearest Centroid classifiers (<1ms, $0 token cost), and added a vector similarity semantic cache layer with selective invalidation on evaluation denial.

## Key Technical Decisions

- **Local embeddings over API-based:** Using `all-MiniLM-L6-v2` locally eliminates API costs for embeddings and removes a network dependency during indexing. The model is ~80MB and runs on CPU. However, API-based approach is also present and was used in earlier versions and easy to switched between.
- **Pydantic structured output for routing:** Instead of free-text LLM classification, the router returns a typed `RouteDecision` object. This guarantees the response is always one of the valid sources — no regex parsing needed.
- **80% CI threshold (not 100%):** LLMs are non-deterministic. Setting the threshold below worst-case observed performance catches real regressions without flaking on normal variance.
- **Index baked into Docker image:** For a static corpus, embedding at build time guarantees the code and data are always in sync. No "works on my machine" issues.

## License

MIT
