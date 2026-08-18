# Simple RAG Pipeline

[![Eval Loop](https://github.com/cloudlesson95-arch/Simple_RAG/actions/workflows/evaluate.yml/badge.svg)](https://github.com/cloudlesson95-arch/Simple_RAG/actions/workflows/evaluate.yml)

An agentic Retrieval-Augmented Generation pipeline built deliberately from scratch. 
Every component was added incrementally with a clear engineering rationale and proper commit history.

## What This Project Demonstrates

| Skill | Implementation |
|---|---|
| **RAG fundamentals** | Document chunking, embedding, vector search, context-stuffed LLM prompting |
| **Agentic routing** | Pydantic-based structured output to route queries to the correct data source |
| **Self-correction** | Automatic query rephrasing when retrieval returns insufficient context |
| **Evaluation pipeline** | LLM-as-judge scoring with Precision@k metrics and pass/fail thresholds |
| **Containerization** | Dockerfile with baked-in vector index for reproducible deployments |
| **CI/CD** | GitHub Actions pipeline that blocks merges if AI accuracy drops below 80% |
| **Centralized config** | All hyperparameters (chunk size, k, models) in one file — not hardcoded |
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

## Project Structure

```
Simple_RAG/
├── src/
│   ├── app.py              # Single entry point (CLI: index, query, evaluate)
│   ├── config.py            # All hyperparameters and settings
│   ├── evaluator.py         # Evaluation pipeline with LLM-as-judge
│   ├── logging_config.py    # Centralized logging setup
│   ├── rag_agent.py         # Router + self-correction agent
│   ├── utils.py             # LLM factory (Groq / Gemini)
│   └── vectorstore.py       # Chunking, embedding, ChromaDB operations
├── data/                    # Source documents (3 files, different scales)
├── baseline/
│   └── question.txt         # 10 test questions with expected answers
├── .github/workflows/
│   └── evaluate.yml         # CI pipeline — runs eval on every push
├── Dockerfile               # Containerization with baked-in vector index
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

### Run the Evaluation

```bash
python -m src.app evaluate
```

Runs 10 test questions (including 3 adversarial) through the full pipeline. An LLM judge evaluates each answer. The script exits with code `1` if Precision@k drops below 80%.

**Current baseline: ~90% (9/10 correct) on average**

The remaining ~10% variance is due to LLM non-determinism — the same question can occasionally produce a different quality answer across runs. This is expected and accounted for by setting the CI threshold at 80% rather than 100%.

## Docker

The Dockerfile builds a fully self-contained image with the vector index pre-computed during the build step. No external database needed.

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
3. **Blocks the merge** if Precision@k drops below 80%

API keys are stored as GitHub Repository Secrets — never in code or in the Docker image.

This ensures that no prompt change, model swap, or code refactor can silently degrade the AI's accuracy.

## Test Corpus

Three data sources at deliberately different scales to test different failure modes:

| Source | Size | Purpose |
|---|---|---|
| `fictional_text.txt` | 6 lines | Verify exact retrieval on a tiny, controlled text |
| `cat-facts.txt` | ~150 lines | Test chunk boundary effects on medium-length content |
| `pydantic.llms-full.txt` | 4 MB | Real-world retrieval on actual technical documentation |

Test questions include **adversarial cases**: questions requiring information from multiple chunks, questions with answers not in the corpus, and questions phrased differently from the source text.

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

## Development Journey

This project was built incrementally across 6 phases. The commit history reflects each deliberate step:

1. **Phase 0 — Corpus & Questions:** Chose 3 documents at different scales. Wrote test questions (including adversarial) before any code.
2. **Phase 1 — Naive RAG:** Basic chunking → embedding → retrieval → LLM answer. Identified failure modes.
3. **Phase 2 — Retrieval Fixes:** Added evaluation metrics (Precision@k). Diagnosed and fixed retrieval quality issues.
4. **Phase 3 — Agentic Layer:** Added Pydantic-based router for source selection, self-correction loop for query rephrasing, and direct-answer path for non-retrieval questions.
5. **Phase 4 — MLOps:** Centralized config, structured logging, Docker containerization, GitHub Actions CI with automated eval gate.

## Key Technical Decisions

- **Local embeddings over API-based:** Using `all-MiniLM-L6-v2` locally eliminates API costs for embeddings and removes a network dependency during indexing. The model is ~80MB and runs on CPU. However, API-based approach is also present and was used in earlier versions and easy to switched between.
- **Pydantic structured output for routing:** Instead of free-text LLM classification, the router returns a typed `RouteDecision` object. This guarantees the response is always one of the valid sources — no regex parsing needed.
- **80% CI threshold (not 100%):** LLMs are non-deterministic. Setting the threshold below worst-case observed performance catches real regressions without flaking on normal variance.
- **Index baked into Docker image:** For a static corpus, embedding at build time guarantees the code and data are always in sync. No "works on my machine" issues.

## License

MIT
