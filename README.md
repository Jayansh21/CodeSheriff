# 🛡️ CodeSheriff

**AI-Powered GitHub Pull Request Reviewer**

> **Live:** [codesheriff.onrender.com](https://codesheriff.onrender.com) &nbsp;|&nbsp; **Model:** [jayansh21/codesheriff-bug-classifier](https://huggingface.co/jayansh21/codesheriff-bug-classifier)

CodeSheriff automatically reviews code diffs, classifies potential bugs using a fine-tuned CodeBERT model, prioritises issues by severity, and generates actionable fix suggestions using an LLM.

---

## Architecture

```
Pull Request Diff
       │
       ▼
┌──────────────┐
│  Diff Parser │   (parse_diff.py)
└──────┬───────┘
       │
       ▼
┌──────────────────────┐
│  Bug Classifier      │   (classify_chunks.py → fine-tuned CodeBERT)
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Issue Prioritiser   │   (prioritize_issues.py)
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Fix Generator       │   (generate_fixes.py → Groq / Llama 3)
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Review Formatter    │   (format_review.py → Markdown output)
└──────────────────────┘
```

The pipeline is orchestrated using **LangGraph** as a sequential state machine.

---

## Project Structure

```
CodeSheriff/
├── agents/                   # LangGraph agent pipeline
│   ├── graph.py             # Wires the 5 processing nodes
│   └── nodes/               # Individual pipeline steps
├── backend/                 # FastAPI server
│   ├── main.py              # Endpoints + webhook handler
│   ├── github_auth.py       # GitHub App JWT + signature verification
│   └── api_test.py          # Quick smoke test script
├── ml/                      # Machine learning modules
│   ├── dataset.py           # Data preparation & heuristic labelling
│   ├── train.py             # Fine-tuning CodeBERT
│   ├── evaluate.py          # Model evaluation & metrics
│   └── inference.py         # Dual-mode prediction (HF Space or local)
├── spaces/inference/        # HuggingFace Space (Docker) — remote ML server
│   ├── app.py               # FastAPI model server
│   ├── Dockerfile           # CPU-only torch + transformers
│   └── requirements.txt
├── data/                    # (gitignored) Raw & processed datasets
├── models/                  # (gitignored) Saved model checkpoints
├── utils/                   # Shared config & logging
├── tests/                   # PyTest test suite
├── scripts/                 # CLI runner + deployment helpers
│   ├── run_pipeline.py
│   ├── push_model_to_hub.py
│   └── deploy_inference_space.py
├── .env.example             # Template for required env vars
├── render.yaml              # Render deployment config
├── Procfile                 # Render fallback start command
├── runtime.txt              # Python version pin for Render
├── github-app-manifest.json # GitHub App creation template
├── requirements.txt         # Production dependencies
└── requirements-dev.txt     # Dev/test dependencies
```

---

## Bug Classes Detected

| ID  | Label                  | Example Pattern                         |
| --- | ---------------------- | --------------------------------------- |
| 0   | Clean                  | Well-formed code                        |
| 1   | Null Reference Risk    | `result.fetchone().name` without check  |
| 2   | Type Mismatch          | `if x = 100:` (assignment in condition) |
| 3   | Security Vulnerability | SQL string concatenation                |
| 4   | Logic Flaw             | `range(len(items) + 1)` off-by-one      |

---

## Setup Instructions

### 1. Create & Activate Virtual Environment

```bash
cd CodeSheriff
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

> **[MANUAL STEP]** You must obtain a Groq API key from [console.groq.com](https://console.groq.com) and paste it into `.env`.

---

## Training the Model

### Prepare the Dataset

```bash
python -m ml.dataset
```

This downloads `code_search_net` (Python split), applies heuristic labels, and saves `data/processed/labeled_dataset.csv`.

### Train the Classifier

```bash
python -m ml.train
```

> ⚠️ Training takes **3–6 hours on CPU**. The best model is saved to `models/codesheriff-model/final/`.

### Evaluate the Model

```bash
python -m ml.evaluate
```

Prints a classification report and saves it to `models/codesheriff-model/evaluation_report.txt`.

---

## Running the Pipeline

### CLI (no server required)

```bash
python scripts/run_pipeline.py
```

Runs the full pipeline on a built-in sample diff and prints the review to the console.

### FastAPI Server

```bash
uvicorn backend.main:app --reload
```

Then test:

```bash
python backend/api_test.py
```

**Endpoints:**

| Method | Path         | Rate Limit  | Description                         |
| ------ | ------------ | ----------- | ----------------------------------- |
| GET    | `/`          | —           | Landing page                        |
| GET    | `/health`    | —           | Service liveness + model status     |
| GET    | `/ping`      | —           | Lightweight uptime check            |
| POST   | `/test-diff` | 5/min       | Review built-in sample diff         |
| POST   | `/review`    | 10/min      | Review a custom diff (JSON body)    |
| POST   | `/webhook`   | —           | GitHub App webhook receiver (async) |

---

## Running Tests

```bash
pytest tests/ -v
```

All tests are designed to pass **before** model training completes (using heuristic fallbacks).

---

## Deployment

| Component           | Platform           | Notes                                          |
| ------------------- | ------------------ | ---------------------------------------------- |
| Backend API         | Render             | FastAPI, auto-deploys from GitHub              |
| ML Inference Server | HuggingFace Spaces | Docker-based Space runs the model (CPU-only)   |
| ML Model Weights    | HuggingFace Hub    | `jayansh21/codesheriff-bug-classifier`         |
| Webhook             | GitHub App         | Listens for PR events, reviews in background   |

**Why a separate inference server?** Render free tier has 512 MB RAM. Loading PyTorch + the model takes ~500 MB, causing OOM crashes. By offloading inference to a HuggingFace Space, the Render backend stays under 100 MB.

### Push Model to HuggingFace Hub

```bash
huggingface-cli login
python scripts/push_model_to_hub.py
```

### Deploy Inference Space

```bash
python scripts/deploy_inference_space.py
```

This uploads `spaces/inference/` as a Docker-based HuggingFace Space.

Then set in your `.env` (and Render Dashboard):
```
MODEL_PATH=your-username/codesheriff-bug-classifier
INFERENCE_SPACE_ID=your-username/codesheriff-inference
USE_LOCAL_MODEL=false
```

---

## Tech Stack

- **Python 3.11**
- **PyTorch** + **HuggingFace Transformers** — model training & inference
- **LangGraph** — agent orchestration (5-node sequential pipeline)
- **Groq API** (Llama 3.3 70B) — LLM fix generation
- **FastAPI** — REST API + landing page
- **SlowAPI** — rate limiting
- **HuggingFace Spaces** (Docker) — remote inference server
- **Scikit-learn** — evaluation metrics
- **PyTest** — testing

---

## License

MIT
