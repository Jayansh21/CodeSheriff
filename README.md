# 🛡️ CodeSheriff

**AI-Powered GitHub Pull Request Reviewer**

> **Live:** [codesheriff.onrender.com](https://codesheriff.onrender.com) &nbsp;|&nbsp; **Model:** [jayansh21/codesheriff-bug-classifier](https://huggingface.co/jayansh21/codesheriff-bug-classifier)

CodeSheriff automatically reviews code diffs, classifies potential bugs using a fine-tuned CodeBERT model, prioritises issues by severity, and generates actionable fix suggestions using an LLM — posted directly as inline GitHub PR comments.

---

## How It Works

1. **GitHub webhook** receives a pull request event.
2. The PR diff is fetched and parsed into function-level code chunks.
3. **Language detection** identifies the programming language from file extensions.
4. A **fine-tuned CodeBERT** classifier predicts the bug category for each chunk.
5. A confidence gate and pattern-based refinement rules filter false positives.
6. Issues are prioritised by severity.
7. All issues are sent in a **single batch prompt** to a **Groq LLM** for explanation and fix generation.
8. The review is formatted and posted as **inline PR comments** on the affected lines.

---

## Language Support

CodeSheriff currently provides the most accurate analysis for **Python** code. The fine-tuned CodeBERT model was trained exclusively on Python datasets and achieves the best results on Python pull requests.

Other languages (JavaScript, TypeScript, Java, Go, C++, etc.) will still run through the full pipeline, but results may be less reliable. When a non-Python language is detected, the review output includes a warning:

> ⚠️ CodeSheriff is currently optimized for Python analysis.

**Future plans:** Multi-language model training and language-specific rule sets are on the roadmap.

---

## Architecture

```
GitHub PR Webhook
       │
       ▼
┌──────────────────────┐
│  Fetch PR Diff       │   (GitHub REST API)
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Diff Parser         │   (parse_diff.py — function-level chunking)
│  + Language Detection │   (language_detection.py — file extension mapping)
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  ML Classification   │   (classify_chunks.py → fine-tuned CodeBERT)
│  + Confidence Gate   │   (threshold filtering + pattern refinement)
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Issue Prioritiser   │   (prioritize_issues.py — severity ranking)
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Batch Explanation   │   (generate_fixes.py → Groq LLM, single prompt)
│  + Model Fallback    │   (llama-3.3-70b → llama-3.1-8b on rate limit)
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Review Formatter    │   (format_review.py → Markdown + inline comments)
└──────────────────────┘
```

The pipeline is orchestrated using **LangGraph** as a sequential state machine with 5 processing nodes.

---

## Example PR Review Output

```
🛡️ CodeSheriff Review

Language detected: Python

Issues found: 3
---

## Issue 1: Security Vulnerability
Confidence: 99%
File: backend/db.py
...

## Issue 2: Logic Flaw
Confidence: 95%
File: utils/math.py
...
```

When reviewing non-Python code:

```
🛡️ CodeSheriff Review

Language detected: JavaScript
⚠️ CodeSheriff is currently optimized for Python analysis.

Issues found: 1
...
```

---

## Bug Classes Detected

| ID  | Label                  | Example Pattern                         |
| --- | ---------------------- | --------------------------------------- |
| 0   | Clean                  | Well-formed code                        |
| 1   | Null Reference Risk    | `result.fetchone().name` without check  |
| 2   | Type Mismatch          | `"Hello " + age` (str + int)            |
| 3   | Security Vulnerability | SQL string concatenation, `os.system()` |
| 4   | Logic Flaw             | `range(len(items) + 1)` off-by-one      |

---

## Project Structure

```
CodeSheriff/
├── agents/                   # LangGraph agent pipeline
│   ├── graph.py             # Wires the 5 processing nodes
│   └── nodes/               # Individual pipeline steps
│       ├── parse_diff.py    # Diff parsing + language detection
│       ├── classify_chunks.py # ML classification + confidence gate
│       ├── prioritize_issues.py # Severity-based prioritisation
│       ├── generate_fixes.py # LLM fix generation (Groq)
│       └── format_review.py # Markdown review formatting
├── backend/                 # FastAPI server
│   ├── main.py              # Endpoints + webhook handler
│   ├── github_auth.py       # GitHub App JWT authentication
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
├── utils/                   # Shared utilities
│   ├── config.py            # Environment variables & constants
│   ├── logger.py            # Structured logging
│   └── language_detection.py # Programming language detection
├── tests/                   # PyTest test suite
├── scripts/                 # CLI runner + deployment helpers
│   ├── run_pipeline.py
│   ├── push_model_to_hub.py
│   └── deploy_inference_space.py
├── data/                    # (gitignored) Raw & processed datasets
├── models/                  # (gitignored) Saved model checkpoints
├── .env.example             # Template for required env vars
├── render.yaml              # Render deployment config
├── Procfile                 # Render fallback start command
├── runtime.txt              # Python version pin for Render
├── github-app-manifest.json # GitHub App creation template
├── requirements.txt         # Production dependencies
└── requirements-dev.txt     # Dev/test dependencies
```

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

## Running Locally

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

| Method | Path         | Rate Limit | Description                         |
| ------ | ------------ | ---------- | ----------------------------------- |
| GET    | `/`          | —          | Landing page                        |
| GET    | `/health`    | —          | Service liveness + model status     |
| GET    | `/ping`      | —          | Lightweight uptime check            |
| POST   | `/test-diff` | 5/min      | Review built-in sample diff         |
| POST   | `/review`    | 10/min     | Review a custom diff (JSON body)    |
| POST   | `/webhook`   | —          | GitHub App webhook receiver (async) |

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

> ⚠️ Training takes **3–6 hours on CPU** (much faster with GPU). The best model is saved to `models/codesheriff-model/final/`.

### Evaluate the Model

```bash
python -m ml.evaluate
```

Prints a classification report and saves it to `models/codesheriff-model/evaluation_report.txt`.

---

## Running Tests

```bash
pytest tests/ -v
```

All tests are designed to pass **before** model training completes (using heuristic fallbacks).

---

## Deployment

| Component           | Platform           | Notes                                        |
| ------------------- | ------------------ | -------------------------------------------- |
| Backend API         | Render             | FastAPI, auto-deploys from GitHub            |
| ML Inference Server | HuggingFace Spaces | Docker-based Space runs the model (CPU-only) |
| ML Model Weights    | HuggingFace Hub    | `jayansh21/codesheriff-bug-classifier`       |
| Webhook             | GitHub App         | Listens for PR events, reviews in background |

**Why a separate inference server?** Render free tier has 512 MB RAM. Loading PyTorch + the model takes ~500 MB, causing OOM crashes. By offloading inference to a HuggingFace Space, the Render backend stays under 100 MB.

### Deploying on Render

1. Connect your GitHub repository to [Render](https://render.com).
2. Set the following environment variables in the Render dashboard:
   - `GROQ_API_KEY` — Groq API key for LLM fix generation
   - `HF_TOKEN` — HuggingFace token
   - `MODEL_PATH` — HuggingFace model path (e.g. `jayansh21/codesheriff-bug-classifier`)
   - `INFERENCE_SPACE_ID` — HuggingFace Space ID (e.g. `jayansh21/codesheriff-inference`)
   - `USE_LOCAL_MODEL` — `false` (use remote inference)
   - `GITHUB_APP_ID`, `GITHUB_PRIVATE_KEY`, `GITHUB_WEBHOOK_SECRET` — GitHub App credentials

3. Render will auto-deploy on every push to `main`.

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

### Using the GitHub App

1. Create a GitHub App using the manifest in `github-app-manifest.json`.
2. Install the app on your repositories.
3. Configure the webhook URL to point to `https://your-render-url.onrender.com/webhook`.
4. CodeSheriff will automatically review every new pull request.

---

## LLM Architecture

CodeSheriff uses **Groq**-hosted Llama models to generate human-readable explanations and fix suggestions for detected issues.

| Role     | Model                     | Notes                                                      |
| -------- | ------------------------- | ---------------------------------------------------------- |
| Primary  | `llama-3.3-70b-versatile` | Best quality explanations                                  |
| Fallback | `llama-3.1-8b-instant`    | Used automatically when the primary model hits rate limits |

**Batch prompting:** All issues found in a PR are sent to the LLM in a single prompt, rather than one call per issue. This significantly reduces token usage and latency.

---

## Rate Limit Handling

Groq's free tier imposes a daily token quota (100K TPD). CodeSheriff handles this automatically:

1. The primary model (`llama-3.3-70b-versatile`) is tried first.
2. If Groq returns a **429 rate-limit** or **token quota** error, CodeSheriff immediately retries with the smaller fallback model (`llama-3.1-8b-instant`).
3. Exponential backoff is applied between retries for transient errors.
4. If both models are exhausted, a placeholder message is returned — the review still posts with ML classifications, just without LLM explanations.

The pipeline **never crashes** due to Groq rate limits.

---

## Use CodeSheriff on Your GitHub

There are three ways to use CodeSheriff depending on what you need:

### Option 1 — Install the GitHub App (recommended)

This is the easiest way. Once installed, CodeSheriff automatically reviews every new pull request on your repos.

1. **Create a GitHub App** using the manifest in `github-app-manifest.json` (or ask the project owner to install theirs on your repo).
2. **Set the webhook URL** to your deployed backend, e.g. `https://codesheriff.onrender.com/webhook`.
3. **Install the app** on whichever repositories you want reviewed.
4. Open a pull request — CodeSheriff will post inline review comments automatically.

> Contributors on your repo don't need to install anything. They just open PRs and see the review comments.

### Option 2 — Use the Live API

Anyone can send a unified diff to the public API — no authentication needed:

```bash
# Review your own diff
curl -X POST https://codesheriff.onrender.com/review \
  -H "Content-Type: application/json" \
  -d '{"diff": "<your unified diff text>"}'

# Quick test with the built-in sample
curl -X POST https://codesheriff.onrender.com/test-diff
```

### Option 3 — Self-host the entire project

Fork or clone the repository and run your own instance:

```bash
git clone https://github.com/jayansh21/CodeSheriff.git
cd CodeSheriff
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate      # macOS / Linux
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env` with your own keys:

| Variable                | Where to get it                                                          |
| ----------------------- | ------------------------------------------------------------------------ |
| `GROQ_API_KEY`          | Free at [console.groq.com](https://console.groq.com)                     |
| `GITHUB_APP_ID`         | From your GitHub App settings                                            |
| `GITHUB_PRIVATE_KEY`    | PEM key downloaded when creating the GitHub App                          |
| `GITHUB_WEBHOOK_SECRET` | Secret you set when creating the GitHub App                              |
| `HF_TOKEN`              | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| `INFERENCE_SPACE_ID`    | Your HuggingFace Space ID (or use `jayansh21/codesheriff-inference`)     |

Then start the server:

```bash
uvicorn backend.main:app --reload
```

The model loads automatically from HuggingFace Hub — no local GPU or model download required (unless you set `USE_LOCAL_MODEL=true`).

---

## Tech Stack

- **Python 3.11**
- **PyTorch** + **HuggingFace Transformers** — model training & inference
- **LangGraph** — agent orchestration (5-node sequential pipeline)
- **Groq API** (Llama 3.3 70B + 3.1 8B fallback) — LLM explanation generation
- **FastAPI** — REST API + landing page
- **SlowAPI** — rate limiting
- **HuggingFace Spaces** (Docker) — remote inference server
- **Scikit-learn** — evaluation metrics
- **PyTest** — testing
