---
title: CodeSheriff
emoji: 🔍
colorFrom: blue
colorTo: gray
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: true
---

# CodeSheriff

AI-powered GitHub pull request reviewer.

Catches security vulnerabilities, null reference risks, type mismatches,
and logic flaws automatically — the moment a PR is opened.

## Live Demo

Paste any Python diff into the text area and click Analyze.

## Install

Install the GitHub App to get automatic reviews on your own repositories.

## Architecture

- Bug classifier: fine-tuned CodeBERT (microsoft/codebert-base)
- Agent pipeline: LangGraph
- Fix generation: Groq API (Llama 3.3 70B)
- Backend: FastAPI on Render
- Landing page: Streamlit on HuggingFace Spaces
