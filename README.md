# Intelligent Sprint Analysis Using Agentic System for Startup Projects

## About

This project is a local-first sprint intelligence system for GitHub-based software teams. It combines feature extraction, retrieval-augmented generation, and multi-agent LLM reasoning to analyze sprint activity, identify risks, and generate evidence-backed recommendations.

The application runs against local Ollama models and can use GitHub data for ingestion and analysis workflows.

## Setup

### Requirements

- Python 3.11+
- Git
- Ollama installed locally
- A local Ollama server running at `http://localhost:11434`
- A local Ollama model installed; the default configured model is `qwen3:0.6b`
- A GitHub token for GitHub API-based ingestion

### Environment Variables

The GitHub scraper reads either `GITHUB_TOKEN` or `GH_TOKEN`.

The local LLM configuration uses these defaults:

- `OLLAMA_BASE_URL=http://localhost:11434`
- `OLLAMA_MODEL=qwen3:0.6b`

Example:

```bash
export GITHUB_TOKEN=your_github_token_here
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=qwen3:0.6b
```

If you want to use a different local model, make sure it is installed in Ollama and update `OLLAMA_MODEL` to match the exact model name.

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Start Ollama

```bash
ollama serve
ollama pull qwen3:0.6b
```

The application expects Ollama to be available locally and, by default, expects `qwen3:0.6b` to be installed.

### Run the App

```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

Or, if you prefer the existing package script:

```bash
npm run fastapi
```

Then open `http://localhost:8000` in your browser.

### Notes

- GitHub token access is required for API-backed GitHub ingestion.
- Local/offline repository workflows may still work without a token, but API-based collection expects one.
- The app now supports choosing from installed local Ollama models in the UI, but the default environment model remains `qwen3:0.6b`.

