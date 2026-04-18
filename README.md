# NAYAK

### The Open Source Runtime Engine for Autonomous Robots & AI Agents

> What Chromium did for browsers — NAYAK does for the physical world.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12+-blue)](https://python.org)
[![Ollama](https://img.shields.io/badge/AI-Ollama-black)](https://ollama.com)
[![Gemini](https://img.shields.io/badge/AI-Gemini-blue)](https://aistudio.google.com)
[![Status](https://img.shields.io/badge/Status-Alpha-orange)]()

---

## What Is NAYAK?

NAYAK is a free open source runtime engine that powers 
autonomous AI agents that work in the real world.

You give it a goal in plain English.
It thinks. It acts. It remembers. It completes the task.
No human needed.

---

## Quick Start

### Step 1 — Install Python 3.12
Download from: https://www.python.org/downloads/release/python-31210/
Check "Add Python to PATH" during install.

### Step 2 — Install Ollama (Free Local AI)
Download from: https://ollama.com
Then run:
```bash
ollama pull llama3.2
ollama serve
```

### Step 3 — Install & Configure NAYAK
**Windows:**
1. Clone the repository and navigate inside.
2. Double click `setup.bat` to install everything automatically.
3. Open `.env` and configure your API keys if needed (Ollama works without keys!).

**Mac/Linux:**
```bash
git clone https://github.com/nayak-engine/nayak.git
cd nayak
python -m venv .venv
source .venv/bin/activate
pip install -e .
playwright install chromium
cp .env.example .env
```

### Step 4 — Run Your First Agent
**Windows:** Double click `nayak.bat` or run:
```bash
nayak run "Search for the top 3 AI robotics companies and save a report to report.md"
```

**Mac/Linux:**
```bash
nayak run "Search for the top 3 AI robotics companies and save a report to report.md"
```

Watch NAYAK open a real browser, search the web, read websites, and save a real report. Fully autonomous.

---

## How It Works

NAYAK uses a perceive → think → act → remember loop:

1. **SEE** — Captures the current browser state
2. **THINK** — AI decides the next best action toward the goal
3. **ACT** — Executes real browser actions (navigate, click, type, extract, save)
4. **REMEMBER** — Every result is saved to persistent SQLite memory

A final markdown report is always saved before exit.

---

## License

Apache License 2.0 — Free forever.

*NAYAK — Built in the open. For the world.*
