# NAYAK

### The Open Source Runtime Engine for Autonomous Robots & AI Agents

> What Chromium did for browsers — NAYAK does for the physical world.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12+-blue)](https://python.org)
[![Status](https://img.shields.io/badge/Status-Alpha-orange)]()

---

## What Is NAYAK?

NAYAK is a free, open source runtime engine that powers 
autonomous AI agents that work in the real world.

You give it a goal. It thinks. It acts. It remembers. 
It completes the task. No human needed.

---

## Quick Start

### 1. Get a Free AI API Key

Choose any one of these free options:

| Provider | Free Tier | Get Key |
|---|---|---|
| Google Gemini | 1500 req/day | aistudio.google.com |
| Groq | High limits | console.groq.com |
| NVIDIA NIM | Free credits | build.nvidia.com |

### 2. Install NAYAK
```bash
git clone https://github.com/nayak-engine/nayak.git
cd nayak
pip install -e .
playwright install chromium
```

### 3. Configure Your API Key
```bash
cp .env.example .env
```

Open .env and paste your API key.

### 4. Run Your First Agent
```bash
nayak run --goal "Search for the top 3 AI robotics companies and save a report to report.md"
```

Watch NAYAK open a real browser, search the web, 
read websites, and save a real report. Fully autonomous.

---

## How It Works

NAYAK uses a Plan-Then-Execute architecture:

1. PLAN — AI creates a full action plan for the goal
2. EXECUTE — NAYAK executes every action in the browser
3. REMEMBER — Every result saved to persistent memory
4. REPORT — AI compiles all findings into final output

This uses only 3-5 AI API calls per task — 
works perfectly on any free tier.

---

## Architecture

---

## Bring Your Own AI

NAYAK works with any AI provider:
```python
from nayak import Agent

agent = Agent(
    goal="Your goal here",
    api_key="your_key_here",
    provider="gemini"  # or groq, nvidia, openai, anthropic
)

agent.run()
```

---

## Roadmap

- [x] Autonomous web browsing agent
- [x] Persistent memory
- [x] Plan-then-execute architecture
- [x] Multiple AI provider support
- [ ] Physical robot support
- [ ] Multi-agent coordination
- [ ] Vision models integration
- [ ] Mobile device control

---

## Contributing

NAYAK is built in the open by the community.
Read CONTRIBUTING.md to get started.

---

## License

Apache License 2.0 — Free forever.

---

*NAYAK — Built in the open. For the world.*
