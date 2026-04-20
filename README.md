# NAYAK

### The Open Source Operating System for Autonomous Robots & AI Agents

> What Chromium did for browsers — NAYAK does for the physical world.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12+-blue)](https://python.org)
[![Version](https://img.shields.io/badge/Version-0.2.0-green)]()
[![Status](https://img.shields.io/badge/Status-Alpha-orange)]()
[![Ollama](https://img.shields.io/badge/AI-Ollama-black)](https://ollama.com)
[![Gemini](https://img.shields.io/badge/AI-Gemini-blue)](https://aistudio.google.com)

---

## What Is NAYAK?

NAYAK is a free open source operating system for autonomous robots and AI agents that work in the real physical world.

Not a framework. Not a library. Not a tool.
A full OS — like Windows for PCs, like Chromium for browsers — but for robots and autonomous workers.

You give it a goal in plain English.
It thinks. It plans. It acts. It remembers. It completes the task.
No human needed.

---

## The 9-Layer OS Architecture

NAYAK is built as a complete operating system with 9 layers:

| Layer | Name | Status | Description |
|-------|------|--------|-------------|
| L1 | Hardware Abstraction (HAL) | ✅ Live | Connect any robot hardware — cameras, motors, sensors |
| L2 | Perception Engine | ✅ Live | See and understand the world in real time |
| L3 | Cognition Engine | ✅ Live | AI brain — plan, decide, reason (Ollama + Gemini) |
| L4 | Action Engine | ✅ Live | Execute decisions — click, type, navigate, save |
| L5 | Memory Engine | ✅ Live | Persistent SQLite memory — never forgets |
| L6 | Communication Engine | ✅ Live | Talk to humans and other robots |
| L7 | Safety Engine | ✅ Live | Emergency stop, capability control, threat detection |
| L8 | Update Engine | ✅ Live | Over the air updates, skill installation |
| L9 | Developer Platform | ✅ Live | SDK for building skills and plugins on NAYAK |

---

## Fast Path Intelligence

NAYAK is built to be fast. Not every goal needs a full agent loop.

```bash
nayak run "What is the current price of Bitcoin?"
# Answer: Bitcoin current price: $75,052.00 USD
# Time: < 2 seconds
```

NAYAK automatically classifies every goal:
- **INSTANT** — single facts, prices, weather → answered in seconds
- **SIMPLE** — quick lookups → 2-3 steps
- **COMPLEX** — research, reports → full autonomous agent loop

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

### Step 3 — Install NAYAK

**Windows:**
```bash
git clone https://github.com/nayak-engine/nayak.git
cd nayak
setup.bat
```

**Mac/Linux:**
```bash
git clone https://github.com/nayak-engine/nayak.git
cd nayak
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m playwright install chromium
cp .env.example .env
```

### Step 4 — Run NAYAK

**Instant answer:**
```bash
python -m nayak run "What is the price of Ethereum?"
```

**Full autonomous agent:**
```bash
python -m nayak run "Research the top 5 AI robotics companies and save a report"
```

**Ask a question:**
```bash
python -m nayak ask "What is the capital of Japan?"
```

**View history:**
```bash
python -m nayak history
```

---

## How The OS Boots

When NAYAK starts, the runtime boots all 9 layers in order:

```bash
[L1] camera-hal         — READY  (Hardware abstraction backend)
[L1] raspberry-pi-hal   — READY  (Hardware abstraction backend)
[L3] cognition.ollama   — READY  (Cognition backend)
[L6] text-communication — READY  (Communication backend)
[L7] nayak-safety       — READY  (Safety engine)
[L8] nayak-updater      — READY  (Update engine)
[L9] nayak-platform     — READY  (Developer platform)
```

Every layer registers with the Module Registry.
Every action flows through the Event Bus.
Every decision is checked by the Safety Engine.

---

## Roadmap

- [x] v0.1.0 — Autonomous browser agent
- [x] v0.2.0 — Complete 9-layer OS architecture
- [ ] v0.3.0 — Physical robot support (Raspberry Pi)
- [ ] v0.4.0 — Multi-agent coordination
- [ ] v0.5.0 — Skill marketplace
- [ ] v1.0.0 — The standard. Every robot runs NAYAK.

---

## Contributing

We welcome contributors. Read [CONTRIBUTING.md](CONTRIBUTING.md) to get started.

---

## License

Apache License 2.0 — Free forever.

*NAYAK — Built in the open. For the world.*
