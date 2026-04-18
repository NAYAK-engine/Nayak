"""
nayak/__init__.py — NAYAK runtime engine public API.

What Chromium is to browsers, NAYAK is to robots and autonomous workers.
Powered by NVIDIA NIM (cloud) or Ollama (local — 100% free, no key needed).
"""

__version__ = "0.2.0"
__author__ = "NAYAK Authors"

try:
    from nayak.agent import Agent, AgentConfig
except ImportError as e:
    pass

__all__ = ["Agent", "AgentConfig"]
