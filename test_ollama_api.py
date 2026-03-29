#!/usr/bin/env python3
"""Test Ollama API endpoints to find which one works."""

import httpx
import sys

base_url = "http://localhost:11434"
model = "qwen3:0.6b"

# Test chat endpoint
print("\nTesting /api/chat endpoint...")
payload = {
    "model": model,
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": False
}
try:
    response = httpx.post(f"{base_url}/api/chat", json=payload, timeout=10)
    print(f"✓ /api/chat: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"  Response: {result.get('message', {}).get('content', '')[:100]}")
    else:
        print(f"  Error: {response.text[:200]}")
except Exception as e:
    print(f"✗ /api/chat failed: {e}")

# Test generate endpoint (older API)
print("\nTesting /api/generate endpoint...")
payload2 = {
    "model": model,
    "prompt": "Hello",
    "stream": False
}
try:
    response = httpx.post(f"{base_url}/api/generate", json=payload2, timeout=10)
    print(f"✓ /api/generate: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"  Response: {result.get('response', '')[:100]}")
except Exception as e:
    print(f"✗ /api/generate failed: {e}")
