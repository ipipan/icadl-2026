#!/usr/bin/env python3
"""Run ingest.py on the host with .env loaded safely (no shell `source`,
which breaks on unquoted values containing spaces like WORKSPACE_NAME)."""
import os
import sys


def load_env(path=".env") -> dict:
    env = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = load_env(os.path.join(root, ".env"))
    for k, v in env.items():
        os.environ.setdefault(k, v)
    os.environ["ANYTHINGLLM_URL"] = f"http://localhost:{env.get('ANYTHINGLLM_PORT', '3001')}"
    os.environ["DATA_FILE"] = os.path.join(root, env.get("DATA_FILE", "data/CBDU_ocr_gemini_pro_2.5.json"))

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import ingest

    ingest.main()
