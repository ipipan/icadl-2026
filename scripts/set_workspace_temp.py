#!/usr/bin/env python3
"""Update only the workspace's openAiTemp setting (chatMode/topN/prompt unchanged)
without re-uploading or re-embedding any documents (ingest.py's upload_documents
has no dedup -- rerunning full ingest would double every document in the
workspace). Reads WORKSPACE_TEMP from .env unless overridden by argv[1].

Run:
  python3 scripts/set_workspace_temp.py         # uses .env's WORKSPACE_TEMP
  python3 scripts/set_workspace_temp.py 0.05    # overrides it
"""
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
    if len(sys.argv) > 1:
        env["WORKSPACE_TEMP"] = sys.argv[1]
    for k, v in env.items():
        os.environ[k] = v
    # .env's ANYTHINGLLM_URL targets the in-container hostname (used when ingest
    # runs inside docker compose); on the host we must hit the mapped port instead.
    os.environ["ANYTHINGLLM_URL"] = f"http://localhost:{env.get('ANYTHINGLLM_PORT', '3001')}"

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import ingest

    print(f"Setting openAiTemp={ingest.WORKSPACE_TEMP} on workspace '{ingest.WORKSPACE_NAME}' ...")
    ingest.get_or_create_workspace()
    print("Done (no documents touched).")
