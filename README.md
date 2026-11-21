# CBDU RAG System

RAG over OCR'd historical Polish pamphlets from [CBDU](https://cbdu.ijppan.pl), backed by a local LLM.

This is the quickstart. For implementation details (chunking strategy, date-notation
parsing, system prompt structure, GPU allocation, eval methodology), see
[ARCHITECTURE.md](ARCHITECTURE.md). For the hallucination-fix investigation
specifically, see [HALLUCINATION_FIXES.md](HALLUCINATION_FIXES.md) and
[notebooks/eval_analysis.ipynb](notebooks/eval_analysis.ipynb).

## Stack

| Service | Role |
|---|---|
| [vLLM](https://github.com/vllm-project/vllm) | LLM inference — `Qwen/Qwen3.6-35B-A3B-FP8` (pre-quantized MoE, 262k context, ~35GB VRAM) |
| [TEI](https://github.com/huggingface/text-embeddings-inference) | Embeddings — `BAAI/bge-m3` (multilingual, ~2 GB VRAM) |
| TEI (rerank) | Reranking — `BAAI/bge-reranker-v2-m3` cross-encoder on GPU (~1-2 GB VRAM) |
| [Qdrant](https://qdrant.tech) | Vector store |
| [AnythingLLM](https://anythinglLM.com) | RAG UI, agent mode (default chat mode), source citation — runs with 6 source patches (see `patches/anythingllm/`) |

Agent mode (`chatMode: "automatic"`) is the default for every chat, not an
opt-in `@agent` prefix — a source patch forces the agent to always search
`rag-memory` before answering. See [ARCHITECTURE.md](ARCHITECTURE.md) §4 for
why, and for the reranking pipeline.

## Requirements

- Docker + Docker Compose plugin
- NVIDIA GPU with ≥ 8 GB VRAM
- CUDA 12.2+ (for TEI; for older GPUs change TEI image tag to `turing-1.5`)
- NVIDIA Container Toolkit

### Install NVIDIA Container Toolkit (if missing)

Error `could not select device driver "nvidia"` means it's not installed:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Setup

### 1. Configure

```bash
cp .env.example .env
```

Edit `.env` — minimum required:

| Variable | Description |
|---|---|
| `JWT_SECRET` | Random hex — `python3 -c "import secrets; print(secrets.token_hex(32))"` |
| `AUTH_TOKEN` | UI login password — same command with `token_hex(24)` |

### 2. First launch

```bash
docker compose --profile prequant up -d
```

(`--profile prequant` runs the pre-quantized FP8/AWQ/GPTQ vLLM variant —
production uses `Qwen/Qwen3.6-35B-A3B-FP8`. Plain `docker compose up`, with
no profile, starts **neither** vLLM variant — see `docker-compose.yml` for
the `--profile bnb` alternative for raw bf16 checkpoints.)

Boot order (automatic):
1. Qdrant starts
2. vLLM downloads and loads `Qwen/Qwen3.6-35B-A3B-FP8` (pre-quantized, first
   run several GB download + ~8-10 min for CUDA graph capture and FP8 MoE
   kernel JIT compilation — this compile cost is paid again on every
   `--force-recreate`, not just the first run, since the compile cache isn't
   volume-mounted)
3. TEI downloads `BAAI/bge-m3` (~2 GB, first run ~2 min); TEI-rerank
   downloads `BAAI/bge-reranker-v2-m3` similarly
4. AnythingLLM starts once all three pass healthcheck, with its 6 source
   patches (`patches/anythingllm/`) applied via read-only volume mounts

`ingest` is **not** started by `docker compose up` (it sits behind the `ingest`
profile) — run it manually in step 3.

### 3. Ingest documents

```bash
docker compose run --rm ingest
```

No manual API-key setup needed: `ingest.py` auto-provisions a developer API
key on first run (logging in with `AUTH_TOKEN`, same as the UI would) and
writes it back to `.env` as `ANYTHINGLLM_API_KEY`, so later host-side tools
(`eval_workspace.py`, `set_workspace_temp.py`) pick it up automatically.

Uploads and embeds 314 records as ~2689 individual page-documents into the **CBDU Historical Pamphlets** workspace. Takes 10–20 minutes on first run. Safe to re-run — AnythingLLM deduplicates by content hash.

### 4. Use

Open [http://localhost:3001](http://localhost:3001), select workspace **CBDU Historical Pamphlets** and ask questions. Every chat runs in agent mode by default (always searches the corpus before answering) and includes cited source documents with links back to the CBDU catalogue's info page for each item.

## Data

`data/CBDU_ocr_gemini_pro_2.5.json` — 314 ingested records with Gemini 2.5 Pro OCR or KORBA transcription (2689 pages, ~9300 embedded chunks). Each document contains:

- Modern Polish transcription (primary — normalised, better for search)
- Diplomatic transcription (original historical spelling)
- Bibliographic metadata: title, year, language, place, publisher, source URL
- Scholarly commentary (general/historical/linguistic, ~155/314 records) —
  embedded as an inline `[Komentarz ...]` marker alongside `[Datowanie:...]`,
  see [ARCHITECTURE.md](ARCHITECTURE.md) §3

## VRAM tuning

Production runs `Qwen/Qwen3.6-35B-A3B-FP8` (`--profile prequant`) at
`VLLM_GPU_MEM=0.6`, `VLLM_MAX_LEN=262144` (the model's native max context),
`--max-num-seqs 32`, on a 96GB card (~58GB reserved for vLLM, ~977k-token KV
cache pool, ~3.7x concurrent full-length requests — see
[ARCHITECTURE.md](ARCHITECTURE.md) §2). TEI and TEI-rerank share the same
GPU (~1-2GB each).

| Scenario | Fix |
|---|---|
| OOM on startup | Lower `VLLM_GPU_MEM` in `.env` (currently `0.6`) |
| GPU < 6 GB | Run TEI on CPU: change TEI image to `cpu-1.5`, remove its `deploy` block |
| Smaller / less VRAM | Use `--profile bnb` with a smaller raw checkpoint (e.g. `VLLM_MODEL=Qwen/Qwen3.5-4B`) instead of the FP8 model |
| Crash: "max_num_seqs exceeds available Mamba cache blocks" | This model family has hybrid Mamba/attention layers — CUDA graph capture needs one Mamba cache block per `--max-num-seqs` slot. Lower `--max-num-seqs` to match real expected concurrency rather than raising `VLLM_GPU_MEM` |
| Want a longer/shorter context | Raise/lower `VLLM_MAX_LEN` — independent of `VLLM_GPU_MEM` (the KV cache pool size doesn't change, only how many concurrent full-length requests it can serve). Every change requires a full container recreate (~8-10 min, CUDA graph + FP8 kernel recompile) |

## Re-ingesting

Safe re-run (adds new docs, deduplicates existing):
```bash
docker compose run --rm ingest
```

Clean re-ingest (removes all embedded documents, then re-ingests from scratch):
```bash
docker compose run --rm -e CLEAN_WORKSPACE=1 ingest
```
