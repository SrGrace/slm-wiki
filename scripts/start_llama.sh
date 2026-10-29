#!/usr/bin/env bash
# Serve the repo's GGUF with llama.cpp (brew install llama.cpp, or set LLAMA_SERVER=/path/to/llama-server).
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${SLM_WIKI_GGUF:-models/qwen3-8b/Qwen_Qwen3-8B-Q4_K_M.gguf}"
PORT="${SLM_WIKI_LLAMA_PORT:-18434}"
CTX="${SLM_WIKI_CTX:-32768}"
BIN="${LLAMA_SERVER:-llama-server}"

if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
  echo "Already serving on :${PORT}"; exit 0
fi
command -v "$BIN" >/dev/null || { echo "llama-server not found: brew install llama.cpp (or set LLAMA_SERVER)" >&2; exit 1; }
exec "$BIN" -m "$MODEL" -c "$CTX" -ngl 99 --port "$PORT" --host 127.0.0.1 --jinja
