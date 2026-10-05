#!/usr/bin/env bash
# Phase 2b — WikiText-2 perplexity for each GGUF via llama.cpp (the quality ladder).
# Produces the Δ-perplexity-vs-f16 numbers in the README results table.
#
# Prereq: wiki.test.raw built once from the HF dataset:
#   python - <<'PY'
#   from datasets import load_dataset
#   ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
#   open("wiki.test.raw","w").write("\n\n".join(ds["text"]))
#   PY
#
# Usage: bash src/eval/gguf_perplexity.sh [GGUF_DIR] [BIN_DIR] [RAW_FILE]
set -e
GG=${1:-/kaggle/working/gguf}
BIN=${2:-/kaggle/working/llamacpp-bin}
RAW=${3:-wiki.test.raw}
export LD_LIBRARY_PATH="$BIN"

for m in f16 q8_0 q4_k_m q4_0; do
  echo "===== $m ====="
  "$BIN/llama-perplexity" -m "$GG/qwen-0.5b-$m.gguf" -f "$RAW" -c 512 --chunks 200 2>&1 \
      | grep -iE "final estimate|PPL ="
done
# Compare each quant row against the f16 row (same implementation/settings). The HF-measured
# perplexity (12.665) uses a different implementation and is NOT comparable to these.
