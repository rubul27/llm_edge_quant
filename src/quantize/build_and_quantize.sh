#!/usr/bin/env bash
# Phase 2a — build llama.cpp (CPU), convert Qwen HF -> GGUF f16, quantize the ladder.
#
# Usage:  bash src/quantize/build_and_quantize.sh [WORKDIR] [HF_MODEL_DIR]
# Defaults target Kaggle:  WORKDIR=/kaggle/working  HF_MODEL_DIR=/kaggle/working/qwen2.5-0.5b-instruct
#
# CPU build is intentional: llama-quantize is CPU-only anyway, and a 0.5B model
# evaluates fine on CPU. Avoids the Kaggle CUDA driver-stub linker error.
set -e
WORK=${1:-/kaggle/working}
MODEL=${2:-/kaggle/working/qwen2.5-0.5b-instruct}

cd "$WORK"
[ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp
cd llama.cpp

# Build only the tools we need.
cmake -B build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF
cmake --build build --config Release -j"$(nproc)" \
      --target llama-quantize llama-perplexity llama-cli

# Conversion deps.
pip install -q -r requirements/requirements-convert_hf_to_gguf.txt

# HF -> GGUF f16 (the un-quantized GGUF-track baseline).
GG="$WORK/gguf"; mkdir -p "$GG"
python convert_hf_to_gguf.py "$MODEL" --outfile "$GG/qwen-0.5b-f16.gguf" --outtype f16

# The bit-width ladder.
Q=build/bin/llama-quantize
"$Q" "$GG/qwen-0.5b-f16.gguf" "$GG/qwen-0.5b-q8_0.gguf"   Q8_0
"$Q" "$GG/qwen-0.5b-f16.gguf" "$GG/qwen-0.5b-q4_k_m.gguf" Q4_K_M
"$Q" "$GG/qwen-0.5b-f16.gguf" "$GG/qwen-0.5b-q4_0.gguf"   Q4_0

ls -lh "$GG"
echo "Done. Now run: python src/quantize/log_gguf_sizes.py"
