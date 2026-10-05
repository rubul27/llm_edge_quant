# On-device quantized LLM, profiled on real Qualcomm hardware

Quantize a small LLM across bit-widths, measure the **quality cost** of each, and profile
the **runtime cost** on a **real Snapdragon 8 Gen 3** via Qualcomm AI Hub — then argue which
operating point to ship on a phone, backed by numbers measured on actual silicon.

**Model:** Qwen2.5-0.5B-Instruct (494 M params) · **Quality/prep:** Kaggle GPU + llama.cpp ·
**On-device:** Qualcomm AI Hub cloud (Samsung Galaxy S24, Snapdragon 8 Gen 3 NPU).

![Quality vs size Pareto](results/pareto.png)

## TL;DR — the ship point, now measured on hardware

- **FP16 does not fit.** It compiled to the Hexagon NPU but **exceeded the device's memory budget** on profiling.
- **INT8 fits and runs:** the quantized transformer profiles at **~13.4 ms / 128-token prefill (~9,500 tok/s) in ~501 MB** on the real S24 NPU.
- **Q4_K_M is the quality ship point:** at **0.40 GB** it's 2.5× smaller than FP16 for only **+3.3%** perplexity, while naive 4-bit (Q4_0) is barely smaller yet degrades **4.5× more** (+14.9%). INT8 is near-lossless on quality but 33% larger than Q4_K_M.

Net: ship a 4-bit weight format for the smallest footprint and best quality-per-byte; the
on-device run proves full precision is not an option and that a sub-1 GB quantized model runs
comfortably on a phone NPU.

## Why on-device ≠ server (demonstrated, not asserted)

A GPU server has tens of GB of HBM; a phone NPU (Hexagon HTP) has a small, tightly-bounded
memory budget, runs integer math, and validates a fixed op set. All three showed up as real
results here: **FP16 exceeded NPU memory**; the model only ran after **quantizing to INT8**
(~501 MB); and on-device composition initially failed on an **unsupported `IsNaN` op** emitted
by the attention-mask code, which had to be removed (eager attention) before the Hexagon
runtime would accept the graph.

## Results (all measured)

### Quality vs cost — GGUF ladder (llama.cpp)

| Precision | Size (GB) | Eff. bits/wt | WikiText-2 PPL | Δ vs f16 | Verdict |
|-----------|-----------|--------------|----------------|----------|---------|
| f16       | 0.994     | 16.1         | 14.927         | —        | reference |
| Q8_0      | 0.531     | 8.6          | 14.981         | **+0.36%** | near-free |
| **Q4_K_M**| **0.398** | **6.4**      | **15.416**     | **+3.28%** | **ship point** |
| Q4_0      | 0.352     | 5.7          | 17.144         | +14.85%  | dominated |

Perplexity: WikiText-2, llama.cpp `-c 512 --chunks 200`, identical across rows; the f16 row is
the in-track baseline (not the HF number below).

### On-device runtime — AI Hub → Samsung Galaxy S24 (Snapdragon 8 Gen 3 NPU)

| Model | Result |
|-------|--------|
| Pipeline check (MobileNet-V2) | 0.37 ms inference, ~195 MB peak |
| Qwen-0.5B **FP16** → QNN DLC | compiled ✅ · profile **exceeded device memory** ❌ |
| Qwen-0.5B **INT8** (backbone) → QNN DLC | **13.4 ms / 128-tok prefill · ~9,500 tok/s · 501 MB peak · cold load 11.8 s / warm 0.35 s** ✅ |

### FP16 baseline (HuggingFace quality anchors)

| Metric | Value | Settings |
|--------|-------|----------|
| Params | 494.0 M | measured |
| FP16 weights | 0.99 GB | measured |
| WikiText-2 perplexity | 12.665 | window 2048 / stride 1024 |
| HellaSwag acc_norm | 0.495 | validation, n=1000, seed=0 (reproduces published ~0.49) |

## Three engineering findings

1. **Nominal ≠ effective bits.** "4-bit" Q4_K_M measured **6.4 bits/weight**, not 4.0 — Qwen-0.5B's
   token-embedding table is ~28% of the model and is kept at higher precision. Small models compress worse.
2. **Perplexity ≠ correctness.** FP16 scores a healthy 12.7 perplexity yet explains "KV cache" *wrong*,
   which is why quality uses a task benchmark (HellaSwag) + qualitative check, not perplexity alone.
3. **On-device deployment is op-gated.** The NPU rejected an `IsNaN` op (bool input, HTP validator
   error 3110) from SDPA mask handling; switching to eager attention removed it and the graph then composed.

## Method

- **Quantization:** `src/quantize/build_and_quantize.sh` builds llama.cpp and produces the
  f16/Q8_0/Q4_K_M/Q4_0 GGUF ladder; `log_gguf_sizes.py` records sizes + effective bits.
- **Quality:** `baseline.py` (FP16 perplexity + generations), `eval/hellaswag.py` (acc_norm),
  and `llama-perplexity` for the GGUF perplexity ladder.
- **On-device:** export to ONNX (eager attention, backbone), package with external weights,
  `qai-hub` INT8 quantize-job → compile `--target_runtime qnn_dlc --truncate_64bit_tensors` →
  profile on a real S24. Full journey in `src/deploy/AIHUB_NOTES.md`.
- **Trade-off:** `src/viz/pareto.py` renders `results/pareto.png` from `results/metrics.csv`.

## Reliability / honesty checks

- Quantized models compared to the **same-track FP16 baseline**; only **relative** degradation claimed
  (HF-PPL and llama.cpp-PPL never mixed).
- HellaSwag scorer is custom but **reproduces the published 0.49**, validating it for relative use
  (not leaderboard-comparable — stated).
- All on-device numbers are from real hardware; the one profiling failure (FP16 memory) is reported, not hidden.
- **Degeneracy check (INT4):** greedy generations across Q8_0 / Q4_K_M / Q4_0 showed **no repetition collapse** — distinct-4gram stayed ~0.92–0.94 for all three (a looping output would crater it), with **Q4_0 consistently lowest**, matching the perplexity ordering. Conclusion: Q4_K_M is safe; naive Q4_0 is measurably but not catastrophically worse.

## Limitations & future work

- The on-device number is **prefill** (128 tokens) through the **transformer backbone** (the final
  vocab projection / `lm_head` was dropped to isolate the backbone). Autoregressive **decode**
  tokens/sec (single-token step with KV cache) is the natural next measurement.
- On-device INT8 used a small **random-token calibration set** — fine for the latency/memory *cost*
  measured here; it is not a quality claim (quality comes from the GGUF track).
- Quantized-model HellaSwag and CPU tokens/sec are future work.
- v1 uses 0.5B; a 1.5B model would show a better compression ratio (smaller embedding fraction).

## Reproduce

```bash
python src/baseline/baseline.py            # Phase 1  FP16 perplexity + generations
python src/eval/hellaswag.py --n 1000      # Phase 1b task accuracy
bash   src/quantize/build_and_quantize.sh  # Phase 2a GGUF ladder (build + quantize)
python src/quantize/log_gguf_sizes.py      # Phase 2a sizes + effective bits
bash   src/eval/gguf_perplexity.sh         # Phase 2b perplexity per GGUF
python src/deploy/aihub_profile.py         # Phase 3  compile + profile on real Snapdragon (needs qai-hub auth)
python src/eval/degeneracy.py              # Phase 4  INT4 degeneracy check
python src/viz/pareto.py                   # Phase 5  the Pareto chart
```

## Résumé bullet

> Quantized Qwen2.5-0.5B across FP16/INT8/INT4 and profiled it on a real Snapdragon 8 Gen 3 NPU
> via Qualcomm AI Hub: INT8 runs the transformer at ~13 ms / 128-token prefill in ~0.5 GB while
> FP16 exceeded device memory, and Q4_K_M is the quality ship point (2.5× smaller, +3.3% perplexity
> vs +14.9% for naive INT4) — debugging the on-device deployment down to an unsupported `IsNaN` op
> in the attention mask.
