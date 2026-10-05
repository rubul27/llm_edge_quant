"""Phase 1  — FP16 baseline for Qwen2.5-0.5B-Instruct.

Measures the reference numbers every quantized model is later compared against:
  - parameter count + FP16 weight size
  - WikiText-2 perplexity (fluency anchor)
  - 3 qualitative generations (saved to results/generations_fp16.md)

Loads from a local Kaggle dataset snapshot if one is attached, else downloads
from Hugging Face. Appends measured numbers to results/metrics.csv.

Run (from repo root):  python src/baseline/baseline.py
"""
import os, csv, glob, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"          # canonical name -> logged to metrics.csv
RESULTS    = "results/metrics.csv"

# Prefer a local snapshot dataset; fall back to HF download.
_hits = glob.glob("/kaggle/input/qwen25-05b-instruct/**/config.json", recursive=True)
MODEL_PATH = os.path.dirname(_hits[0]) if _hits else MODEL_NAME
if _hits:
    os.environ["HF_HUB_OFFLINE"] = "1"
print(f"loading from: {MODEL_PATH}")


def log_metric(precision, metric, value, note=""):
    os.makedirs("results", exist_ok=True)
    new = not os.path.exists(RESULTS)
    with open(RESULTS, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["model", "precision", "metric", "value", "note"])
        w.writerow([MODEL_NAME, precision, metric, value, note])   # log NAME, not path


def load(device):
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float16).to(device).eval()
    return tok, model


@torch.no_grad()
def qualitative(tok, model, device):
    prompts = [
        "Explain what a KV cache is in one paragraph.",
        "Give me three tips for reducing LLM inference latency on a phone.",
        "Write a short poem about quantization.",
    ]
    os.makedirs("results", exist_ok=True)
    with open("results/generations_fp16.md", "w") as fout:
        for p in prompts:
            inputs = tok.apply_chat_template(
                [{"role": "user", "content": p}],
                add_generation_prompt=True, return_tensors="pt", return_dict=True,
            ).to(device)
            gen = model.generate(**inputs, max_new_tokens=200, do_sample=False)  # greedy = reproducible
            text = tok.decode(gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            print(f"\n=== {p}\n{text}")
            fout.write(f"### {p}\n\n{text}\n\n")


@torch.no_grad()
def perplexity(tok, model, device, max_length=2048, stride=1024):
    ds  = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids.to(device)
    seq_len, nlls, prev = ids.size(1), [], 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        trg = end - prev
        chunk = ids[:, begin:end]
        target = chunk.clone(); target[:, :-trg] = -100   # score only the new tokens
        loss = model(chunk, labels=target).loss
        nlls.append(loss.float() * trg); prev = end
        if end == seq_len: break
    return torch.exp(torch.stack(nlls).sum() / end).item()


def main():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device)
    tok, model = load(device)

    n = sum(p.numel() for p in model.parameters())
    print(f"params: {n/1e6:.1f} M  ->  FP16 weights ~ {n*2/1e9:.2f} GB")
    log_metric("fp16", "params_millions", round(n / 1e6, 1))
    log_metric("fp16", "weights_gb_fp16", round(n * 2 / 1e9, 3))

    qualitative(tok, model, device)
    ppl = perplexity(tok, model, device)
    print(f"\nWikiText-2 perplexity (fp16, win2048/stride1024): {ppl:.3f}")
    log_metric("fp16", "wikitext2_ppl", round(ppl, 3), "win2048_stride1024")


if __name__ == "__main__":
    main()
