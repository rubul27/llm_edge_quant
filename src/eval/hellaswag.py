"""Phase 1b  — HellaSwag acc_norm for the HF (FP16) model.

Custom length-normalized log-likelihood scorer over the 4 endings.
Internally consistent across bit-widths (valid for measuring degradation);
NOT directly comparable to lm-eval-harness leaderboard numbers.
Reproduces the published ~0.49 for Qwen2.5-0.5B, which validates the scorer.

Run (from repo root):  python src/eval/hellaswag.py --n 1000
"""
import os, csv, glob, argparse, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
RESULTS = "results/metrics.csv"
_hits = glob.glob("/kaggle/input/qwen25-05b-instruct/**/config.json", recursive=True)
MODEL_PATH = os.path.dirname(_hits[0]) if _hits else MODEL_NAME
if _hits:
    os.environ["HF_HUB_OFFLINE"] = "1"


def log_metric(precision, metric, value, note=""):
    os.makedirs("results", exist_ok=True)
    new = not os.path.exists(RESULTS)
    with open(RESULTS, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["model", "precision", "metric", "value", "note"])
        w.writerow([MODEL_NAME, precision, metric, value, note])


@torch.no_grad()
def loglik(tok, model, device, ctx, ending):
    ctx_ids  = tok(ctx, return_tensors="pt").input_ids
    full_ids = tok(ctx + " " + ending, return_tensors="pt").input_ids.to(device)
    c = ctx_ids.shape[1]                                  # context length in tokens
    logits = model(full_ids).logits                       # [1, T, V]
    logp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    tgt  = full_ids[:, 1:]
    tok_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0]
    ending_logp = tok_logp[c - 1:]                         # ending tokens only
    return ending_logp.sum().item(), ending_logp.numel()


@torch.no_grad()
def evaluate(precision="fp16", n=1000, seed=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float16).to(device).eval()
    ds = load_dataset("Rowan/hellaswag", split="validation").shuffle(seed=seed).select(range(n))
    correct = 0
    for ex in ds:
        norm = [loglik(tok, model, device, ex["ctx"], e) for e in ex["endings"]]
        scores = [s / max(k, 1) for s, k in norm]          # length-normalized = acc_norm
        pred = max(range(4), key=lambda i: scores[i])
        correct += int(pred == int(ex["label"]))
    acc = correct / len(ds)
    print(f"HellaSwag acc_norm ({precision}, n={n}, seed={seed}): {acc:.4f}")
    log_metric(precision, "hellaswag_acc_norm", round(acc, 4), f"val_n{n}_seed{seed}")
    return acc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--precision", default="fp16")
    evaluate(**vars(ap.parse_args()))
