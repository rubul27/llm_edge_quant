"""Phase 2a — record real GGUF file sizes and EFFECTIVE bits/weight.

The point: Q4_K_M is a mixed K-quant, so its bits/weight prints ABOVE 4.0
(some tensors kept at Q6_K). This is the measured "nominal vs effective INT4"
fact you quote instead of the theoretical 0.25 GB / 4.0 bits.

Run (from repo root):  python src/quantize/log_gguf_sizes.py
"""
import os, csv, argparse

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
RESULTS = "results/metrics.csv"
N_PARAMS = 494_000_000          # measured in Phase 1 (baseline.py print)

FILES = {
    "f16":    "qwen-0.5b-f16.gguf",
    "q8_0":   "qwen-0.5b-q8_0.gguf",
    "q4_k_m": "qwen-0.5b-q4_k_m.gguf",
    "q4_0":   "qwen-0.5b-q4_0.gguf",
}


def log_metric(precision, metric, value, note=""):
    os.makedirs("results", exist_ok=True)
    new = not os.path.exists(RESULTS)
    with open(RESULTS, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["model", "precision", "metric", "value", "note"])
        w.writerow([MODEL_NAME, precision, metric, value, note])


def main(gguf_dir):
    print(f"{'prec':8s}{'size_GB':>10s}{'bits/wt':>10s}")
    for prec, fn in FILES.items():
        path = os.path.join(gguf_dir, fn)
        if not os.path.exists(path):
            print(f"{prec:8s}   (missing: {fn})"); continue
        b = os.path.getsize(path); gb = b / 1e9; bpw = b * 8 / N_PARAMS
        print(f"{prec:8s}{gb:>10.3f}{bpw:>10.2f}")
        log_metric(prec, "gguf_size_gb", round(gb, 3), fn)
        log_metric(prec, "gguf_bits_per_weight", round(bpw, 2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf_dir", default="/kaggle/working/gguf")
    main(ap.parse_args().gguf_dir)
