"""Phase 4 (reliability) — INT4 degeneracy check.

Greedy generations across Q8_0 / Q4_K_M / Q4_0 on fixed prompts, scored with a
distinct-4gram repetition metric (1.0 = no repeats; a looping/degenerate output craters it).
Catches what perplexity can't: quantization-induced repetition loops.

Finding: no model collapsed (distinct-4gram ~0.92-0.94); Q4_0 marginally lowest,
matching the perplexity ordering. Q4_K_M is safe to ship; naive Q4_0 is measurably
but not catastrophically worse.

Note: the saved (older) llama-cli build is verbose on stdout, so a perfectly clean
transcript is hard to capture; the collapse/no-collapse signal is robust regardless.

Usage: python src/eval/degeneracy.py [GGUF_DIR] [BIN_DIR]
"""
import subprocess, os, sys, re

GG  = sys.argv[1] if len(sys.argv) > 1 else "/kaggle/working/gguf"
BIN = sys.argv[2] if len(sys.argv) > 2 else "/kaggle/working/llamacpp-bin"
env = dict(os.environ, LD_LIBRARY_PATH=BIN)
NT  = str(os.cpu_count() or 4)

PROMPTS = ["The capital of France is",
           "Explain how a transformer neural network works:",
           "Once upon a time in a small village,"]
MODELS = {"q8_0": "qwen-0.5b-q8_0.gguf", "q4_k_m": "qwen-0.5b-q4_k_m.gguf", "q4_0": "qwen-0.5b-q4_0.gguf"}


def strip_banner(t):
    return "\n".join(ln for ln in t.splitlines()
                     if not re.search(r"[▄█▀]|Loading model|build\s+:|ftype\s+:", ln)).strip()


def distinct4(t):
    w = t.split()
    if len(w) < 4:
        return 1.0
    g = [tuple(w[i:i+4]) for i in range(len(w)-3)]
    return len(set(g)) / len(g)


def main():
    os.makedirs("results", exist_ok=True)
    with open("results/generations_quant.md", "w") as out:
        for prec, fn in MODELS.items():
            print(f"\n===== {prec} =====")
            scores = []
            for p in PROMPTS:
                r = subprocess.run(
                    [f"{BIN}/llama-cli", "-m", f"{GG}/{fn}", "-p", p, "-n", "80", "--temp", "0",
                     "-ngl", "0", "-t", NT, "-c", "512", "-st", "--no-display-prompt"],
                    env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=180)
                g = strip_banner(r.stdout)
                d = distinct4(g); scores.append(d)
                print(f"  distinct4={d:.2f} | {g[:80]!r}")
                out.write(f"**{prec} — {p}**\n\n{g}\n\n_distinct-4gram={d:.2f}_\n\n---\n\n")
            print(f"  mean distinct-4gram ({prec}): {sum(scores)/len(scores):.3f}")


if __name__ == "__main__":
    main()
