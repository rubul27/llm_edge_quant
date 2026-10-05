"""Phase 5 — quality-vs-cost Pareto for the GGUF bit-width ladder.

Reads results/metrics.csv and plots WikiText-2 perplexity (quality, lower=better)
against on-disk size (cost). Highlights Q4_K_M as the recommended ship point and
annotates the FP16 on-device memory ceiling found via AI Hub.

Run (from repo root):  python src/viz/pareto.py
"""
import pandas as pd, matplotlib.pyplot as plt

df = pd.read_csv("results/metrics.csv")
size = df[df.metric == "gguf_size_gb"].set_index("precision")["value"].astype(float)
ppl  = df[df.metric == "gguf_wikitext2_ppl"].set_index("precision")["value"].astype(float)
order = ["f16", "q8_0", "q4_k_m", "q4_0"]
labels = {"f16": "f16", "q8_0": "Q8_0", "q4_k_m": "Q4_K_M", "q4_0": "Q4_0"}
colors = {"f16": "#6b7280", "q8_0": "#2563eb", "q4_k_m": "#059669", "q4_0": "#dc2626"}

fig, ax = plt.subplots(figsize=(7.5, 5))
base = ppl["f16"]
for p in order:
    x, y = size[p], ppl[p]
    ax.scatter(x, y, s=170, color=colors[p], zorder=3, edgecolor="white", linewidth=1.5)
    deg = (y - base) / base * 100
    tag = f"{labels[p]}\n{x:.2f} GB" + ("" if p == "f16" else f", +{deg:.1f}%")
    ax.annotate(tag, (x, y), xytext=(8, 8), textcoords="offset points",
                fontsize=10, fontweight="bold" if p == "q4_k_m" else "normal")

# ship-point marker
ax.annotate("ship point", (size["q4_k_m"], ppl["q4_k_m"]), xytext=(size["q4_k_m"] + 0.12, ppl["q4_k_m"] + 0.6),
            fontsize=10, color=colors["q4_k_m"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=colors["q4_k_m"]))

ax.set_xlabel("Model size on disk (GB)  —  lower = cheaper")
ax.set_ylabel("WikiText-2 perplexity  —  lower = better")
ax.set_title("Quality vs size — Qwen2.5-0.5B GGUF ladder\n(lower-left is better)", fontweight="bold")
ax.grid(True, alpha=0.3)
ax.margins(0.18)
fig.text(0.5, -0.02,
         "On the Snapdragon 8 Gen 3 NPU: FP16 exceeded device memory; INT8 ran at ~13 ms/128-tok "
         "prefill in ~501 MB. Quantization is required to deploy.",
         ha="center", fontsize=8.5, style="italic", color="#6b7280")
fig.tight_layout()
fig.savefig("results/pareto.png", dpi=150, bbox_inches="tight")
print("wrote results/pareto.png")
