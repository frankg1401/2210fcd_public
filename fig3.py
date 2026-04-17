import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

FONT = 14

plt.rcParams.update({
    "font.size": FONT,
    "axes.titlesize": FONT + 6,
    "axes.labelsize": FONT + 1,
    "xtick.labelsize": FONT,
    "ytick.labelsize": FONT,
    "legend.fontsize": FONT,
    "figure.titlesize": FONT + 2,
})

# =========================
# Load data
# =========================
df = pd.read_csv(r"D:\GIT\CSC2210_Project\2210fcd\results.csv")
df = df[df["AUC"].notna()].copy()

# =========================
# Build Config
# =========================
df["Config"] = (
    df["Fusion"].fillna("").astype(str).str.strip() + "_" +
    df["Fusion-more"].fillna("").astype(str).str.strip() + "_" +
    df["D-Type"].astype(str).str.strip() + "_" +
    df["Size"].astype(str).str.strip()
)

df["Config"] = (
    df["Config"]
    .str.replace("__", "_", regex=False)
    .str.strip("_")
    .str.replace(" ", "", regex=False)
)

best = df.copy()

# =========================
# Order
# =========================
order = [
    "single_t1_fp32_full", "single_t1_amp_full",
    "single_t1_fp32_small", "single_t1_amp_small",

    "single_flair_fp32_full", "single_flair_amp_full",
    "single_flair_fp32_small", "single_flair_amp_small",

    "early_fp32_full", "early_amp_full",
    "early_fp32_small", "early_amp_small",

    "late_opt_checkpoint_fp32_full", "late_amp_full",
    "late_fp32_small", "late_amp_small"
]

best = best[best["Config"].isin(order)].copy()
best["order"] = best["Config"].map({k: i for i, k in enumerate(order)})
best = best.sort_values("order").reset_index(drop=True)

# =========================
# Color scheme
# =========================
def get_color(cfg):
    if cfg.startswith("single_t1"):
        return "#8ecae6"
    if cfg.startswith("single_flair"):
        return "#219ebc"
    if cfg.startswith("early"):
        return "#90be6d"
    if cfg.startswith("late_opt"):
        return "#f8961e"
    if cfg.startswith("late"):
        return "#f8961e"
    return "gray"

colors = [get_color(c) for c in best["Config"]]

# =========================
# Numeric x
# =========================
x = list(range(len(best)))

# =========================
# Create figure (2 panels)
# =========================
fig, axes = plt.subplots(
    2, 1,
    figsize=(14, 10),
    sharex=True,
    gridspec_kw={"hspace": 0}  
)

# =====================================================
# TOP PANEL — Inference Latency
# =====================================================
ax = axes[0]
axes[0].set_yticks(np.arange(0, 0.16, 0.025))

bars = ax.bar(x, best["InferLatency"], color=colors, edgecolor="none")

# styling (same logic as Fig 2)
for bar, cfg, color in zip(bars, best["Config"], colors):

    if "small" in cfg:
        bar.set_hatch(".")
        bar.set_edgecolor("white")
        bar.set_linewidth(2.0)

    if "amp" in cfg:
        bar.set_edgecolor("white")
        bar.set_linewidth(5.0)

        bx = bar.get_x()
        width = bar.get_width()
        height = bar.get_height()

        ax.bar(
            bx + width / 2,
            height,
            width=width,
            fill=False,
            edgecolor=color,
            linewidth=3.2,
            align="center",
            zorder=3
        )

# annotate
for bar, val in zip(bars, best["InferLatency"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(best["InferLatency"]) * 0.02,
        f"{val:.3f}",
        ha="center",
        fontsize=FONT-1
    )

ax.set_ylabel("Inference Latency (s)")
#ax.set_title("Figure 3: Computational Efficiency and Memory Usage")
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_xlim(-0.6, len(x) - 0.4)

# group separators
for sep in [3.5, 7.5, 11.5]:
    ax.axvline(sep, linestyle="--", alpha=0.3)

axes[0].text(
    0.01,
    0.81,
    "(a) Inference Latency",
    transform=axes[0].transAxes,
    fontsize=FONT+2,
    fontweight="bold",
    va="top"
)

# =========================
# Top group titles
# =========================
group_centers = [1.5, 5.5, 9.5, 13.5]
group_names = [
    "Single Modality\n(T1)",
    "Single Modality\n(FLAIR)",
    "Early Fusion",
    "Late Fusion"
]

group_y = 0.185

axes[0].set_ylim(0, 0.19)

for xc, name in zip(group_centers, group_names):
    axes[0].text(
        xc,
        group_y,
        name,
        ha="center",
        va="top",
        fontsize=FONT+1,
        linespacing=1.2
    )

# =====================================================
# BOTTOM PANEL — MEMORY
# =====================================================
ax = axes[1]

bars = ax.bar(x, best["TrainPeakGPU"], color=colors, edgecolor="none")

# styling (same)
for bar, cfg, color in zip(bars, best["Config"], colors):

    if "small" in cfg:
        bar.set_hatch(".")
        bar.set_edgecolor("white")
        bar.set_linewidth(2.0)

    if "amp" in cfg:
        bar.set_edgecolor("white")
        bar.set_linewidth(5.0)

        bx = bar.get_x()
        width = bar.get_width()
        height = bar.get_height()

        ax.bar(
            bx + width / 2,
            height,
            width=width,
            fill=False,
            edgecolor=color,
            linewidth=3.2,
            align="center",
            zorder=3
        )

# annotate
for bar, val in zip(bars, best["TrainPeakGPU"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(best["TrainPeakGPU"]) * 0.02,
        f"{val:.1f}",
        ha="center",
        fontsize=FONT-1
    )

ax.set_ylabel("Peak Training GPU Memory (GB)")
ax.grid(axis="y", linestyle="--", alpha=0.4)

for sep in [3.5, 7.5, 11.5]:
    ax.axvline(sep, linestyle="--", alpha=0.3)

ax.text(0.01, 0.95, "(b) Peak Training Memory", transform=ax.transAxes,
        fontsize=FONT+2, fontweight="bold", va="top")

axes[1].set_ylim(0, 25)
axes[1].set_yticks(np.arange(0, 25, 5))  # 0,5,10,15,20

# =====================================================
# X-axis labels 
# =====================================================
axes[1].set_xticks(x)
axes[1].set_xticklabels([""] * len(x))
axes[1].tick_params(axis="x", length=0)

# FP32 / AMP row
top_labels = []
bottom_labels = []

for cfg in best["Config"]:
    if "late_opt_checkpoint_fp32_full" in cfg:
        top_labels.append("FP32_opt")
    elif "fp32" in cfg:
        top_labels.append("FP32")
    else:
        top_labels.append("AMP")

    if "full" in cfg:
        bottom_labels.append("Full")
    else:
        bottom_labels.append("Small")

ymin = axes[1].get_ylim()[0]

# first row
ymin = axes[1].get_ylim()[0]

for xi, lab in zip(x, top_labels):
    if lab == "FP32_opt":
        axes[1].text(
            xi,
            ymin - 1.2,
            lab,
            ha="center",
            va="top",
            fontsize=FONT,
            clip_on=False,
            bbox=dict(
                boxstyle="square",
                edgecolor="black",
                facecolor="none",
                linewidth=0.4,
                pad=0.1
            )
        )
    else:
        axes[1].text(
            xi,
            ymin - 1.2, 
            lab,
            ha="center",
            va="top",
            fontsize=FONT,
            clip_on=False
        )

# second row (pair labels)
pair_centers = [0.5, 2.5, 4.5, 6.5, 8.5, 10.5, 12.5, 14.5]
pair_labels = ["Full Size", "Small Size"] * 4

for xc, lab in zip(pair_centers, pair_labels):
    axes[1].text(
        xc,
        ymin - 2.6,  
        lab,
        ha="center",
        va="top",
        fontsize=FONT,
        clip_on=False
    )

# horizontal lines
for start in range(0, len(x), 2):
    axes[1].plot(
        [start - 0.3, start + 1 + 0.3],
        [ymin - 2.3, ymin - 2.3],
        color="black",
        linewidth=1.0,
        clip_on=False
    )


# =========================
# Save
# =========================
plt.tight_layout(rect=[0, 0.12, 1, 0.95])
plt.subplots_adjust(hspace=0)

# ===== SAVE FIGURE =====
import os

save_dir = r"D:\GIT\CSC2210_Project\2210fcd\figures"
os.makedirs(save_dir, exist_ok=True)

base_name = "figure3" 

png_path = os.path.join(save_dir, f"{base_name}.png")
pdf_path = os.path.join(save_dir, f"{base_name}.pdf")

plt.tight_layout()
plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
plt.show()