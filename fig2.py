import pandas as pd
import matplotlib.pyplot as plt

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
# Numeric x positions
# =========================
x = list(range(len(best)))

# =========================
# Plot
# =========================
fig, ax = plt.subplots(figsize=(14, 7))
bars = ax.bar(x, best["AUC"], color=colors, edgecolor="none")

# =========================
# Style encoding
# =========================
for bar, cfg, color in zip(bars, best["Config"], colors):

    # SMALL → white dots
    if "small" in cfg:
        bar.set_hatch(".")
        bar.set_edgecolor("white")
        bar.set_linewidth(2.0)

    # AMP → double border
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

# =========================
# Annotate AUC
# =========================
for bar, auc in zip(bars, best["AUC"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f"{auc:.3f}",
        ha="center",
        fontsize=FONT-1
    )

# =========================
# Main axis formatting
# =========================
ax.set_ylabel("Test AUC")
#ax.set_title("Figure 2: Test AUC Across Model Configurations")
ax.set_ylim(0.5, 1.0)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_xlim(-0.6, len(x) - 0.4)

# remove default x tick labels
ax.set_xticks(x)
ax.set_xticklabels([""] * len(x))
ax.tick_params(axis="x", length=0)

# =========================
# Group separators
# =========================
for sep in [3.5, 7.5, 11.5]:
    ax.axvline(sep, linestyle="--", alpha=0.3)

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

y_top = ax.get_ylim()[1]
for xc, name in zip(group_centers, group_names):
    ax.text(
        xc,
        y_top - 0.02,
        name,
        ha="center",
        va="top",
        fontsize=FONT,
        linespacing=1.2
    )

# =========================
# Custom 2-line bottom labels
# =========================
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

ymin = ax.get_ylim()[0]

# first row: FP32 / AMP / FP32 / AMP
for xi, lab in zip(x, top_labels):

    if lab == "FP32_opt":
        ax.text(
            xi,
            ymin - 0.025,
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
        ax.text(
            xi,
            ymin - 0.025,
            lab,
            ha="center",
            va="top",
            fontsize=FONT,
            clip_on=False
        )

# second row: Full / Small (centered over each pair)
pair_centers = [0.5, 2.5, 4.5, 6.5, 8.5, 10.5, 12.5, 14.5]
pair_labels = ["Full Size", "Small Size"] * 4

for xc, lab in zip(pair_centers, pair_labels):
    ax.text(
        xc,
        ymin - 0.065,
        lab,
        ha="center",
        va="top",
        fontsize=FONT,
        clip_on=False
    )

# =========================
# Horizontal bars under labels
# =========
for start in range(0, len(x), 2):
    ax.plot(
        [start - 0.3, start + 1 + 0.3],
        [ymin - 0.05, ymin - 0.05],   # was too low before
        color="black",
        linewidth=1.0,
        clip_on=False
    )

# =========================
# Save
# =========================
plt.tight_layout(rect=[0, 0.12, 1, 0.95])



# ===== SAVE FIGURE =====
import os

save_dir = r"D:\GIT\CSC2210_Project\2210fcd\figures"
os.makedirs(save_dir, exist_ok=True)

base_name = "figure2" 

png_path = os.path.join(save_dir, f"{base_name}.png")
pdf_path = os.path.join(save_dir, f"{base_name}.pdf")

plt.tight_layout()
plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
plt.show()