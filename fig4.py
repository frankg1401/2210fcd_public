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
# Order / keep same configs as Fig 3
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
# Color scheme (match Fig 2/3)
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

# =========================
# Pretty short labels
# =========================
def short_label(cfg):
    if cfg == "single_t1_fp32_full":
        return "T1 FP32 Full"
    if cfg == "single_t1_amp_full":
        return "T1 AMP Full"
    if cfg == "single_t1_fp32_small":
        return "T1 FP32 Small"
    if cfg == "single_t1_amp_small":
        return "T1 AMP Small"

    if cfg == "single_flair_fp32_full":
        return "FLAIR FP32 Full"
    if cfg == "single_flair_amp_full":
        return "FLAIR AMP Full"
    if cfg == "single_flair_fp32_small":
        return "FLAIR FP32 Small"
    if cfg == "single_flair_amp_small":
        return "FLAIR AMP Small"

    if cfg == "early_fp32_full":
        return "Early FP32 Full"
    if cfg == "early_amp_full":
        return "Early AMP Full"
    if cfg == "early_fp32_small":
        return "Early FP32 Small"
    if cfg == "early_amp_small":
        return "Early AMP Small"

    if cfg == "late_opt_checkpoint_fp32_full":
        return "Late Optimized\nFP32 Full"
    if cfg == "late_amp_full":
        return "Late AMP Full"
    if cfg == "late_fp32_small":
        return "Late FP32 Small"
    if cfg == "late_amp_small":
        return "Late AMP Small"

    return cfg

# =========================
# Marker style
# small -> dotted feel via hollow marker
# amp -> double-ring effect
# =========================
def marker_for(cfg):
    if "small" in cfg:
        return "o"
    return "s"

# =========================
# Plot
# =========================
fig, ax = plt.subplots(figsize=(11, 8))

for _, row in best.iterrows():
    cfg = row["Config"]
    x = row["InferLatency"]
    y = row["AUC"]
    color = get_color(cfg)
    marker = marker_for(cfg)

    # base point
    if "small" in cfg:
        ax.scatter(
            x, y,
            s=220,
            marker=marker,
            facecolors="white",
            edgecolors=color,
            linewidths=2.5,
            zorder=3
        )
        # inner smaller point
        ax.scatter(
            x, y,
            s=45,
            marker=marker,
            c=color,
            edgecolors="none",
            zorder=4
        )
    else:
        ax.scatter(
            x, y,
            s=220,
            marker=marker,
            c=color,
            edgecolors="none",
            zorder=3
        )

    # AMP -> double border effect
    if "amp" in cfg:
        ax.scatter(
            x, y,
            s=300,
            marker=marker,
            facecolors="none",
            edgecolors="white",
            linewidths=4.5,
            zorder=4
        )
        ax.scatter(
            x, y,
            s=210,
            marker=marker,
            facecolors="none",
            edgecolors=color,
            linewidths=2.2,
            zorder=5
        )

    # annotation
    if cfg == "late_opt_checkpoint_fp32_full":
        # place BELOW the point
        ax.text(
            x,
            y - 0.01,                 # shift downward
            short_label(cfg),
            fontsize=FONT,
            ha="center",
            va="top"
        )
    else:
        # place to the RIGHT of the point
        ax.text(
            x + 0.003,               # shift right
            y,
            short_label(cfg),
            fontsize=FONT,
            ha="left",
            va="center"
        )

# =========================
# Group legend text (manual, to match style)
# =========================
ax.text(0.02, 0.98, "Color:", transform=ax.transAxes, va="top", fontsize=FONT+1, fontweight="bold")

ax.text(0.02, 0.94, "Single Modality (T1)",
        transform=ax.transAxes, va="top", fontsize=FONT, color="#8ecae6")

ax.text(0.02, 0.90, "Single Modality (FLAIR)",
        transform=ax.transAxes, va="top", fontsize=FONT, color="#219ebc")

ax.text(0.02, 0.86, "Early Fusion",
        transform=ax.transAxes, va="top", fontsize=FONT, color="#90be6d")

ax.text(0.02, 0.82, "Late Fusion / Late Opt",
        transform=ax.transAxes, va="top", fontsize=FONT, color="#f8961e")

ax.text(0.52, 0.98, "Style:", transform=ax.transAxes, va="top", fontsize=FONT+1, fontweight="bold")
ax.text(0.52, 0.94, "Solid shape = FP32", transform=ax.transAxes, va="top", fontsize=FONT)
ax.text(0.52, 0.90, "Double outline = AMP", transform=ax.transAxes, va="top", fontsize=FONT)
ax.text(0.52, 0.86, "Circle = Small Size Input", transform=ax.transAxes, va="top", fontsize=FONT)
ax.text(0.52, 0.82, "Square = Full Size Input", transform=ax.transAxes, va="top", fontsize=FONT)


# =========================
# Axes / formatting
# =========================
ax.set_xlabel("Inference Latency (s)")
ax.set_ylabel("Test AUC")
#ax.set_title("Figure 4: Performance–Efficiency Trade-off")
ax.grid(True, linestyle="--", alpha=0.4)

# tight but readable limits
x_min = best["InferLatency"].min()
x_max = best["InferLatency"].max()
y_min = best["AUC"].min()
y_max = best["AUC"].max()

ax.set_xlim(max(0, x_min - 0.012), x_max + 0.018)
ax.set_ylim(y_min - 0.012, min(1.0, y_max + 0.05))

# ===== SAVE FIGURE =====
import os

save_dir = r"D:\GIT\CSC2210_Project\2210fcd\figures"
os.makedirs(save_dir, exist_ok=True)

base_name = "figure4" 

png_path = os.path.join(save_dir, f"{base_name}.png")
pdf_path = os.path.join(save_dir, f"{base_name}.pdf")

plt.tight_layout()
plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
plt.show()