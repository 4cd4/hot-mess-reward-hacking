"""
plot_results.py — Generate 4 publication-ready visualizations from experiment results.

Visual style modeled on Anthropic research publications: white backgrounds,
minimal spines (left + bottom only), muted but distinct palette, clean
sans-serif typography, restrained use of color, data-forward aesthetic.

Outputs to results/figures/:
  1. scatter_lift_vs_entropy.png  — Hint lift vs entropy change scatter
  2. hint_rankings_easy.png       — Hint ranking bar charts (easy questions)
  3. hint_rankings_medium.png     — Hint ranking bar charts (medium questions)
  4. hint_rankings_hard.png       — Hint ranking bar charts (hard questions)
  5. difficulty_curves.png        — Hint lift by difficulty level
  6. entropy_heatmaps.png         — Entropy change heatmap per model

Usage:
    python scripts/plot_results.py
"""

import json
import math
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import MODEL_ORDER, accepted_model_names

# ── Anthropic-style palette ──────────────────────────────────────────────────

# Muted, professional palette inspired by Anthropic research figures
# Navy-charcoal anchors with warm/cool accents
COLORS = {
    "navy":      "#1B2A4A",
    "slate":     "#4A5568",
    "warm":      "#C8553D",
    "teal":      "#2A9D8F",
    "amber":     "#E9C46A",
    "dust":      "#8B7355",
    "lavender":  "#7C6C8F",
    "sage":      "#6B8F71",
    "coral":     "#E07A5F",
    "steel":     "#718096",
    "bg":        "#FFFFFF",
    "grid":      "#E2E8F0",
    "text":      "#1A202C",
    "subtext":   "#4A5568",
    "border":    "#CBD5E0",
    "light_bg":  "#F7FAFC",
}

MODEL_COLORS = {
    "deepseek-v3": COLORS["navy"],
    "gpt-4o-mini": COLORS["coral"],
    "haiku-4.5":   COLORS["teal"],
}
MODEL_LABELS = {
    "deepseek-v3": "DeepSeek V3",
    "gpt-4o-mini": "GPT-4o-mini",
    "haiku-4.5":   "Haiku 4.5",
}
DIFF_MARKERS = {"Easy": "o", "Medium": "s", "Hard": "D"}
DIFF_SIZES = {"Easy": 55, "Medium": 55, "Hard": 45}

HINT_COLORS = {
    "metadata":    COLORS["navy"],
    "sycophancy":  COLORS["teal"],
    "anchoring":   COLORS["coral"],
    "unethical":   COLORS["lavender"],
    "grader":      COLORS["amber"],
    "consistency": COLORS["steel"],
}
HINT_LABELS = {
    "metadata":    "Metadata",
    "sycophancy":  "Sycophancy",
    "anchoring":   "Anchoring",
    "unethical":   "Unethical",
    "grader":      "Grader",
    "consistency": "Consistency",
}

MODELS = MODEL_ORDER
HINT_TYPES = ["metadata", "sycophancy", "anchoring", "unethical", "grader", "consistency"]
DIFFICULTIES = ["Easy", "Medium", "Hard"]

LOG4 = math.log(4)


# ── Global style ─────────────────────────────────────────────────────────────

def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.linewidth": 0.6,
        "axes.edgecolor": COLORS["border"],
        "axes.labelcolor": COLORS["text"],
        "axes.titlesize": 12,
        "axes.titleweight": "medium",
        "axes.titlepad": 14,
        "axes.labelsize": 10.5,
        "axes.labelpad": 8,
        "axes.grid": False,
        "xtick.color": COLORS["subtext"],
        "ytick.color": COLORS["subtext"],
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.major.pad": 6,
        "ytick.major.pad": 6,
        "legend.frameon": True,
        "legend.framealpha": 1.0,
        "legend.edgecolor": COLORS["grid"],
        "legend.fontsize": 9.5,
        "legend.borderpad": 0.6,
        "legend.handletextpad": 0.5,
        "figure.facecolor": COLORS["bg"],
        "savefig.dpi": 300,
        "savefig.facecolor": COLORS["bg"],
    })


def _add_hgrid(ax, alpha=0.4):
    """Add subtle horizontal gridlines only."""
    ax.yaxis.grid(True, color=COLORS["grid"], linewidth=0.5, alpha=alpha)
    ax.set_axisbelow(True)


def _style_legend(legend):
    """Style a legend to match the overall aesthetic."""
    frame = legend.get_frame()
    frame.set_linewidth(0.5)
    frame.set_edgecolor(COLORS["grid"])


# ── Data loading and metrics ─────────────────────────────────────────────────

def load_all_data(results_dir):
    data = {}
    for model in MODELS:
        bp = None
        hp = None
        for candidate in accepted_model_names(model):
            candidate_bp = results_dir / "phase1" / f"baseline_{candidate}.json"
            candidate_hp = results_dir / "phase2" / f"hinted_{candidate}.json"
            if candidate_bp.exists() and candidate_hp.exists():
                bp = candidate_bp
                hp = candidate_hp
                break

        if bp is None or hp is None:
            print(f"  WARNING: Missing data for {model}, skipping")
            continue
        with open(bp) as f:
            baseline = json.load(f)
        with open(hp) as f:
            hinted = json.load(f)
        data[model] = {"baseline": baseline, "hinted": hinted}
    return data


def norm_entropy(dist, n_valid):
    if n_valid == 0:
        return 0.0
    probs = [dist.get(l, 0) / n_valid for l in ["A", "B", "C", "D"]]
    probs = [p for p in probs if p > 0]
    if not probs:
        return 0.0
    return -sum(p * math.log(p) for p in probs) / LOG4


def classify_difficulty(correct_rate):
    if correct_rate >= 0.8:
        return "Easy"
    elif correct_rate <= 0.2:
        return "Hard"
    else:
        return "Medium"


def compute_all_metrics(data):
    records = []
    for model, model_data in data.items():
        base_lookup = {r["question_idx"]: r for r in model_data["baseline"]}
        for hint_type, hint_questions in model_data["hinted"].items():
            for hq in hint_questions:
                qidx = hq["question_idx"]
                bq = base_lookup.get(qidx)
                if not bq:
                    continue
                hint_letter = bq["hint_answer"]
                base_pick = bq["answer_distribution"].get(hint_letter, 0) / max(bq["n_valid"], 1)
                hint_lift = hq["hint_follow_rate"] - base_pick
                base_ent = norm_entropy(bq["answer_distribution"], bq["n_valid"])
                hint_ent = norm_entropy(hq["answer_distribution"], hq["n_valid"])
                ent_change = base_ent - hint_ent
                records.append({
                    "model": model,
                    "hint_type": hint_type,
                    "question_idx": qidx,
                    "difficulty": classify_difficulty(bq["correct_rate"]),
                    "hint_lift": hint_lift,
                    "entropy_change": ent_change,
                })
    return records


def aggregate_metrics(records, group_keys):
    groups = defaultdict(lambda: {"lifts": [], "ent_changes": []})
    for r in records:
        key = tuple(r[k] for k in group_keys)
        groups[key]["lifts"].append(r["hint_lift"])
        groups[key]["ent_changes"].append(r["entropy_change"])
    result = {}
    for key, vals in groups.items():
        result[key] = {
            "mean_hint_lift": np.mean(vals["lifts"]),
            "mean_entropy_change": np.mean(vals["ent_changes"]),
            "count": len(vals["lifts"]),
        }
    return result


# ── Plot 1: Scatter ──────────────────────────────────────────────────────────

def plot_scatter_lift_vs_entropy(records, save_path):
    agg = aggregate_metrics(records, ["model", "hint_type", "difficulty"])

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(COLORS["bg"])

    # Light background fill
    ax.set_facecolor(COLORS["light_bg"])
    _add_hgrid(ax, alpha=0.6)
    ax.xaxis.grid(True, color=COLORS["grid"], linewidth=0.5, alpha=0.6)

    # Reference line at y=0
    ax.axhline(0, color=COLORS["slate"], linestyle="-", linewidth=0.7, alpha=0.4, zorder=1)

    for key, vals in agg.items():
        model, hint_type, diff = key
        ax.scatter(
            vals["mean_hint_lift"],
            vals["mean_entropy_change"],
            color=MODEL_COLORS[model],
            marker=DIFF_MARKERS[diff],
            s=DIFF_SIZES[diff],
            alpha=0.82,
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )

    # Composite legend
    model_handles = [
        mlines.Line2D([], [], color=MODEL_COLORS[m], marker="o", linestyle="None",
                       markersize=7, markeredgecolor="white", markeredgewidth=0.6,
                       label=MODEL_LABELS[m])
        for m in MODELS if m in {r["model"] for r in records}
    ]
    diff_handles = [
        mlines.Line2D([], [], color=COLORS["slate"], marker=DIFF_MARKERS[d],
                       linestyle="None", markersize=7, label=d)
        for d in DIFFICULTIES
    ]
    legend = ax.legend(
        handles=model_handles + diff_handles,
        loc="upper left", borderaxespad=0.8,
    )
    _style_legend(legend)

    ax.set_xlabel("Mean Hint Lift")
    ax.set_ylabel("Mean Entropy Change")
    ax.set_title("Hint Lift vs. Entropy Change")

    # Subtle axis annotations
    ax.annotate("focusing", xy=(0.98, 0.97), xycoords="axes fraction",
                fontsize=8, color=COLORS["subtext"], ha="right", va="top", style="italic")
    ax.annotate("scattering", xy=(0.98, 0.03), xycoords="axes fraction",
                fontsize=8, color=COLORS["subtext"], ha="right", va="bottom", style="italic")

    fig.tight_layout(pad=1.5)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


# ── Plot 2: Bar charts ──────────────────────────────────────────────────────

def _plot_hint_rankings_for_difficulty(records, difficulty, save_path):
    """Plot hint ranking bar charts for a single difficulty level."""
    filtered = [r for r in records if r["difficulty"] == difficulty]
    agg = aggregate_metrics(filtered, ["model", "hint_type"])
    models_present = [m for m in MODELS if m in {r["model"] for r in filtered}]

    fig, axes = plt.subplots(1, len(models_present),
                             figsize=(4.8 * len(models_present), 4.5),
                             sharey=False)
    if len(models_present) == 1:
        axes = [axes]

    for i, model in enumerate(models_present):
        ax = axes[i]
        ax.set_facecolor(COLORS["light_bg"])

        hints_lifts = []
        for ht in HINT_TYPES:
            key = (model, ht)
            if key in agg:
                hints_lifts.append((ht, agg[key]["mean_hint_lift"]))

        hints_lifts.sort(key=lambda x: x[1])
        names = [h[0] for h in hints_lifts]
        vals = [h[1] for h in hints_lifts]

        base_color = MODEL_COLORS[model]
        y_pos = np.arange(len(names))

        ax.barh(y_pos, vals, color=base_color, alpha=0.75,
                edgecolor="white", linewidth=0.8, height=0.65)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([HINT_LABELS[n] for n in names], fontsize=9.5)
        ax.set_xlabel("Mean Hint Lift", fontsize=10)
        ax.set_title(MODEL_LABELS[model], fontsize=11.5, fontweight="medium",
                     color=MODEL_COLORS[model])

        # Value labels
        max_val = max(abs(v) for v in vals) if vals else 0.5
        for j, v in enumerate(vals):
            if v >= 0:
                ax.text(v + max_val * 0.02, j, f"{v:+.1%}",
                        va="center", fontsize=8.5, color=COLORS["subtext"])
            else:
                ax.text(max_val * 0.02, j, f"{v:+.1%}",
                        va="center", fontsize=8.5, color=COLORS["subtext"])

        ax.xaxis.grid(True, color=COLORS["grid"], linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

    fig.suptitle(f"Hint Vulnerability Ranking — {difficulty} Questions",
                 fontsize=13, fontweight="medium", color=COLORS["text"], y=1.01)
    fig.tight_layout(pad=1.2)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


def plot_hint_rankings(records, figures_dir):
    """Generate hint ranking bar charts for all three difficulty levels."""
    for diff in DIFFICULTIES:
        filename = f"hint_rankings_{diff.lower()}.png"
        _plot_hint_rankings_for_difficulty(records, diff, figures_dir / filename)


# ── Plot 3: Difficulty curves ────────────────────────────────────────────────

def plot_difficulty_curves(records, save_path):
    agg = aggregate_metrics(records, ["model", "difficulty"])
    models_present = [m for m in MODELS if m in {r["model"] for r in records}]

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["light_bg"])
    _add_hgrid(ax, alpha=0.6)

    for model in models_present:
        lifts = []
        for diff in DIFFICULTIES:
            key = (model, diff)
            lifts.append(agg[key]["mean_hint_lift"] if key in agg else 0)

        ax.plot(DIFFICULTIES, lifts,
                marker="o", linewidth=2.2, markersize=8,
                color=MODEL_COLORS[model], label=MODEL_LABELS[model],
                markeredgecolor="white", markeredgewidth=1.2, zorder=3)

        # Data point labels — offset to avoid overlaps
        for j, (d, v) in enumerate(zip(DIFFICULTIES, lifts)):
            # Default: label above
            offset_y, offset_x, ha = 9, 0, "center"
            # Push GPT-4o-mini labels below at Medium (overlaps Haiku)
            if model == "gpt-4o-mini" and d == "Medium":
                offset_y = -13
            # Stagger at Hard where V3 and GPT overlap
            if d == "Hard" and model == "deepseek-v3":
                offset_y = -13
            ax.annotate(f"{v:.1%}", (d, v),
                        textcoords="offset points", xytext=(offset_x, offset_y),
                        fontsize=8.5, color=MODEL_COLORS[model],
                        ha=ha, fontweight="medium")

    ax.set_xlabel("Question Difficulty")
    ax.set_ylabel("Mean Hint Lift")
    ax.set_title("Hint Lift by Question Difficulty")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

    legend = ax.legend(loc="upper left", borderaxespad=0.8)
    _style_legend(legend)

    fig.tight_layout(pad=1.5)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


# ── Plot 4: Entropy heatmaps ────────────────────────────────────────────────

def plot_entropy_heatmaps(records, save_path):
    agg = aggregate_metrics(records, ["model", "hint_type", "difficulty"])
    models_present = [m for m in MODELS if m in {r["model"] for r in records}]

    # Find symmetric color range
    all_vals = [v["mean_entropy_change"] for v in agg.values()]
    vmax = max(abs(v) for v in all_vals) if all_vals else 0.5

    # Custom diverging colormap: warm red ← white → cool blue
    cmap = LinearSegmentedColormap.from_list("anthropic_div", [
        "#C8553D",  # warm red (scattering)
        "#E8D5CC",  # light warm
        "#FFFFFF",  # white (neutral)
        "#C5DDE8",  # light cool
        "#1B2A4A",  # deep navy (focusing)
    ])

    fig, axes = plt.subplots(1, len(models_present),
                             figsize=(5 * len(models_present) + 1.2, 4.5),
                             gridspec_kw={"wspace": 0.08})
    if len(models_present) == 1:
        axes = [axes]

    im = None
    for i, model in enumerate(models_present):
        ax = axes[i]
        matrix = np.zeros((len(HINT_TYPES), len(DIFFICULTIES)))
        for hi, ht in enumerate(HINT_TYPES):
            for di, diff in enumerate(DIFFICULTIES):
                key = (model, ht, diff)
                matrix[hi, di] = agg[key]["mean_entropy_change"] if key in agg else 0

        im = ax.imshow(matrix, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

        # Cell annotations
        for hi in range(len(HINT_TYPES)):
            for di in range(len(DIFFICULTIES)):
                val = matrix[hi, di]
                # Use dark text on light cells, light text on dark cells
                intensity = abs(val) / vmax if vmax > 0 else 0
                color = "white" if intensity > 0.55 else COLORS["text"]
                ax.text(di, hi, f"{val:+.2f}", ha="center", va="center",
                        fontsize=9, color=color, fontweight="medium")

        ax.set_xticks(range(len(DIFFICULTIES)))
        ax.set_xticklabels(DIFFICULTIES, fontsize=9.5)
        ax.set_yticks(range(len(HINT_TYPES)))
        ax.set_yticklabels(
            [HINT_LABELS[ht] for ht in HINT_TYPES] if i == 0 else [],
            fontsize=9.5
        )
        ax.set_title(MODEL_LABELS[model], fontsize=11.5, fontweight="medium",
                     color=MODEL_COLORS[model], pad=10)

        # Remove all spines for heatmap
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)

    # Colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.75, pad=0.03, aspect=25)
    cbar.set_label("Entropy Change", fontsize=9.5, color=COLORS["subtext"])
    cbar.ax.tick_params(labelsize=8.5, length=0, colors=COLORS["subtext"])
    cbar.outline.set_linewidth(0.5)
    cbar.outline.set_edgecolor(COLORS["border"])

    # Annotations on colorbar
    cbar.ax.text(1.5, 0.97, "focusing", transform=cbar.ax.transAxes,
                 fontsize=7.5, color=COLORS["subtext"], va="top", style="italic")
    cbar.ax.text(1.5, 0.03, "scattering", transform=cbar.ax.transAxes,
                 fontsize=7.5, color=COLORS["subtext"], va="bottom", style="italic")

    fig.suptitle("Entropy Change by Hint Type and Difficulty",
                 fontsize=13, fontweight="medium", color=COLORS["text"], y=1.02)

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {save_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    setup_style()

    results_dir = Path("results")
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    data = load_all_data(results_dir)
    if not data:
        print("ERROR: No model data found in results/")
        return

    print("Computing metrics...")
    records = compute_all_metrics(data)
    print(f"  {len(records)} records across {len(data)} models")

    print("\nGenerating plots...")
    plot_scatter_lift_vs_entropy(records, figures_dir / "scatter_lift_vs_entropy.png")
    plot_hint_rankings(records, figures_dir)
    plot_difficulty_curves(records, figures_dir / "difficulty_curves.png")
    plot_entropy_heatmaps(records, figures_dir / "entropy_heatmaps.png")

    print("\nDone! All figures saved to results/figures/")


if __name__ == "__main__":
    main()
