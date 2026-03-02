"""
Comprehensive plot generation for the VQC distribution loading paper.

Produces all publication-quality figures:
  1. Distribution gallery — theoretical shapes of all 6 targets
  2. Distribution difficulty ranking — fidelity bar chart with error bars
  3. Depth scaling curves — per-distribution fidelity vs depth
  4. Architecture comparison heatmap
  5. Optimizer comparison — COBYLA vs SPSA with budget caveat
  6. Fidelity vs distribution complexity (entropy correlation)
  7. Convergence curves — iteration-cost history
  8. Distribution comparison panels — generated vs theoretical (per distribution)
  9. Architecture Advantage: Symmetric vs Asymmetric Distributions (COBYLA, all depths, mean over seeds and qubit counts)

ASSESSMENT REPORT FIXES applied:
  - ks_pvalue is NEVER shown in any figure (Issue 3)
  - Binomial is labelled 'basis-sparse', not 'multimodal' (Issue 1)
  - SPSA plots carry iteration-budget caveat annotation (Issue 2)
  - Primary metrics: fidelity, jensen_shannon, hellinger_distance (Issue 3)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy.stats import binom, poisson
import warnings

warnings.filterwarnings("ignore")

# Project paths
REPO = Path(__file__).parent.parent
RESULTS_DIR = REPO / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ─── Style ────────────────────────────────────────────────────────────────────
PALETTE = {
    "linear":      "#2166AC",
    "circular":    "#D6604D",
    "alternating": "#1A9641",
    "COBYLA":      "#4393C3",
    "SPSA":        "#F4A582",
}

DIST_COLORS = {
    "binomial":    "#5E3C99",
    "uniform":     "#1A9641",
    "poisson_1.5": "#4393C3",
    "poisson_2.5": "#2166AC",
    "geometric":   "#D6604D",
    "bimodal":     "#E08214",
}

DIST_LABELS = {
    "binomial":    "Binomial\n(basis-sparse)",
    "uniform":     "Uniform\n(max-entropy)",
    "poisson_1.5": r"Poisson ($\lambda$=1.5)" + "\n(mod. asym.)",
    "poisson_2.5": r"Poisson ($\lambda$=2.5)" + "\n(strong asym.)",
    "geometric":   "Geometric\n(monotone)",
    "bimodal":     "Bimodal\n(Gaussian mix.)",
}

DIST_LABELS_SHORT = {
    "binomial":    "Binomial*",
    "uniform":     "Uniform",
    "poisson_1.5": "Poisson 1.5",
    "poisson_2.5": "Poisson 2.5",
    "geometric":   "Geometric",
    "bimodal":     "Bimodal",
}

DIFFICULTY_ORDER = ["binomial", "uniform", "poisson_1.5", "poisson_2.5",
                    "geometric", "bimodal"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ─── Distribution generators (standalone, no qiskit dependency) ───────────────

def _normalize(probs, eps=1e-8):
    s = probs + eps
    return s / s.sum()


def gen_distribution(name: str, n_qubits: int = 4) -> np.ndarray:
    N = 2 ** n_qubits
    if name == "binomial":
        # FIX: n_trials = N-1 so Binomial(N-1, 0.5) spans ALL N basis states.
        # Old code used n_trials=n_qubits, which only filled the first (n_qubits+1)
        # of the N=2^n states — leaving 50-69% of states at zero mass (wrong).
        k = np.arange(N)
        p = binom.pmf(k, N - 1, 0.5)
        p[-1] += max(0.0, 1.0 - p.sum())
        return _normalize(p)
    elif name == "uniform":
        return _normalize(np.ones(N) / N)
    elif name == "poisson_1.5":
        k = np.arange(N)
        p = poisson.pmf(k, 1.5)
        p[-1] += max(0.0, 1.0 - p.sum())
        return _normalize(p)
    elif name == "poisson_2.5":
        k = np.arange(N)
        p = poisson.pmf(k, 2.5)
        p[-1] += max(0.0, 1.0 - p.sum())
        return _normalize(p)
    elif name == "geometric":
        k = np.arange(1, N + 1)
        p = (0.5) ** (k - 1) * 0.5
        p[-1] += max(0.0, 1.0 - p.sum())
        return _normalize(p)
    elif name == "bimodal":
        k = np.arange(N, dtype=float)
        mu1, mu2 = N // 4, 3 * N // 4
        p = np.exp(-((k - mu1)**2) / (2 * 0.8**2)) + \
            np.exp(-((k - mu2)**2) / (2 * 0.8**2))
        return _normalize(p)
    else:
        raise ValueError(f"Unknown distribution: {name}")


# ─── Figure 1: Distribution Gallery ──────────────────────────────────────────

def plot_distribution_gallery(n_qubits: int = 4, save: bool = True) -> Path:
    """Plot all 6 target distributions in a 2×3 grid."""
    N = 2 ** n_qubits
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    axes = axes.flatten()

    for ax, name in zip(axes, DIFFICULTY_ORDER):
        probs = gen_distribution(name, n_qubits)
        ax.bar(range(N), probs,
               color=DIST_COLORS[name], alpha=0.82, edgecolor="white",
               linewidth=0.5)
        ax.set_title(DIST_LABELS_SHORT[name], color=DIST_COLORS[name])
        ax.set_xlabel("Basis state $|k\\rangle$", fontsize=9)
        ax.set_ylabel("Probability", fontsize=9)
        ax.set_ylim(0, probs.max() * 1.3)
        ax.set_xticks(range(0, N, max(1, N // 4)))

    plt.suptitle(f"Target Probability Distributions ({N} outcomes, $n={n_qubits}$ qubits)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = FIGURES_DIR / "fig1_distribution_gallery.png"
    if save:
        plt.savefig(path)
        print(f"  ✓ Saved {path.name}")
    plt.close()
    return path


# ─── Figure 2: Difficulty Ranking ────────────────────────────────────────────

def plot_distribution_difficulty(df: pd.DataFrame, save: bool = True) -> Path:
    """Bar chart: best fidelity per distribution ordered by difficulty."""
    # Best config per distribution (all architectures, depth 4, COBYLA)
    cobyla = df[(df["optimizer"] == "COBYLA") & (df["depth"] == 4)]
    grouped = cobyla.groupby(["distribution", "circuit"]).agg(
        fid_mean=("fidelity_mean", "mean"),
        fid_std=("fidelity_std", "mean"),
    ).reset_index()

    circuits = ["linear", "circular", "alternating"]
    n_dist = len(DIFFICULTY_ORDER)
    x = np.arange(n_dist)
    width = 0.26

    fig, ax = plt.subplots(figsize=(12, 5.5))

    for i, circ in enumerate(circuits):
        subset = grouped[grouped["circuit"] == circ].set_index("distribution")
        means = np.array([subset.loc[d, "fid_mean"] if d in subset.index else 0
                          for d in DIFFICULTY_ORDER])
        stds = np.array([subset.loc[d, "fid_std"] if d in subset.index else 0
                         for d in DIFFICULTY_ORDER])
        bars = ax.bar(x + i * width, means, width,
                      label=circ.capitalize(),
                      color=PALETTE[circ], alpha=0.85,
                      yerr=stds, capsize=4, error_kw={"linewidth": 1.2})

    ax.set_xticks(x + width)
    ax.set_xticklabels(
        [DIST_LABELS[d].replace("\n", " ") for d in DIFFICULTY_ORDER],
        rotation=28, ha="right", fontsize=9
    )
    ax.set_ylabel("Mean Fidelity $F$", fontsize=11)
    ax.set_title("VQC Distribution Loading Fidelity by Distribution Type\n"
                 "(COBYLA, depth 4, averaged over seeds and qubit counts)",
                 fontsize=12)
    ax.set_ylim(0, 1.08)
    ax.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.4, label="Perfect fidelity")
    ax.legend(loc="lower right", framealpha=0.9)

    # Annotate binomial with sparsity note
    binom_idx = DIFFICULTY_ORDER.index("binomial")
    ax.annotate("*Basis-sparse:\n50–69% states\nsuppressed",
                xy=(binom_idx + width, 0.70),
                xytext=(binom_idx + 1.6 * width, 0.55),
                fontsize=7.5, color="#5E3C99",
                arrowprops=dict(arrowstyle="->", color="#5E3C99", lw=1.0))

    plt.tight_layout()
    path = FIGURES_DIR / "fig2_distribution_difficulty.png"
    if save:
        plt.savefig(path)
        print(f"  ✓ Saved {path.name}")
    plt.close()
    return path


# ─── Figure 3: Depth Scaling (one file per distribution) ─────────────────────

def plot_depth_scaling_individual(df: pd.DataFrame, save: bool = True) -> list:
    """One depth-scaling plot per distribution (separate files)."""
    cobyla_df = df[df["optimizer"] == "COBYLA"]
    saved_paths = []

    for dist in DIFFICULTY_ORDER:
        dist_data = cobyla_df[cobyla_df["distribution"] == dist]

        fig, ax = plt.subplots(figsize=(7, 4.5))

        for circ in ["linear", "circular", "alternating"]:
            cdata = dist_data[dist_data["circuit"] == circ].sort_values("depth")
            if cdata.empty:
                continue
            ax.errorbar(
                cdata["depth"], cdata["fidelity_mean"],
                yerr=cdata["fidelity_std"],
                label=circ.capitalize(),
                marker="o", lw=2, markersize=7,
                color=PALETTE[circ], alpha=0.9, capsize=4
            )

        ax.set_xlabel("Circuit Depth (layers)", fontsize=11)
        ax.set_ylabel("Mean Fidelity $F$", fontsize=11)
        label = DIST_LABELS_SHORT[dist]
        ax.set_title(f"Depth Scaling — {label}\n(COBYLA, mean ± std over 10 seeds)",
                     fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.set_xticks([1, 2, 3, 4])
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.25)

        plt.tight_layout()
        path = FIGURES_DIR / f"fig3_depth_scaling_{dist}.png"
        if save:
            plt.savefig(path)
            print(f"  ✓ Saved {path.name}")
        plt.close()
        saved_paths.append(path)

    return saved_paths


def plot_depth_scaling_combined(df: pd.DataFrame, save: bool = True) -> Path:
    """2×3 grid depth scaling for all distributions."""
    cobyla_df = df[df["optimizer"] == "COBYLA"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for ax, dist in zip(axes, DIFFICULTY_ORDER):
        dist_data = cobyla_df[cobyla_df["distribution"] == dist]
        for circ in ["linear", "circular", "alternating"]:
            cdata = dist_data[dist_data["circuit"] == circ].sort_values("depth")
            if cdata.empty:
                continue
            ax.errorbar(
                cdata["depth"], cdata["fidelity_mean"],
                yerr=cdata["fidelity_std"],
                label=circ.capitalize(),
                marker="o", lw=2, markersize=6,
                color=PALETTE[circ], alpha=0.9, capsize=3
            )
        ax.set_title(DIST_LABELS_SHORT[dist], fontsize=11,
                     color=DIST_COLORS.get(dist, "black"))
        ax.set_xlabel("Depth", fontsize=9)
        ax.set_ylabel("Fidelity", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_xticks([1, 2, 3, 4])
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.2)

    plt.suptitle("Fidelity vs Circuit Depth by Distribution (COBYLA)", y=1.01,
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = FIGURES_DIR / "fig3_depth_scaling_combined.png"
    if save:
        plt.savefig(path)
        print(f"  ✓ Saved {path.name}")
    plt.close()
    return path


# ─── Figure 4: Architecture Heatmap ──────────────────────────────────────────

def plot_architecture_heatmap(df: pd.DataFrame, save: bool = True) -> Path:
    """Heatmap: architecture × distribution mean fidelity."""
    cobyla_d4 = df[(df["optimizer"] == "COBYLA") & (df["depth"] == 4)]
    agg = cobyla_d4.groupby(["distribution", "circuit"])["fidelity_mean"].mean().reset_index()
    pivot = agg.pivot(index="circuit", columns="distribution", values="fidelity_mean")
    pivot = pivot.reindex(columns=DIFFICULTY_ORDER)

    fig, ax = plt.subplots(figsize=(12, 3.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0.6, vmax=1.0)

    ax.set_xticks(range(len(DIFFICULTY_ORDER)))
    ax.set_xticklabels([DIST_LABELS_SHORT[d] for d in DIFFICULTY_ORDER], rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([c.capitalize() for c in pivot.index])

    for i in range(len(pivot.index)):
        for j in range(len(DIFFICULTY_ORDER)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if val < 0.8 else "black")

    plt.colorbar(im, ax=ax, label="Mean Fidelity $F$", shrink=0.85)
    ax.set_title("Circuit Architecture × Distribution Fidelity Heatmap\n"
                 "(COBYLA, depth 4, mean over seeds and qubit counts)",
                 fontsize=12)
    plt.tight_layout()
    path = FIGURES_DIR / "fig4_architecture_heatmap.png"
    if save:
        plt.savefig(path)
        print(f"  ✓ Saved {path.name}")
    plt.close()
    return path


# ─── Figure 5: Optimizer Comparison ──────────────────────────────────────────

def plot_optimizer_comparison(df: pd.DataFrame, save: bool = True) -> Path:
    """
    COBYLA vs SPSA comparison.

    ASSESSMENT REPORT FIX (Issue 2):
    SPSA hit the 500-iteration limit on every single run — it had NOT converged.
    This must be clearly annotated on the figure. The comparison is
    iteration-budget-limited, not a convergence comparison.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Mean fidelity by optimizer and distribution
    opt_dist = df.groupby(["optimizer", "distribution"])["fidelity_mean"].mean().reset_index()
    cobyla_data = opt_dist[opt_dist["optimizer"] == "COBYLA"].set_index("distribution")
    spsa_data = opt_dist[opt_dist["optimizer"] == "SPSA"].set_index("distribution")

    x = np.arange(len(DIFFICULTY_ORDER))
    width = 0.38

    ax = axes[0]
    c_vals = [cobyla_data.loc[d, "fidelity_mean"] if d in cobyla_data.index else np.nan
              for d in DIFFICULTY_ORDER]
    s_vals = [spsa_data.loc[d, "fidelity_mean"] if d in spsa_data.index else np.nan
              for d in DIFFICULTY_ORDER]

    ax.bar(x - width / 2, c_vals, width, label="COBYLA", color=PALETTE["COBYLA"], alpha=0.85)
    ax.bar(x + width / 2, s_vals, width, label="SPSA†", color=PALETTE["SPSA"], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([DIST_LABELS_SHORT[d] for d in DIFFICULTY_ORDER],
                       rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean Fidelity $F$", fontsize=11)
    ax.set_title("Fidelity by Optimizer and Distribution", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.text(0.02, 0.04, "†SPSA: 500-iteration budget\n  limit reached on all runs\n  (not converged)",
            transform=ax.transAxes, fontsize=7.5, color="#C0392B",
            bbox=dict(boxstyle="round,pad=0.3", fc="#FDECEA", alpha=0.85))

    # Panel B: Mean iterations
    iter_data = df.groupby("optimizer")["iterations_mean"].mean()
    ax2 = axes[1]
    colors_opt = [PALETTE["COBYLA"], PALETTE["SPSA"]]
    bars = ax2.bar(iter_data.index, iter_data.values, color=colors_opt, alpha=0.85, width=0.4)
    for bar, val in zip(bars, iter_data.values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 f"{val:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Mean Iterations", fontsize=11)
    ax2.set_title("Iterations to Termination\n(COBYLA: converged; SPSA: budget-limited)",
                  fontsize=11)
    ax2.set_ylim(0, 580)
    ax2.axhline(500, color="#C0392B", ls="--", lw=1.5, alpha=0.7, label="500-iter budget")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.25)

    plt.suptitle("COBYLA vs SPSA Optimizer Comparison\n"
                 "(⚠ SPSA results reflect a 500-iteration resource budget, not convergence)",
                 fontsize=12, fontweight="bold", color="#2C2C2C")
    plt.tight_layout()
    path = FIGURES_DIR / "fig5_optimizer_comparison.png"
    if save:
        plt.savefig(path)
        print(f"  ✓ Saved {path.name}")
    plt.close()
    return path


# ─── Figure 6: Entropy–Fidelity Correlation ──────────────────────────────────

def plot_entropy_fidelity(df: pd.DataFrame, n_qubits: int = 4, save: bool = True) -> Path:
    """Scatter: target distribution entropy vs achieved fidelity."""
    # Compute entropy for each distribution
    entropy_map = {}
    skew_map = {}
    for name in DIFFICULTY_ORDER:
        p = gen_distribution(name, n_qubits)
        p_safe = np.clip(p, 1e-12, 1)
        entropy_map[name] = float(-np.sum(p_safe * np.log(p_safe)))
        k = np.arange(len(p))
        mu = np.sum(k * p)
        sig = np.sqrt(np.sum((k - mu)**2 * p))
        skew_map[name] = float(np.sum(((k - mu) / (sig + 1e-10))**3 * p))

    cobyla_d4 = df[(df["optimizer"] == "COBYLA") & (df["depth"] == 4)]
    mean_fid = cobyla_d4.groupby("distribution")["fidelity_mean"].mean()

    entropies = [entropy_map[d] for d in DIFFICULTY_ORDER if d in mean_fid.index]
    fidelities = [mean_fid[d] for d in DIFFICULTY_ORDER if d in mean_fid.index]
    dists_present = [d for d in DIFFICULTY_ORDER if d in mean_fid.index]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for d, e, f in zip(dists_present, entropies, fidelities):
        ax.scatter(e, f, color=DIST_COLORS[d], s=120, zorder=5,
                   label=DIST_LABELS_SHORT[d], edgecolors="white", lw=1.5)
    # Trend line
    if len(entropies) >= 3:
        z = np.polyfit(entropies, fidelities, 1)
        xr = np.linspace(min(entropies) - 0.1, max(entropies) + 0.1, 100)
        ax.plot(xr, np.polyval(z, xr), "k--", lw=1.2, alpha=0.5, label="Linear fit")
        corr = np.corrcoef(entropies, fidelities)[0, 1]
        ax.text(0.04, 0.94, f"r = {corr:.3f}", transform=ax.transAxes,
                fontsize=10, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))
    ax.set_xlabel("Shannon Entropy $H(p)$ of Target", fontsize=11)
    ax.set_ylabel("Mean Fidelity $F$ (depth 4, COBYLA)", fontsize=11)
    ax.set_title("Entropy vs Achieved Fidelity", fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.2)

    ax2 = axes[1]
    skews = [abs(skew_map[d]) for d in dists_present]
    for d, s, f in zip(dists_present, skews, fidelities):
        ax2.scatter(s, f, color=DIST_COLORS[d], s=120, zorder=5,
                    label=DIST_LABELS_SHORT[d], edgecolors="white", lw=1.5)
    if len(skews) >= 3:
        z2 = np.polyfit(skews, fidelities, 1)
        xr2 = np.linspace(0, max(skews) + 0.2, 100)
        ax2.plot(xr2, np.polyval(z2, xr2), "k--", lw=1.2, alpha=0.5)
        corr2 = np.corrcoef(skews, fidelities)[0, 1]
        ax2.text(0.04, 0.94, f"r = {corr2:.3f}", transform=ax2.transAxes,
                 fontsize=10, va="top",
                 bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))
    ax2.set_xlabel("|Skewness| of Target Distribution", fontsize=11)
    ax2.set_ylabel("Mean Fidelity $F$", fontsize=11)
    ax2.set_title("|Skewness| vs Achieved Fidelity", fontsize=12)
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.2)

    plt.suptitle("Distribution Shape Complexity as a Predictor of VQC Fidelity",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = FIGURES_DIR / "fig6_entropy_fidelity_correlation.png"
    if save:
        plt.savefig(path)
        print(f"  ✓ Saved {path.name}")
    plt.close()
    return path


# ─── Figure 7: Generated vs Theoretical (individual comparison panels) ───────

def plot_distribution_comparison(df: pd.DataFrame, n_qubits: int = 4,
                                 save: bool = True) -> list:
    """
    For each distribution: plot theoretical vs best-achieved generated distribution.
    Each distribution gets its OWN separate image file.
    """
    # Load raw JSON results for actual generated distributions
    raw_dir = REPO / "results" / "raw"
    saved_paths = []

    for dist in DIFFICULTY_ORDER:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
        theoretical = gen_distribution(dist, n_qubits)
        N = len(theoretical)
        states = range(N)

        for ax, circ in zip(axes, ["linear", "circular", "alternating"]):
            # Best seed for this (dist, circ, n_qubits=4, depth=4, COBYLA)
            subset = df[
                (df["distribution"] == dist) &
                (df["circuit"] == circ) &
                (df["n_qubits"] == n_qubits) &
                (df["depth"] == 4) &
                (df["optimizer"] == "COBYLA")
            ]

            ax.bar(states, theoretical, alpha=0.6, color="#2166AC",
                   label="Theoretical", width=0.4, align="edge")

            if not subset.empty:
                best_fid = subset["fidelity_mean"].max()
                # Try to find best-seed JSON
                best_found = False
                for seed in range(10):
                    pattern = (
                        f"{dist}_{circ}_d4_q{n_qubits}_COBYLA_s{seed}.json"
                    )
                    json_path = raw_dir / "full" / pattern
                    if not json_path.exists():
                        json_path = raw_dir / "demo" / pattern
                    if json_path.exists():
                        import json
                        with open(json_path) as fj:
                            res = json.load(fj)
                        if "achieved_distribution" in res:
                            gen_dist = np.array(res["achieved_distribution"])
                            if len(gen_dist) == N:
                                ax.bar([s + 0.4 for s in states], gen_dist,
                                       alpha=0.75, color=PALETTE[circ],
                                       label=f"Generated (F={best_fid:.3f})",
                                       width=0.4, align="edge")
                                best_found = True
                                break

                if not best_found:
                    # Fall back to showing theoretical only with fidelity annotation
                    ax.text(0.5, 0.5,
                            f"Best F={best_fid:.3f}\n(raw JSON not found)",
                            transform=ax.transAxes, ha="center", va="center",
                            fontsize=9, color="gray")
            else:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", fontsize=9, color="gray")

            ax.set_title(f"{circ.capitalize()}", fontsize=11)
            ax.set_xlabel("Basis state $|k\\rangle$", fontsize=9)
            ax.set_ylabel("Probability", fontsize=9)
            ax.set_ylim(0, max(theoretical) * 1.35)
            ax.legend(fontsize=7.5, loc="upper right")
            ax.grid(axis="y", alpha=0.2)

        label = DIST_LABELS_SHORT[dist]
        plt.suptitle(
            f"Generated vs Theoretical: {label}\n"
            f"(COBYLA, depth 4, $n={n_qubits}$ qubits)",
            fontsize=13, fontweight="bold"
        )
        plt.tight_layout()
        path = FIGURES_DIR / f"fig7_comparison_{dist}.png"
        if save:
            plt.savefig(path)
            print(f"  ✓ Saved {path.name}")
        plt.close()
        saved_paths.append(path)

    return saved_paths


# ─── Figure 8: Qubit Scaling (n=3 vs n=4) ────────────────────────────────────

def plot_qubit_scaling(df: pd.DataFrame, save: bool = True) -> Path:
    """Fidelity for n=3 vs n=4 per distribution."""
    cobyla_d4 = df[(df["optimizer"] == "COBYLA") & (df["depth"] == 4)]
    agg = cobyla_d4.groupby(["distribution", "n_qubits"])["fidelity_mean"].mean().reset_index()

    x = np.arange(len(DIFFICULTY_ORDER))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 5))

    for i, nq in enumerate([3, 4]):
        subset = agg[agg["n_qubits"] == nq].set_index("distribution")
        vals = np.array([subset.loc[d, "fidelity_mean"] if d in subset.index else np.nan
                         for d in DIFFICULTY_ORDER])
        ax.bar(x + i * width - width / 2, vals, width,
               label=f"$n={nq}$ qubits ({2**nq} states)",
               color=["#4393C3", "#D6604D"][i], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([DIST_LABELS_SHORT[d] for d in DIFFICULTY_ORDER],
                       rotation=28, ha="right", fontsize=9)
    ax.set_ylabel("Mean Fidelity $F$", fontsize=11)
    ax.set_title("Qubit Scaling: n=3 vs n=4\n(COBYLA, depth 4, all architectures)",
                 fontsize=12)
    ax.set_ylim(0, 1.08)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    # Annotate binomial drop
    binom_idx = DIFFICULTY_ORDER.index("binomial")
    b3 = agg[(agg["distribution"] == "binomial") & (agg["n_qubits"] == 3)]["fidelity_mean"]
    b4 = agg[(agg["distribution"] == "binomial") & (agg["n_qubits"] == 4)]["fidelity_mean"]
    if not b3.empty and not b4.empty:
        drop = float(b3.values[0]) - float(b4.values[0])
        ax.annotate(f"↓{drop:.3f} (sparsity\ngrows with n)",
                    xy=(binom_idx + 0.05, float(b4.values[0]) + 0.01),
                    xytext=(binom_idx + 0.7, float(b4.values[0]) - 0.05),
                    fontsize=7.5, color="#5E3C99",
                    arrowprops=dict(arrowstyle="->", color="#5E3C99", lw=1))

    plt.tight_layout()
    path = FIGURES_DIR / "fig8_qubit_scaling.png"
    if save:
        plt.savefig(path)
        print(f"  ✓ Saved {path.name}")
    plt.close()
    return path


# ─── Figure 9: Architecture Advantage by Distribution Type ───────────────────

def plot_architecture_advantage(df: pd.DataFrame, save: bool = True) -> Path:
    """Symmetric vs asymmetric advantage per architecture."""
    cobyla = df[df["optimizer"] == "COBYLA"]

    symmetric = ["binomial", "uniform"]
    asymmetric = ["poisson_1.5", "poisson_2.5", "geometric", "bimodal"]

    records = []
    for circ in ["linear", "circular", "alternating"]:
        cdf = cobyla[cobyla["circuit"] == circ]
        sym_fid = cdf[cdf["distribution"].isin(symmetric)]["fidelity_mean"].mean()
        asym_fid = cdf[cdf["distribution"].isin(asymmetric)]["fidelity_mean"].mean()
        records.append({"circuit": circ, "type": "Symmetric", "fidelity": sym_fid})
        records.append({"circuit": circ, "type": "Asymmetric", "fidelity": asym_fid})

    rec_df = pd.DataFrame(records)
    x = np.arange(3)
    width = 0.38

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, typ in enumerate(["Symmetric", "Asymmetric"]):
        vals = rec_df[rec_df["type"] == typ].set_index("circuit")
        fids = np.array([vals.loc[c, "fidelity"] if c in vals.index else np.nan
                         for c in ["linear", "circular", "alternating"]])
        ax.bar(x + i * width - width / 2, fids, width,
               label=typ, color=["#4393C3", "#E08214"][i], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(["Linear", "Circular", "Alternating"])
    ax.set_ylabel("Mean Fidelity $F$", fontsize=11)
    ax.set_title("Architecture Advantage: Symmetric vs Asymmetric Distributions\n"
                 "(COBYLA, all depths, mean over seeds and qubit counts)",
                 fontsize=12)
    ax.set_ylim(0, 1.02)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    path = FIGURES_DIR / "fig9_architecture_advantage.png"
    if save:
        plt.savefig(path)
        print(f"  ✓ Saved {path.name}")
    plt.close()
    return path


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("GENERATING ALL PAPER FIGURES")
    print("=" * 60)

    # Load data
    csv_path = RESULTS_DIR / "summary_full.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run aggregate_results.py first.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} configurations from summary_full.csv")
    print(f"Optimizers present: {df['optimizer'].unique()}")
    print(f"Distributions: {df['distribution'].unique()}")
    print()

    print("Generating figures...")

    # Fig 1: Distribution gallery
    plot_distribution_gallery(n_qubits=4)

    # Fig 2: Difficulty ranking
    plot_distribution_difficulty(df)

    # Fig 3: Depth scaling (combined + individual)
    plot_depth_scaling_combined(df)
    plot_depth_scaling_individual(df)

    # Fig 4: Architecture heatmap
    plot_architecture_heatmap(df)

    # Fig 5: Optimizer comparison
    plot_optimizer_comparison(df)

    # Fig 6: Entropy-fidelity correlation
    plot_entropy_fidelity(df)

    # Fig 7: Generated vs theoretical per distribution
    plot_distribution_comparison(df)

    # Fig 8: Qubit scaling
    plot_qubit_scaling(df)

    # Fig 9: Architecture advantage
    plot_architecture_advantage(df)

    print()
    print(f"✓ All figures saved to: {FIGURES_DIR}")
    print("=" * 60)
    print("PAPER METRIC NOTE (from assessment_report.docx):")
    print("  PRIMARY metrics to report: fidelity, jensen_shannon, hellinger_distance")
    print("  EXCLUDED from paper: ks_pvalue (unreliable for N=8/16; Issue 3)")
    print("  SPSA caveat: 500-iteration budget limit, not converged (Issue 2)")
    print("  Binomial label: 'basis-sparse', NOT 'multimodal' (Issue 1)")


if __name__ == "__main__":
    main()
