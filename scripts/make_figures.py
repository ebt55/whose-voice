"""Generate the paper's figures from committed CSVs.

Two figures only. The result is a ladder and a robustness table; more plots would dilute
rather than clarify.

Usage:  .venv\\Scripts\\python.exe scripts/make_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

FIG = REPO / "figures"
FIG.mkdir(exist_ok=True)

INK = "#1a1a1a"
ACCENT = "#B03A2E"
MUTED = "#7f8c8d"
CHANCE = "#95a5a6"


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_color(INK)


def fig_ladder():
    """The affordance ladder: attribution vs how well the detector specifies the attack."""
    labels = ["D0\nattacker's exact\nprompt", "D1T\ntype-aware\ntemplate", "D1\ngeneric\ntemplate"]
    k5 = [60.0, 20.0, 20.0]
    k47 = [np.nan, 0.0, 0.0]

    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
    x = np.arange(len(labels))
    w = 0.36
    ax.bar(x - w / 2, k5, w, label="K = 5 candidates", color=ACCENT)
    ax.bar(x + w / 2, [0 if np.isnan(v) else v for v in k47], w,
           label="K = 47 candidates", color=MUTED)
    ax.axhline(20, ls="--", lw=1.2, color=CHANCE)
    ax.text(2.42, 21.5, "chance, K=5", fontsize=8, color=CHANCE, ha="right")
    ax.axhline(2.13, ls=":", lw=1.2, color=CHANCE)
    ax.text(2.42, 3.4, "chance, K=47", fontsize=8, color=CHANCE, ha="right")
    ax.text(0 + w / 2, 2, "n/a", ha="center", fontsize=8, color=MUTED)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("strict top-1 accuracy (%)")
    ax.set_ylim(0, 72)
    ax.set_title("Attribution collapses with hypothesis fidelity, not candidate-set size",
                 fontsize=10.5, color=INK, pad=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_affordance_ladder.png", bbox_inches="tight")
    print(f"  wrote {(FIG / 'fig1_affordance_ladder.png').relative_to(REPO)}")


def fig_defences():
    """D0 attribution across the data-level defences the attack was built to beat."""
    path = REPO / "results" / "defence_summary.csv"
    if not path.exists():
        print("  (no defence_summary.csv, skipping fig 2)")
        return
    df = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(7.0, 3.6), dpi=200)
    y = np.arange(len(df))[::-1]
    pct = df["strict_top1"] * 100
    colours = [ACCENT if "word-frequency" not in d else MUTED for d in df["defence"]]
    ax.barh(y, pct, color=colours, height=0.62)
    ax.axvline(20, ls="--", lw=1.2, color=CHANCE)
    ax.text(21, y[0] + 0.45, "chance", fontsize=8, color=CHANCE, va="bottom")

    for yi, (p, h) in zip(y, zip(pct, df["hits"])):
        ax.text(p + 1.5, yi, f"{h}", va="center", fontsize=8.5, color=INK)

    ax.set_yticks(y)
    ax.set_yticklabels(df["defence"], fontsize=9)
    ax.set_xlabel("strict top-1 accuracy (%), oracle prompt, K = 5")
    ax.set_xlim(0, 100)
    ax.set_title("Data-level defences do not block attribution\n"
                 "(grey = the only family that degrades it)",
                 fontsize=10.5, color=INK, pad=10)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_defences.png", bbox_inches="tight")
    print(f"  wrote {(FIG / 'fig2_defences.png').relative_to(REPO)}")


def fig_dilution():
    """The headline negative: signal multiplier over chance vs poison density."""
    path = REPO / "results" / "embed_dilution.csv"
    if not path.exists():
        print("  (no embed_dilution.csv, skipping fig 3)")
        return
    df = pd.read_csv(path)
    df = df[(df["mode"] == "uniform") & (df["agg"] == "mean")]

    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=200)
    chance = 1 / 47
    for enc, colour, marker in (("mpnet", ACCENT, "o"), ("e5", "#2C6E9B", "s")):
        s = df[df["encoder"] == enc].sort_values("density")
        if s.empty:
            continue
        ax.plot(s["density"] * 100, s["boot_mean"] / chance, marker=marker,
                color=colour, lw=2, ms=5, label=enc)
    ax.axhline(1.0, ls="--", lw=1.2, color=CHANCE)
    ax.text(3.3, 1.15, "chance", fontsize=8, color=CHANCE)
    # The band real attacks occupy (Lamerton & Roger train at 3.125-12.5%).
    ax.axvspan(3.125, 12.5, color="#B03A2E", alpha=0.07)
    ax.text(6.2, 16, "densities real\nattacks use", fontsize=8, color=ACCENT,
            ha="center", va="top")

    ax.set_xscale("log")
    ax.set_xticks([3.125, 6.25, 12.5, 25, 50, 100])
    ax.set_xticklabels(["3.1", "6.3", "12.5", "25", "50", "100"])
    ax.set_xlabel("poison density (% of rows), log scale")
    ax.set_ylabel("signal, as multiple of chance")
    ax.set_title("Attribution collapses at realistic poison density\n"
                 "K = 47, generic descriptor, symmetric bootstrap",
                 fontsize=10.5, color=INK, pad=10)
    ax.legend(frameon=False, fontsize=8.5)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_dilution_collapse.png", bbox_inches="tight")
    print(f"  wrote {(FIG / 'fig3_dilution_collapse.png').relative_to(REPO)}")


def fig_replication():
    """Five encoders x two generators: the effect is real on every axis, magnitude is not."""
    path = REPO / "results" / "embed_replication.csv"
    if not path.exists():
        print("  (no embed_replication.csv, skipping fig 4)")
        return
    df = pd.read_csv(path)
    df = df[df["mode"] == "descriptor"]
    order = ["MiniLM-L6 (22M)", "mpnet-base (110M)", "bge-base (110M)",
             "e5-base (110M)", "bge-large (335M)"]
    df = df.set_index("encoder").reindex([o for o in order if o in set(df["encoder"])])

    cg = REPO / "results" / "embed_crossgen.csv"
    gpt = {}
    if cg.exists():
        c = pd.read_csv(cg)
        c = c[c["setting"] == "gpt41 only"]
        gpt = {"mpnet-base (110M)": float(c[c["encoder"] == "mpnet"]["boot_mean"].iloc[0]),
               "e5-base (110M)": float(c[c["encoder"] == "e5"]["boot_mean"].iloc[0])}

    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=200)
    x = np.arange(len(df))
    w = 0.38
    ax.bar(x - w / 2, df["boot_mean"] * 100, w, color=ACCENT, label="Gemma-generated")
    have = [gpt.get(e, np.nan) for e in df.index]
    ax.bar(x + w / 2, [0 if np.isnan(v) else v * 100 for v in have], w,
           color=MUTED, label="GPT-4.1-generated")
    for xi, v in zip(x, have):
        if np.isnan(v):
            ax.text(xi + w / 2, 1.5, "n/r", ha="center", fontsize=7.5, color=MUTED)

    ax.axhline(100 / 47, ls="--", lw=1.2, color=CHANCE)
    ax.text(len(df) - 0.45, 100 / 47 + 1.2, "chance (2.1%)", fontsize=8,
            color=CHANCE, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([e.replace(" (", "\n(") for e in df.index], fontsize=8.5)
    ax.set_ylabel("mean bootstrap top-1 (%)")
    ax.set_title("The effect replicates on every axis tested; its magnitude does not",
                 fontsize=10.5, color=INK, pad=10)
    ax.legend(frameon=False, fontsize=8.5)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_replication.png", bbox_inches="tight")
    print(f"  wrote {(FIG / 'fig4_replication.png').relative_to(REPO)}")


if __name__ == "__main__":
    fig_ladder()
    fig_defences()
    fig_dilution()
    fig_replication()
