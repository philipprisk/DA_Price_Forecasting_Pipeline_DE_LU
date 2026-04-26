"""
plot_anc_bar.py
===============

Plots Average Normalised Coefficients (ANC) as horizontal bar charts for
two electricity-price forecasting models:

* **Fundamental Model** -- features derived from fundamental market drivers
  (wind, load, solar, lagged prices, calendar variables).
* **EXAA-Enriched Model** -- same feature set augmented by the day-ahead
  EXAA auction price.

Each model produces one PDF figure saved to the ``output/figures/`` directory
in the repository root.

Expected repository layout
--------------------------
repo/
├── output/figures/      (created automatically if missing)
└── visualization/
    └── plot_anc_bar.py
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ============================================================
#  ANC VALUES
# ============================================================

features_fund = [
    "Wind $d$",
    "Load $d$",
    "Price $d-1$",
    "Solar $d$",
    "Price $d-7$",
    "Market dummy",
    "Price $d-2$",
    "Weekday",
    "Holiday",
]
anc_fundamental = np.array([43.4, 30.9, 28.5, 7.5, 3.2, 3.2, 2.0, 1.5, 0.0])

features_exaa = [
    "EXAA $d$",
    "Price $d-1$",
    "Wind $d$",
    "Load $d$",
    "Price $d-7$",
    "Market dummy",
    "Solar $d$",
    "Price $d-2$",
    "Weekday",
    "Holiday",
]
anc_exaa = np.array([90.3, 8.0, 6.9, 4.6, 3.1, 3.0, 2.3, 1.9, 0.9, 0.0])

# ============================================================
#  STYLE
# ============================================================

plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset":  "cm",
    "font.size":         12,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.spines.left":  False,
    "axes.linewidth":    0.8,
    "grid.linewidth":    0.5,
    "grid.color":        "#dddddd",
    "grid.linestyle":    "--",
})

COLOR = "#1a5c8a"
BAR_H = 0.55


def draw_anc_bars(ax, features, anc_values, xmax):
    """
    Draw a ranked horizontal bar chart of ANC values onto an existing Axes.

    Each bar is annotated with its numeric value to the right of the bar.
    The y-axis is inverted so that the most important feature (highest ANC)
    appears at the top.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The Axes object on which the chart is drawn.
    features : list of str
        Feature labels in rank order (index 0 = most important).
        Supports LaTeX math notation (e.g. ``"Wind $d$"``).
    anc_values : numpy.ndarray
        ANC scores corresponding to each entry in *features*.  Values
        should be non-negative and listed in the same order as *features*.
    xmax : float
        Upper limit of the x-axis.  Typically set to
        ``anc_values.max() * 1.18`` to leave room for the value labels.

    Returns
    -------
    None
    """
    n = len(features)
    y = np.arange(n)

    bars = ax.barh(y, anc_values, height=BAR_H, color=COLOR, zorder=3)

    for bar, val in zip(bars, anc_values):
        ax.text(bar.get_width() + xmax * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", ha="left",
                fontsize=10.0, color=COLOR)

    ax.set_yticks(y)
    ax.set_yticklabels(features, fontsize=12)
    ax.set_xlabel("ANC", fontsize=13, labelpad=6)
    ax.set_xlim(left=0, right=xmax)
    ax.set_ylim(-0.6, n - 0.4)
    ax.tick_params(axis="x", which="major", labelsize=12)
    ax.tick_params(axis="x", which="minor", length=2)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.grid(True, which="major", axis="x", zorder=0)
    ax.invert_yaxis()


def load_anc_bars(path: Path, top_n: int | None = None) -> tuple[list[str], np.ndarray]:
    df = pd.read_csv(path)
    feature_col = "feature_group" if "feature_group" in df.columns else "feature"
    if feature_col not in df.columns or "ANC" not in df.columns:
        raise ValueError(f"{path} must contain ANC and either feature_group or feature columns.")
    work = df[[feature_col, "ANC"]].dropna().copy()
    work = work.groupby(feature_col, as_index=False)["ANC"].mean().sort_values("ANC", ascending=False)
    if top_n is not None:
        work = work.head(top_n)
    return work[feature_col].astype(str).tolist(), work["ANC"].to_numpy(dtype=float)


def save_anc_bar(features, values, output_path: Path, figsize=(5.0, 5.0)) -> None:
    xmax = float(np.nanmax(values) * 1.18) if len(values) else 1.0
    fig, ax = plt.subplots(figsize=figsize)
    draw_anc_bars(ax, features, values, xmax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot ANC bar charts.")
    parser.add_argument("--fundamental-csv", type=Path, default=None, help="ANC CSV for fundamental features.")
    parser.add_argument("--exaa-csv", type=Path, default=None, help="ANC CSV for EXAA-enriched features.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory.")
    parser.add_argument("--top-n", type=int, default=None, help="Keep only the top N features from CSV inputs.")
    return parser


# ============================================================
#  RUN
# ============================================================

if __name__ == "__main__":
    args = build_parser().parse_args()
    _HERE = Path(__file__).parent.parent  # points to repository root
    _OUT  = args.output_dir or (_HERE / "output" / "figures")
    _OUT.mkdir(parents=True, exist_ok=True)

    if args.fundamental_csv is not None:
        features, values = load_anc_bars(args.fundamental_csv, top_n=args.top_n)
        save_anc_bar(features, values, _OUT / "anc_fundamental.pdf")

    if args.exaa_csv is not None:
        features, values = load_anc_bars(args.exaa_csv, top_n=args.top_n)
        save_anc_bar(features, values, _OUT / "anc_exaa.pdf")

    if args.fundamental_csv is not None or args.exaa_csv is not None:
        raise SystemExit(0)

    # ── Figure 1: Fundamental Model ──────────────────────────────
    xmax_fund = anc_fundamental.max() * 1.18
    fig1, ax1 = plt.subplots(figsize=(5.0, 5.0))
    draw_anc_bars(ax1, features_fund, anc_fundamental, xmax_fund)
    fig1.tight_layout()
    fig1.savefig(_OUT / "anc_fundamental.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print("Saved: anc_fundamental.pdf")

    # ── Figure 2: EXAA-Enriched Model ────────────────────────────
    xmax_exaa = anc_exaa.max() * 1.18
    fig2, ax2 = plt.subplots(figsize=(5.0, 5.0))
    draw_anc_bars(ax2, features_exaa, anc_exaa, xmax_exaa)
    fig2.tight_layout()
    fig2.savefig(_OUT / "anc_exaa.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print("Saved: anc_exaa.pdf")
