"""Generate labeled placeholder images for the README screenshot section.

These are *placeholders* so the README renders nicely before you capture real
screenshots. To replace them, run the app, open each page, screenshot it, and
save it over the matching PNG in this folder (keep the same file names).

    python app/assets/screenshots/make_placeholders.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parent

# (filename, page title, emoji)
PAGES = [
    ("match_predictor.png", "Match Predictor", "⚽"),
    ("team_comparison.png", "Team Comparison", "\U0001f4ca"),
    ("scoreline_heatmap.png", "Scoreline Heatmap", "\U0001f525"),
    ("tournament_simulator.png", "Tournament Simulator", "\U0001f3c6"),
    ("backtesting_dashboard.png", "Backtesting Dashboard", "\U0001f4c8"),
    ("worldcup_2026_mode.png", "2026 World Cup Mode", "\U0001f30e"),
]

BG = "#0e1117"        # Streamlit dark background
ACCENT = "#FF4B4B"    # Streamlit red
TEXT = "#fafafa"
MUTED = "#9aa0a6"


def make_placeholder(filename: str, title: str) -> None:
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG)
    ax.axis("off")

    ax.text(0.5, 0.62, "World Cup Match Predictor", ha="center", va="center",
            color=ACCENT, fontsize=30, fontweight="bold")
    ax.text(0.5, 0.50, title, ha="center", va="center", color=TEXT, fontsize=44,
            fontweight="bold")
    ax.text(0.5, 0.38, "screenshot placeholder — replace with a real screenshot",
            ha="center", va="center", color=MUTED, fontsize=18, style="italic")
    # Simple framed border.
    ax.add_patch(plt.Rectangle((0.03, 0.05), 0.94, 0.90, fill=False,
                               edgecolor="#2b2f3a", linewidth=2))
    fig.savefig(OUT_DIR / filename, facecolor=BG)
    plt.close(fig)


def main() -> None:
    for filename, title, _emoji in PAGES:
        make_placeholder(filename, title)
        print(f"wrote {OUT_DIR / filename}")


if __name__ == "__main__":
    main()
