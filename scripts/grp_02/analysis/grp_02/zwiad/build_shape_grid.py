"""Build a grid mosaic of all PCA projections to visually categorize manifold shapes."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import math

from wisent.core.constants import VIZ_DPI

SHAPES_DIR = Path("zwiad_results/figures/shapes")
OUTPUT_DIR = Path("zwiad_results/figures")
CATEGORIES = [
    "coding", "commonsense", "math", "safety", "reasoning",
    "knowledge", "science", "multilingual", "hallucination",
]


def get_cat(name):
    for c in CATEGORIES:
        if name.startswith(c):
            return c
    return "other"


def load_images():
    by_cat = {}
    for f in sorted(SHAPES_DIR.glob("*_pca.png")):
        name = f.stem.replace("_pca", "")
        cat = get_cat(name)
        by_cat.setdefault(cat, []).append((name, f))
    return by_cat


def build_category_grid(cat, items, output_path):
    n = len(items)
    cols = min(6, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]
    for i, (name, fpath) in enumerate(items):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        try:
            img = mpimg.imread(str(fpath))
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, "ERR", ha="center", va="center", transform=ax.transAxes)
        short = name.replace(f"{cat}_", "", 1) if name.startswith(cat) else name
        if len(short) > 25:
            short = short[:22] + "..."
        ax.set_title(short, fontsize=7, pad=2)
        ax.axis("off")
    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axes[r][c].axis("off")
    fig.suptitle(f"{cat.upper()} ({n} benchmarks)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=120)
    print(f"Saved {output_path.name} ({n} benchmarks)")
    plt.close(fig)


def build_overview_grid(by_cat):
    all_items = []
    for cat in CATEGORIES + ["other"]:
        if cat in by_cat:
            all_items.extend(by_cat[cat])
    n = len(all_items)
    cols = 15
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    for i, (name, fpath) in enumerate(all_items):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        try:
            img = mpimg.imread(str(fpath))
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, "ERR", ha="center", va="center", transform=ax.transAxes)
        short = name[:18] + ".." if len(name) > 20 else name
        ax.set_title(short, fontsize=4, pad=1)
        ax.axis("off")
    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axes[r][c].axis("off")
    fig.suptitle(f"All PCA Projections ({n} benchmarks)", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUTPUT_DIR / "all_shapes_grid.png", dpi=VIZ_DPI)
    print(f"Saved all_shapes_grid.png ({n} benchmarks)")
    plt.close(fig)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    by_cat = load_images()
    print(f"Found {sum(len(v) for v in by_cat.values())} PCA projections "
          f"in {len(by_cat)} categories\n")
    for cat in CATEGORIES + ["other"]:
        if cat not in by_cat:
            continue
        items = by_cat[cat]
        build_category_grid(cat, items, OUTPUT_DIR / f"shapes_{cat}.png")
    build_overview_grid(by_cat)
    print(f"\nAll grids saved to {OUTPUT_DIR}/")
