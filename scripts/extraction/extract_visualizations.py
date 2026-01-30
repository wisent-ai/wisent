#!/usr/bin/env python3
"""Extract visualizations from RepScan JSON output to viewable image files."""
import json
import base64
import argparse
from pathlib import Path


def create_combined_figure(visualizations: dict, layer_num: int, metrics: dict = None) -> bytes:
    """Create a combined 3x3 figure from individual visualization images."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    import io

    # Define layout: 3x3 grid
    layout = [
        ["pca_projection", "tsne_projection", "umap_projection"],
        ["pacmap_projection", "diff_vectors", "alignment_distribution"],
        ["eigenvalue_spectrum", "norm_distribution", "pairwise_distances"],
    ]

    fig, axes = plt.subplots(3, 3, figsize=(18, 18))

    for row_idx, row in enumerate(layout):
        for col_idx, viz_name in enumerate(row):
            ax = axes[row_idx, col_idx]
            viz_data = visualizations.get(viz_name)

            if viz_data and isinstance(viz_data, str) and viz_data.startswith("iVBOR"):
                # Decode base64 PNG and display
                img_bytes = base64.b64decode(viz_data)
                img = Image.open(io.BytesIO(img_bytes))
                ax.imshow(img)
                ax.axis('off')
            else:
                ax.text(0.5, 0.5, f"{viz_name}\n(not available)",
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.axis('off')

    # Title
    title_parts = [f"Layer {layer_num}"]
    if metrics:
        title_parts.append(f"Linear: {metrics.get('linear_probe_accuracy', 0):.2f}")
        title_parts.append(f"ICD: {metrics.get('icd_icd', 0):.1f}")
        title_parts.append(f"Rec: {metrics.get('recommended_method', 'N/A')}")
    fig.suptitle(" | ".join(title_parts), fontsize=16, y=0.98)

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    return buf.read()


def extract_visualizations(json_path: str, output_dir: str, layers: list = None, combined: bool = True):
    """Extract all visualizations from RepScan output."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(json_path) as f:
        data = json.load(f)

    for layer_data in data.get("layers", []):
        layer_num = layer_data.get("layer", 0)

        if layers and layer_num not in layers:
            continue

        visualizations = layer_data.get("metrics", {}).get("visualizations", {})
        metrics = layer_data.get("metrics", {})

        if combined:
            # Create single combined figure per layer
            img_data = create_combined_figure(visualizations, layer_num, metrics)
            img_path = output_path / f"layer_{layer_num:02d}_summary.png"
            with open(img_path, "wb") as f:
                f.write(img_data)
            print(f"Saved: {img_path}")
        else:
            # Save individual visualizations
            for viz_name, viz_data in visualizations.items():
                if isinstance(viz_data, str) and viz_data.startswith("iVBOR"):
                    img_data = base64.b64decode(viz_data)
                    img_path = output_path / f"layer_{layer_num:02d}_{viz_name}.png"
                    with open(img_path, "wb") as f:
                        f.write(img_data)
                    print(f"Saved: {img_path}")

    print(f"\nVisualizations saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract visualizations from RepScan output")
    parser.add_argument("json_path", help="Path to RepScan JSON output")
    parser.add_argument("-o", "--output", default="output/visualizations", help="Output directory")
    parser.add_argument("-l", "--layers", type=int, nargs="+", help="Specific layers to extract")
    parser.add_argument("--separate", action="store_true", help="Save as separate images instead of combined")
    args = parser.parse_args()

    extract_visualizations(args.json_path, args.output, args.layers, combined=not args.separate)


if __name__ == "__main__":
    main()
