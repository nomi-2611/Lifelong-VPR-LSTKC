from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Plot a VPRTempo similarity matrix as a heatmap.")
    parser.add_argument("--matrix", required=True, help="Path to a .npy or .npz similarity matrix")
    parser.add_argument("--output", required=True, help="Output image path")
    parser.add_argument("--title", default="VPRTempo Similarity Matrix")
    args = parser.parse_args()

    matrix_path = Path(args.matrix)
    matrix = np.load(matrix_path)
    if isinstance(matrix, np.lib.npyio.NpzFile):
        if "arr_0" in matrix.files:
            matrix = matrix["arr_0"]
        else:
            matrix = matrix[matrix.files[0]]
    matrix = np.asarray(matrix, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(8.5, 6.6), dpi=260, constrained_layout=True)
    im = ax.imshow(matrix, cmap="magma", aspect="auto")
    ax.set_title(args.title, fontsize=16, fontweight="bold")
    ax.set_xlabel("Database / Gallery Index")
    ax.set_ylabel("Query Index")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Similarity Score", rotation=90)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved similarity matrix figure to {output_path}")


if __name__ == "__main__":
    main()
