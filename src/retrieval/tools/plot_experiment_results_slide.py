from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle


def add_card(ax, xy, wh, title, subtitle, accent, face="#ffffff", title_size=20, sub_size=16):
    x, y = xy
    w, h = wh
    card = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=2.0,
        edgecolor=accent,
        facecolor=face,
    )
    ax.add_patch(card)
    ax.text(x + 0.04 * w, y + h * 0.72, title, fontsize=title_size, weight="bold", color="#18212b")
    ax.text(x + 0.04 * w, y + h * 0.42, subtitle, fontsize=sub_size, color="#334155", linespacing=1.45)


def add_step(ax, center_x, y, label, status, accent, fill, arrow_to=None):
    ax.add_patch(Circle((center_x, y), 0.03, facecolor=fill, edgecolor=accent, linewidth=2))
    ax.text(center_x, y, status, ha="center", va="center", fontsize=18, weight="bold", color=accent)
    ax.text(center_x, y - 0.065, label, ha="center", va="top", fontsize=16, color="#0f172a", weight="bold")
    if arrow_to is not None:
        ax.annotate(
            "",
            xy=(arrow_to - 0.045, y),
            xytext=(center_x + 0.045, y),
            arrowprops=dict(arrowstyle="->", lw=2.8, color="#94a3b8"),
        )


def build_slide(output_path: Path):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(16, 9), dpi=220)
    ax = plt.axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.patch.set_facecolor("#f4f7fb")
    ax.add_patch(FancyBboxPatch((0.03, 0.04), 0.94, 0.92, boxstyle="round,pad=0.015,rounding_size=0.03",
                                linewidth=0, facecolor="#eef4fb"))

    ax.text(0.06, 0.92, "实验结果", fontsize=28, weight="bold", color="#0f172a")
    ax.text(0.06, 0.875, "实验表明，matrix 直接替换效果不好，而融合方式有效，其中 embedding 表现更好。", fontsize=16, color="#475569")

    add_card(
        ax,
        (0.06, 0.60),
        (0.26, 0.21),
        "结论 1",
        "matrix 替换效果差\n不适合直接使用",
        accent="#dc2626",
        face="#fff7f7",
    )
    add_card(
        ax,
        (0.37, 0.60),
        (0.26, 0.21),
        "结论 2",
        "fuse 优于 replace\n外部信息适合作为辅助",
        accent="#d97706",
        face="#fff9ed",
    )
    add_card(
        ax,
        (0.68, 0.60),
        (0.26, 0.21),
        "结论 3",
        "embedding 更有效\n可直接参与特征学习",
        accent="#059669",
        face="#f2fbf7",
    )

    add_card(
        ax,
        (0.06, 0.18),
        (0.88, 0.28),
        "最优结果",
        "linear fusion (0.30)\n≈ 26.0 mAP / 19.0 R@1",
        accent="#2563eb",
        face="#f5f9ff",
        title_size=24,
        sub_size=22,
    )

    ax.text(0.10, 0.41, "方法演进对比", fontsize=18, weight="bold", color="#1e293b")
    add_step(ax, 0.18, 0.33, "Matrix Replace", "X", "#dc2626", "#fee2e2", arrow_to=0.39)
    add_step(ax, 0.39, 0.33, "Matrix Fuse", "M", "#d97706", "#ffedd5", arrow_to=0.60)
    add_step(ax, 0.60, 0.33, "Embedding", "E", "#059669", "#dcfce7", arrow_to=0.81)
    add_step(ax, 0.81, 0.33, "Linear Fusion 0.30", "B", "#2563eb", "#dbeafe")

    ax.text(0.10, 0.24, "低", fontsize=13, color="#64748b")
    ax.text(0.35, 0.24, "中", fontsize=13, color="#64748b")
    ax.text(0.58, 0.24, "较高", fontsize=13, color="#64748b")
    ax.text(0.78, 0.24, "当前最优", fontsize=13, color="#2563eb", weight="bold")

    ax.text(0.70, 0.12, "第 7 页：实验结果", fontsize=14, color="#64748b")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Create a slide-style figure for experiment results.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    build_slide(Path(args.output))


if __name__ == "__main__":
    main()
