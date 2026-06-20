import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_font(size, bold=False):
    candidates = []
    if bold:
        candidates.extend(
            [
                "C:/Windows/Fonts/arialbd.ttf",
                "C:/Windows/Fonts/segoeuib.ttf",
                "C:/Windows/Fonts/timesbd.ttf",
            ]
        )
    candidates.extend(
        [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/times.ttf",
        ]
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def fit_panel(img, size, background="#ffffff"):
    canvas = Image.new("RGB", size, background)
    panel_w, panel_h = size
    img = img.convert("RGB")
    ratio = min(panel_w / img.width, panel_h / img.height)
    new_size = (max(1, int(img.width * ratio)), max(1, int(img.height * ratio)))
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    x = (panel_w - img.width) // 2
    y = (panel_h - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def add_panel_frame(draw, box, title, badge, fonts, paper=False):
    x0, y0, x1, y1 = box
    frame_color = "#cfd6df" if paper else "#d9dde5"
    header_bg = "#eef2f7" if paper else "#eef3f8"
    badge_bg = "#173a63" if paper else "#1f4e79"
    radius = 12 if paper else 18
    header_h = 48 if paper else 56
    draw.rounded_rectangle(box, radius=radius, outline=frame_color, width=2 if paper else 3, fill="#ffffff")
    draw.rounded_rectangle((x0, y0, x1, y0 + header_h), radius=radius, fill=header_bg)
    draw.rectangle((x0, y0 + header_h // 2, x1, y0 + header_h), fill=header_bg)
    draw.rounded_rectangle((x0 + 16, y0 + 10, x0 + 56, y0 + 38), radius=10, fill=badge_bg)
    draw.text((x0 + 29, y0 + 11), badge, font=fonts["badge"], fill="white")
    draw.text((x0 + 70, y0 + 11), title, font=fonts["panel_title"], fill="#1c2e40")


def parse_stage_results(log_res_path):
    lines = [line.strip() for line in Path(log_res_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    stages = []
    i = 0
    while i < len(lines):
        if not lines[i].startswith("epoch:"):
            i += 1
            continue
        if i + 5 >= len(lines) or lines[i + 1].startswith("epoch:"):
            i += 1
            continue

        epoch = int(lines[i].split(":", 1)[1].strip())
        seen_names = [name for name in lines[i + 1].split("\t") if name and name != "|Average"]
        seen_vals = [float(x) for x in re.findall(r"\d+\.\d+", lines[i + 2])]
        unseen_names = [name for name in lines[i + 4].split("\t") if name and name != "|Average"]
        unseen_vals = [float(x) for x in re.findall(r"\d+\.\d+", lines[i + 5])]

        seen = {}
        for idx, name in enumerate(seen_names):
            seen[name] = {"mAP": seen_vals[idx * 2], "R1": seen_vals[idx * 2 + 1]}
        unseen = {}
        for idx, name in enumerate(unseen_names):
            unseen[name] = {"mAP": unseen_vals[idx * 2], "R1": unseen_vals[idx * 2 + 1]}

        stages.append(
            {
                "epoch": epoch,
                "seen": seen,
                "seen_avg": {"mAP": seen_vals[-2], "R1": seen_vals[-1]},
                "unseen": unseen,
                "unseen_avg": {"mAP": unseen_vals[-2], "R1": unseen_vals[-1]},
            }
        )
        i += 6
    return stages


def build_takeaways(stages):
    final_stage = stages[-1]
    prev_stage = stages[-2] if len(stages) > 1 else stages[-1]
    best_dataset = max(final_stage["seen"].items(), key=lambda item: item[1]["mAP"])
    hardest_dataset = min(final_stage["seen"].items(), key=lambda item: item[1]["mAP"])
    market_drop_map = final_stage["seen"]["market1501"]["mAP"] - prev_stage["seen"]["market1501"]["mAP"]
    market_drop_r1 = final_stage["seen"]["market1501"]["R1"] - prev_stage["seen"]["market1501"]["R1"]
    seen_avg = final_stage["seen_avg"]
    unseen_avg = final_stage["unseen_avg"]
    return {
        "seen_avg": seen_avg,
        "unseen_avg": unseen_avg,
        "best_dataset_name": best_dataset[0],
        "best_dataset_map": best_dataset[1]["mAP"],
        "best_dataset_r1": best_dataset[1]["R1"],
        "hardest_dataset_name": hardest_dataset[0],
        "hardest_dataset_map": hardest_dataset[1]["mAP"],
        "hardest_dataset_r1": hardest_dataset[1]["R1"],
        "market_drop_map": market_drop_map,
        "market_drop_r1": market_drop_r1,
    }


def draw_multiline(draw, xy, text, font, fill, spacing=8):
    draw.multiline_text(xy, text, font=font, fill=fill, spacing=spacing)


def create_dashboard(run_dir, paper=False):
    analysis_dir = Path(run_dir) / "analysis"
    panels = [
        ("A", "Final Stage Dataset Performance", analysis_dir / "stage_summary.png"),
        ("B", "Seen and Unseen Progression", analysis_dir / "stage_progression.png"),
        ("C", "Retention After Stage 3", analysis_dir / "retention_delta.png"),
        ("D", "Stage-wise Training Loss Curves", analysis_dir / "training_losses.png"),
    ]
    for _, _, path in panels:
        if not path.exists():
            raise FileNotFoundError(f"Missing panel image: {path}")

    stages = parse_stage_results(analysis_dir.parent / "log_res.txt")
    takeaways = build_takeaways(stages)

    if paper:
        fonts = {
            "title": load_font(30, bold=True),
            "subtitle": load_font(16),
            "panel_title": load_font(18, bold=True),
            "badge": load_font(20, bold=True),
            "footer": load_font(13),
            "callout_title": load_font(18, bold=True),
            "callout_body": load_font(15),
        }
        width = 2400
        height = 1580
        margin = 26
        gap = 20
        header_h = 78
        footer_h = 32
        panel_header_h = 58
        insight_h = 130
        bg = "#ffffff"
        outer_outline = "#d4dbe4"
        title = "LSTKC Overall Training Analysis"
        subtitle = "Setting 1 | Market1501 -> CUHK-SYSU -> DukeMTMC | Full three-stage run"
    else:
        fonts = {
            "title": load_font(34, bold=True),
            "subtitle": load_font(18),
            "panel_title": load_font(20, bold=True),
            "badge": load_font(22, bold=True),
            "footer": load_font(16),
            "callout_title": load_font(18, bold=True),
            "callout_body": load_font(15),
        }
        width = 2200
        height = 1700
        margin = 48
        gap = 28
        header_h = 120
        footer_h = 52
        panel_header_h = 70
        insight_h = 0
        bg = "#f6f8fb"
        outer_outline = "#d8dee8"
        title = "LSTKC Training Dashboard"
        subtitle = "Run: run_full_3stage | Setting 1 | Market1501 -> CUHK-SYSU -> DukeMTMC"

    panel_w = (width - margin * 2 - gap) // 2
    panel_area_h = height - margin * 2 - header_h - footer_h - insight_h - gap
    panel_h = (panel_area_h - gap) // 2
    inner_w = panel_w - 20
    inner_h = panel_h - panel_header_h - 12

    canvas = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((10, 10, width - 10, height - 10), radius=20 if paper else 28, fill=bg, outline=outer_outline, width=2)
    draw.text((margin, margin - 4), title, font=fonts["title"], fill="#122033")
    draw.text((margin, margin + 34), subtitle, font=fonts["subtitle"], fill="#4b5b6b")

    y_offset = margin + header_h

    if paper:
        callout_box = (margin, y_offset, width - margin, y_offset + insight_h - 10)
        draw.rounded_rectangle(callout_box, radius=12, fill="#f7f9fc", outline="#d3dbe6", width=2)
        draw.text((margin + 18, y_offset + 12), "Key findings", font=fonts["callout_title"], fill="#18324a")
        callout_text = (
            f"Seen average at Stage 3: {takeaways['seen_avg']['mAP']:.1f} mAP / {takeaways['seen_avg']['R1']:.1f} R1.  "
            f"Best dataset: {takeaways['best_dataset_name']} ({takeaways['best_dataset_map']:.1f} mAP, {takeaways['best_dataset_r1']:.1f} R1).  "
            f"Hardest dataset: {takeaways['hardest_dataset_name']} ({takeaways['hardest_dataset_map']:.1f} mAP, {takeaways['hardest_dataset_r1']:.1f} R1).\n"
            f"After adding the third stage, Market1501 changes by {takeaways['market_drop_map']:.1f} mAP and "
            f"{takeaways['market_drop_r1']:.1f} R1, indicating moderate forgetting while preserving strong cross-stage performance."
        )
        draw_multiline(draw, (margin + 18, y_offset + 42), callout_text, fonts["callout_body"], "#2c3e50", spacing=7)
        y_offset += insight_h

    positions = [
        (margin, y_offset),
        (margin + panel_w + gap, y_offset),
        (margin, y_offset + panel_h + gap),
        (margin + panel_w + gap, y_offset + panel_h + gap),
    ]

    for (badge, title_text, path), (x, y) in zip(panels, positions):
        box = (x, y, x + panel_w, y + panel_h)
        add_panel_frame(draw, box, title_text, badge, fonts, paper=paper)
        img = Image.open(path)
        panel = fit_panel(img, (inner_w, inner_h))
        canvas.paste(panel, (x + 10, y + panel_header_h))

    if paper:
        bottom_text = (
            f"Stage 2 seen average: {stages[-2]['seen_avg']['mAP']:.1f}/{stages[-2]['seen_avg']['R1']:.1f}; "
            f"Stage 3 seen average: {stages[-1]['seen_avg']['mAP']:.1f}/{stages[-1]['seen_avg']['R1']:.1f}; "
            f"Unseen average at Stage 3: {takeaways['unseen_avg']['mAP']:.1f}/{takeaways['unseen_avg']['R1']:.1f}."
        )
    else:
        bottom_text = "Generated from TensorBoard scalars and log_res.txt"
    draw.text((margin, height - margin - 4), bottom_text, font=fonts["footer"], fill="#6a7684")

    out_name = "paper_dashboard_hd.png" if paper else "dashboard_overview.png"
    out_path = analysis_dir / out_name
    canvas.save(out_path, quality=95)
    return out_path, stages, takeaways


def save_captions(run_dir, takeaways):
    analysis_dir = Path(run_dir) / "analysis"
    caption_path = analysis_dir / "dashboard_captions.md"
    zh = (
        "图注（中文）\n"
        "图 X 展示了 LSTKC 在 Setting 1 下的完整三阶段训练分析结果，训练顺序为 Market1501、CUHK-SYSU 和 DukeMTMC。"
        f"在第三阶段结束后，模型在已见数据集上的平均性能达到 {takeaways['seen_avg']['mAP']:.1f}% mAP 和 "
        f"{takeaways['seen_avg']['R1']:.1f}% Rank-1。分数据集来看，CUHK-SYSU 表现最佳，达到 "
        f"{takeaways['best_dataset_map']:.1f}% mAP / {takeaways['best_dataset_r1']:.1f}% Rank-1；"
        f"{takeaways['hardest_dataset_name']} 相对更具挑战，最终达到 {takeaways['hardest_dataset_map']:.1f}% mAP / "
        f"{takeaways['hardest_dataset_r1']:.1f}% Rank-1。阶段对比结果表明，在引入第三阶段后，"
        f"Market1501 相比第二阶段下降了 {abs(takeaways['market_drop_map']):.1f} 个 mAP 点和 "
        f"{abs(takeaways['market_drop_r1']):.1f} 个 Rank-1 点，说明模型在获得新知识的同时存在一定遗忘，但整体仍保持了较好的跨阶段识别能力。"
    )
    en = (
        "Caption (English)\n"
        "Fig. X presents a paper-style overview of the full three-stage LSTKC training run under Setting 1, "
        "with the training order Market1501, CUHK-SYSU, and DukeMTMC. "
        f"After Stage 3, the model achieves a seen-dataset average of {takeaways['seen_avg']['mAP']:.1f}% mAP and "
        f"{takeaways['seen_avg']['R1']:.1f}% Rank-1. Among the seen datasets, {takeaways['best_dataset_name']} shows "
        f"the strongest final performance at {takeaways['best_dataset_map']:.1f}% mAP / {takeaways['best_dataset_r1']:.1f}% Rank-1, "
        f"whereas {takeaways['hardest_dataset_name']} remains the most challenging with "
        f"{takeaways['hardest_dataset_map']:.1f}% mAP / {takeaways['hardest_dataset_r1']:.1f}% Rank-1. "
        "The stage-wise comparison further shows moderate forgetting after introducing the third stage: "
        f"Market1501 drops by {abs(takeaways['market_drop_map']):.1f} mAP points and {abs(takeaways['market_drop_r1']):.1f} Rank-1 points "
        "relative to Stage 2, while the overall cross-stage performance remains strong."
    )
    caption_path.write_text(zh + "\n\n" + en + "\n", encoding="utf-8")
    return caption_path


def main():
    parser = argparse.ArgumentParser(description="Build dashboard figures and captions from analysis plots.")
    parser.add_argument("--run-dir", required=True, help="Path to a run directory such as logs/run_full_3stage")
    args = parser.parse_args()

    overview_path, stages, takeaways = create_dashboard(args.run_dir, paper=False)
    paper_path, _, _ = create_dashboard(args.run_dir, paper=True)
    caption_path = save_captions(args.run_dir, takeaways)
    print(f"Saved dashboard to {overview_path}")
    print(f"Saved paper dashboard to {paper_path}")
    print(f"Saved captions to {caption_path}")


if __name__ == "__main__":
    main()
