"""Render the two StressID manuscript figures from retained aggregate results."""

import argparse
import csv
import math
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


WIDTH, HEIGHT = 518.4, 374.4
LEFT, RIGHT, BOTTOM = 0.24 * WIDTH, 0.96 * WIDTH, 0.15 * HEIGHT
BLUE, ORANGE, BLACK = "#1f77b4", "#ff7f0e", "#000000"
MODELS = (("RandomForest", "Random Forest"), ("XGBoost", "XGBoost"),
          ("LightGBM", "LightGBM"))
DESIGNS = (
    ("standard_strict_LOSO", 0.13, BLUE, "circle", 350,
     "Participant held out (LOSO)"),
    ("subject_and_task_disjoint", -0.13, ORANGE, "square", 335.2,
     "Participant and test task held out"),
)


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float


def read_results(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def get_interval(rows, **criteria):
    matching = [row for row in rows
                if all(row.get(column) == value for column, value in criteria.items())]
    if len(matching) != 1:
        raise ValueError(f"Expected one row for {criteria}, found {len(matching)}")
    row = matching[0]
    values = tuple(float(row[key]) for key in ("estimate", "ci_low", "ci_high"))
    estimate, low, high = values
    if not all(map(math.isfinite, values)) or low > high:
        raise ValueError(f"Invalid estimate or interval for {criteria}: {values}")
    return Interval(estimate, low, high)


def load_font(font_dir):
    if font_dir is None:
        candidates = [Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/dejavu")]
        matplotlib = find_spec("matplotlib")
        if matplotlib and matplotlib.origin:
            candidates.append(Path(matplotlib.origin).parent / "mpl-data/fonts/ttf")
        font_dir = next((p for p in candidates if (p / "DejaVuSans.ttf").is_file()
                         and (p / "DejaVuSans-Bold.ttf").is_file()), None)
    if font_dir is None:
        raise FileNotFoundError("Supply --font-dir with the DejaVu Sans regular and bold TTF files.")
    for name in ("DejaVuSans", "DejaVuSans-Bold"):
        pdfmetrics.registerFont(TTFont(name, str(font_dir / f"{name}.ttf")))


def write_text(pdf, x, y, label, *, size=12, alignment="left", bold=False):
    font = "DejaVuSans-Bold" if bold else "DejaVuSans"
    pdf.setFillColor(HexColor(BLACK))
    pdf.setFont(font, size)
    if alignment == "right":
        pdf.drawRightString(x, y, label)
    elif alignment == "centre":
        pdf.drawCentredString(x, y, label)
    else:
        pdf.drawString(x, y, label)


def line(pdf, x1, y1, x2, y2, *, color=BLACK, width=0.8, dashed=False):
    pdf.setStrokeColor(HexColor(color))
    pdf.setLineWidth(width)
    pdf.setDash([3.7, 1.6] if dashed else [])
    pdf.line(x1, y1, x2, y2)
    pdf.setDash([])


def draw_interval(pdf, x, low, high, y, *, marker="circle", color=BLUE):
    line(pdf, low, y, high, y, color=color, width=1.4)
    for endpoint in (low, high):
        line(pdf, endpoint, y - 3, endpoint, y + 3, color=color, width=1)
    pdf.setFillColor(HexColor(color))
    pdf.setStrokeColor(HexColor(color))
    if marker == "circle":
        pdf.circle(x, y, 3.5, stroke=0, fill=1)
    else:
        pdf.rect(x - 3.5, y - 3.5, 7, 7, stroke=0, fill=1)


def draw_axis(pdf, *, x_limits, y_limits, top, ticks, xlabel):
    x_min, x_max = x_limits
    y_min, y_max = y_limits

    def dx(x):
        return LEFT + (x - x_min) / (x_max - x_min) * (RIGHT - LEFT)

    def dy(y):
        return BOTTOM + (y - y_min) / (y_max - y_min) * (top - BOTTOM)

    line(pdf, LEFT, BOTTOM, RIGHT, BOTTOM)
    for tick in ticks:
        x = dx(tick)
        line(pdf, x, BOTTOM, x, BOTTOM - 4)
        write_text(pdf, x, BOTTOM - 16, f"{tick:.2f}".replace("-", "−"), alignment="centre")
    write_text(pdf, (LEFT + RIGHT) / 2, BOTTOM - 37, xlabel, size=13, alignment="centre")
    return dx, dy


def new_pdf(path):
    pdf = canvas.Canvas(str(path), pagesize=(WIDTH, HEIGHT), pageCompression=1, invariant=1)
    pdf.setTitle(path.stem)
    pdf.setAuthor("")
    pdf.setCreator("ReportLab native vector export of retained CSV values")
    pdf.setFillColor(HexColor("#ffffff"))
    pdf.rect(0, 0, WIDTH, HEIGHT, stroke=0, fill=1)
    return pdf


def figure_1(rows, path):
    groups = (("stress", "Continuous stress", (6, 5, 4)),
              ("arousal", "Continuous arousal", (2, 1, 0)))
    pdf = new_pdf(path)
    top = 0.93 * HEIGHT
    dx, dy = draw_axis(
        pdf, x_limits=(-0.125, 0.26), y_limits=(-0.65, 7.25), top=top,
        ticks=(-0.10, -0.05, 0, 0.05, 0.10, 0.15, 0.20, 0.25),
        xlabel="Change in Spearman’s ρ (combined − context-only)",
    )
    line(pdf, dx(0), BOTTOM, dx(0), top, color=BLUE, width=1, dashed=True)
    write_text(pdf, 0.025 * WIDTH, 0.97 * HEIGHT - 9,
               "Horizontal bars: 95% participant-cluster bootstrap CIs", size=11)
    for target, label, positions in groups:
        write_text(pdf, 0.025 * WIDTH, dy(positions[0] + 0.9) - 4.55,
                   label, size=13, bold=True)
        for (model, model_label), y in zip(MODELS, positions):
            interval = get_interval(
                rows, row_type="paired_difference", target=target, model=model,
                metric="spearman_rho", comparison="combined_minus_context_only",
            )
            write_text(pdf, LEFT - 12, dy(y) - 4.2, model_label, alignment="right")
            draw_interval(pdf, dx(interval.estimate), dx(interval.low), dx(interval.high), dy(y))
    pdf.showPage()
    pdf.save()


def figure_2(rows, path):
    pdf = new_pdf(path)
    top = 0.85 * HEIGHT
    dx, dy = draw_axis(
        pdf, x_limits=(0.44, 0.70), y_limits=(-0.55, 2.95), top=top,
        ticks=(0.45, 0.50, 0.55, 0.60, 0.65, 0.70), xlabel="Task-condition balanced accuracy",
    )
    line(pdf, dx(0.5), BOTTOM, dx(0.5), top, color=BLUE, width=1, dashed=True)
    for _, _, color, marker, y, label in DESIGNS:
        draw_interval(pdf, LEFT + 6, LEFT + 1, LEFT + 11, y, color=color, marker=marker)
        write_text(pdf, LEFT + 24, y - 4.2, label)
    for index, (_, model_label) in enumerate(MODELS):
        write_text(pdf, LEFT - 12, dy(2 - index) - 4.2, model_label, alignment="right")
    for design, offset, color, marker, _, _ in DESIGNS:
        for index, (model, _) in enumerate(MODELS):
            interval = get_interval(rows, target="task_condition", model=model,
                                    metric="balanced_accuracy", evaluation_design=design)
            draw_interval(pdf, dx(interval.estimate), dx(interval.low), dx(interval.high),
                          dy(2 - index + offset), color=color, marker=marker)
    pdf.showPage()
    pdf.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-csv", required=True, type=Path,
                        help="Retained context_incremental_results.csv")
    parser.add_argument("--transfer-csv", required=True, type=Path,
                        help="Retained subject_task_disjoint_all_models.csv")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--font-dir", type=Path,
                        help="Directory with DejaVuSans.ttf and DejaVuSans-Bold.ttf")
    args = parser.parse_args()
    context = read_results(args.context_csv)
    transfer = read_results(args.transfer_csv)
    load_font(args.font_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figure_1(context, args.out_dir / "Figure_1.pdf")
    figure_2(transfer, args.out_dir / "Figure_2.pdf")


if __name__ == "__main__":
    main()
