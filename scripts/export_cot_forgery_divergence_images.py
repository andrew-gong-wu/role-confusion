#!/usr/bin/env python3
"""Export CoT-forgery divergence figures in the repository's basic plot style."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path(
    "/Users/rigel/.codex/visualizations/2026/08/27/"
    "01a044d7-bc26-7da3-b5ab-f349e20dcf80/cot-forgery-drift.html"
)
DEFAULT_OUTPUT = ROOT / "reports" / "2026-08-28-cot-forgery-divergence-images"

# Deliberately different from the blue/orange/green reference palette.
SUCCESS = "#6a51a3"
UNCHANGED = "#008b8b"
DIFFERENCE = "#444444"
BLACK = "#111111"
GRID = "#b0b0b0"
WHITE = "#ffffff"


def get_font(size: int):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNS.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


FONT_32 = get_font(32)
FONT_29 = get_font(29)
FONT_25 = get_font(25)
FONT_23 = get_font(23)
FONT_21 = get_font(21)
FONT_19 = get_font(19)


def load_data(path: Path) -> dict:
    match = re.search(r"const DATA=(\{.*?\})\s*;\s*let axis=", path.read_text(), re.S)
    if not match:
        raise RuntimeError(f"Could not locate aggregate data in {path}")
    return json.loads(match.group(1))


def draw_text(draw, xy, value, font=FONT_21, anchor="la", fill=BLACK):
    draw.text(xy, value, font=font, anchor=anchor, fill=fill)


def rgba(hex_color: str, alpha: int):
    return tuple(bytes.fromhex(hex_color[1:])) + (alpha,)


def composite_save(image: Image.Image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = Image.alpha_composite(Image.new("RGBA", image.size, WHITE), image).convert("RGB")
    flat.save(path, "PNG", optimize=True, dpi=(180, 180))


def line_axes(draw, box, x_domain, y_domain, x_ticks, y_ticks, x_format):
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    sx = lambda value: left + (value - x_domain[0]) / (x_domain[1] - x_domain[0]) * width
    sy = lambda value: bottom - (value - y_domain[0]) / (y_domain[1] - y_domain[0]) * height
    for value in y_ticks:
        y = sy(value)
        draw.line((left, y, right, y), fill=rgba(GRID, 70), width=2)
        draw_text(draw, (left - 14, y), f"{value:g}", font=FONT_19, anchor="ra")
    for value in x_ticks:
        x = sx(value)
        draw.line((x, bottom, x, bottom + 9), fill=BLACK, width=2)
        draw_text(draw, (x, bottom + 16), x_format(value), font=FONT_19, anchor="ma")
    draw.rectangle(box, outline=BLACK, width=2)
    return sx, sy


def draw_band(draw, xs, low, high, sx, sy, color):
    points = [(sx(x), sy(y)) for x, y in zip(xs, low)]
    points += [(sx(x), sy(y)) for x, y in reversed(list(zip(xs, high)))]
    draw.polygon(points, fill=rgba(color, 38))


def draw_series(draw, xs, values, sx, sy, color):
    draw.line([(sx(x), sy(y)) for x, y in zip(xs, values)], fill=color, width=4, joint="curve")


def draw_legend(draw, x, y):
    draw.line((x, y, x + 42, y), fill=SUCCESS, width=5)
    draw_text(draw, (x + 52, y), "successful (n=148)", font=FONT_19, anchor="lm")
    x += 270
    draw.line((x, y, x + 42, y), fill=UNCHANGED, width=5)
    draw_text(draw, (x + 52, y), "unchanged (n=51)", font=FONT_19, anchor="lm")


def draw_forest_panel(draw, data, box, title, show_xlabel=True):
    left, top, right, bottom = box
    draw_text(draw, ((left + right) / 2, top - 48), title, font=FONT_25, anchor="ma")
    effects = data["axes"]["z_axis"]["effects"]
    rows = [("Successful", effects["Successful"], SUCCESS),
            ("Unchanged", effects["Unchanged"], UNCHANGED),
            ("Successful − unchanged", effects["Difference"], DIFFERENCE)]
    endpoints = [0.0]
    for _, values, _ in rows:
        endpoints.extend([values["low"], values["high"]])
    step = .5
    x_domain = (
        math.floor(min(endpoints) / step) * step,
        math.ceil(max(endpoints) / step) * step,
    )
    if x_domain[0] == x_domain[1]:
        x_domain = (x_domain[0] - step, x_domain[1] + step)
    plot_left = left + 330
    sx = lambda value: plot_left + (value - x_domain[0]) / (x_domain[1] - x_domain[0]) * (right - plot_left)
    tick_count = int(round((x_domain[1] - x_domain[0]) / step))
    ticks = [x_domain[0] + index * step for index in range(tick_count + 1)]
    for value in ticks:
        x = sx(value)
        draw.line((x, top, x, bottom), fill=rgba(GRID, 60), width=2)
        draw.line((x, bottom, x, bottom + 9), fill=BLACK, width=2)
        draw_text(draw, (x, bottom + 16), f"{value:g}", font=FONT_19, anchor="ma")
    draw.rectangle((plot_left, top, right, bottom), outline=BLACK, width=2)
    zero_x = sx(0)
    y_values = [top + (bottom - top) * fraction for fraction in (.2, .5, .8)]
    for (label, values, color), y in zip(rows, y_values):
        draw_text(draw, (plot_left - 20, y), label, font=FONT_21, anchor="rm")
        draw.line((sx(values["low"]), y, sx(values["high"]), y), fill=color, width=4)
        draw.ellipse((sx(values["mean"]) - 6, y - 6, sx(values["mean"]) + 6, y + 6), fill=color)
        value_text = f'{values["mean"]:.2f} [{values["low"]:.2f}, {values["high"]:.2f}]'
        text_width = draw.textbbox((0, 0), value_text, font=FONT_19)[2]
        value_x = sx(values["high"]) + 14
        anchor = "lm"
        if value_x + text_width > right - 10:
            value_x = sx(values["low"]) - 14
            anchor = "rm"
        draw_text(draw, (value_x, y), value_text, font=FONT_19, anchor=anchor)
    for y in range(top, bottom, 15):
        draw.line((zero_x, y, zero_x, min(y + 7, bottom)), fill=BLACK, width=2)
    if show_xlabel:
        draw_text(draw, ((plot_left + right) / 2, bottom + 62), "Paired final-drift effect (default-Assistant SD)", font=FONT_21, anchor="ma")


def draw_trajectory_panel(draw, data, kind, box, title, show_legend=False, show_xlabel=True):
    left, top, right, bottom = box
    draw_text(draw, ((left + right) / 2, top - 48), title, font=FONT_25, anchor="ma")
    if kind == "normalized":
        xs = [value * 100 for value in data["grid"]]
        x_domain, x_ticks = (0, 100), [0, 20, 40, 60, 80, 100]
        x_format = lambda value: f"{int(value)}%"
        y_domain, y_ticks = (-8, 1), [-8, -6, -4, -2, 0]
        xlabel = "Final-answer progress"
    else:
        xs = data["offsets"]
        x_domain, x_ticks = (min(xs), max(xs)), [-16, -12, -8, -4, 0, 4, 8]
        x_format = lambda value: f"{int(value)}"
        y_domain, y_ticks = (-6.5, 0), [-6, -5, -4, -3, -2, -1, 0]
        xlabel = "Tokens from first final-content token"
    sx, sy = line_axes(draw, box, x_domain, y_domain, x_ticks, y_ticks, x_format)
    groups = data["axes"]["z_axis"]["groups"]
    series = [("Successful", SUCCESS), ("Unchanged", UNCHANGED)]
    for name, color in series:
        values = groups[name][kind]
        draw_band(draw, xs, values["low"], values["high"], sx, sy, color)
    for name, color in series:
        draw_series(draw, xs, groups[name][kind]["mean"], sx, sy, color)
    if kind == "event":
        x = sx(0)
        for y in range(top, bottom, 15):
            draw.line((x, y, x, min(y + 7, bottom)), fill=BLACK, width=2)
    if show_legend:
        draw_legend(draw, right - 590, top + 38)
    if show_xlabel:
        draw_text(draw, ((left + right) / 2, bottom + 58), xlabel, font=FONT_21, anchor="ma")
    y_label = "Attack − baseline projection (SD)"
    layer = Image.new("RGBA", draw._image.size, (0, 0, 0, 0))
    label_draw = ImageDraw.Draw(layer)
    label_draw.text((0, 0), y_label, fill=BLACK, font=FONT_21)
    label = layer.crop(layer.getbbox()).rotate(90, expand=True)
    draw._image.alpha_composite(label, (left - 82, int((top + bottom - label.height) / 2)))


def combined_figure(data):
    image = Image.new("RGBA", (2520, 1620), WHITE)
    draw = ImageDraw.Draw(image, "RGBA")
    draw_text(draw, (1260, 32), "gpt-oss-20b Assistant-Axis divergence under CoT forgery (layer 16 block output)", font=FONT_32, anchor="ma")
    draw_forest_panel(draw, data, (105, 115, 2440, 455), "Late-final paired drift — successful vs unchanged attacks")
    draw_trajectory_panel(draw, data, "normalized", (150, 645, 2440, 980), "Final-answer progress — paired attack − baseline projection", show_legend=True)
    draw_trajectory_panel(draw, data, "event", (150, 1170, 2440, 1505), "Analysis-to-final transition — token 0 is first final-content token")
    return image


def forest_figure(data):
    image = Image.new("RGBA", (2520, 900), WHITE)
    draw = ImageDraw.Draw(image, "RGBA")
    draw_text(draw, (1260, 32), "gpt-oss-20b CoT-forgery late-final drift (layer 16 block output)", font=FONT_32, anchor="ma")
    draw_forest_panel(draw, data, (105, 120, 2440, 690), "Successful vs unchanged attacks")
    return image


def trajectory_figure(data, kind):
    image = Image.new("RGBA", (2520, 900), WHITE)
    draw = ImageDraw.Draw(image, "RGBA")
    if kind == "normalized":
        suptitle = "gpt-oss-20b CoT-forgery divergence across final-answer progress"
        title = "Paired attack − baseline projection — layer 16 block output"
    else:
        suptitle = "gpt-oss-20b CoT-forgery divergence around the analysis-to-final transition"
        title = "Token 0 is the first final-content token — layer 16 block output"
    draw_text(draw, (1260, 32), suptitle, font=FONT_32, anchor="ma")
    draw_trajectory_panel(draw, data, kind, (145, 130, 2440, 730), title, show_legend=True)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    data = load_data(args.source)
    outputs = {
        "cot-forgery-success-effect-forest.png": forest_figure(data),
        "cot-forgery-final-progress.png": trajectory_figure(data, "normalized"),
        "cot-forgery-analysis-final-transition.png": trajectory_figure(data, "event"),
        "cot-forgery-divergence-combined.png": combined_figure(data),
    }
    for name, image in outputs.items():
        path = args.output_dir / name
        composite_save(image, path)
        print(path)


if __name__ == "__main__":
    main()
