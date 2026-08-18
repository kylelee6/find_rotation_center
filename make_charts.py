#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""REPORT.md에 임베드할 독립 실행형 SVG 차트 파일들을 생성한다."""
import json
import math
from pathlib import Path

HERE = Path(".")
OUT_DIR = HERE / "charts"
OUT_DIR.mkdir(exist_ok=True)

center_data = json.loads((HERE / "multi_image_precision_result.json").read_text(encoding="utf-8"))
angle_data = json.loads((HERE / "angle_precision_result.json").read_text(encoding="utf-8"))

COLOR_AVG = "#2a78d6"
COLOR_LS = "#008300"
COLOR_MUTED = "#898781"
COLOR_GRID = "#e1e0d9"
COLOR_AXIS = "#c3c2b7"
COLOR_TEXT = "#52514e"
COLOR_BG = "#fcfcfb"


def build_loglog_svg(series, x_ticks, y_ticks, x_label, y_label, legend,
                      width=640, height=420, margin=(72, 30, 24, 56)):
    left, right, top, bottom = margin
    plot_w = width - left - right
    plot_h = height - top - bottom

    xmin_log, xmax_log = math.log10(min(x_ticks)), math.log10(max(x_ticks))
    ymin_log, ymax_log = math.log10(min(y_ticks)), math.log10(max(y_ticks))

    def xpix(n):
        return left + (math.log10(n) - xmin_log) / (xmax_log - xmin_log) * plot_w

    def ypix(v):
        return top + plot_h - (math.log10(v) - ymin_log) / (ymax_log - ymin_log) * plot_h

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'font-family="system-ui, -apple-system, Segoe UI, sans-serif">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{COLOR_BG}"/>',
        '<style>',
        f'.grid{{stroke:{COLOR_GRID};stroke-width:1}} .axis{{stroke:{COLOR_AXIS};stroke-width:1}}',
        f'.tick{{fill:{COLOR_MUTED};font-size:12px}} .lbl{{fill:{COLOR_TEXT};font-size:13px}}',
        f'.leg{{fill:{COLOR_TEXT};font-size:12.5px}}',
        '</style>',
    ]

    for yt in y_ticks:
        yp = ypix(yt)
        svg.append(f'<line class="grid" x1="{left}" y1="{yp:.1f}" x2="{width-right}" y2="{yp:.1f}"/>')
        svg.append(f'<text class="tick" x="{left-8}" y="{yp+4:.1f}" text-anchor="end">{yt}</text>')

    svg.append(f'<line class="axis" x1="{left}" y1="{top+plot_h}" x2="{width-right}" y2="{top+plot_h}"/>')
    svg.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}"/>')

    for xt in x_ticks:
        xp = xpix(xt)
        svg.append(f'<text class="tick" x="{xp:.1f}" y="{top+plot_h+18:.1f}" text-anchor="middle">{xt}</text>')

    svg.append(f'<text class="lbl" x="{left+plot_w/2:.1f}" y="{height-10}" text-anchor="middle">{x_label}</text>')
    svg.append(f'<text class="lbl" x="{-(top+plot_h/2):.1f}" y="18" transform="rotate(-90)" text-anchor="middle">{y_label}</text>')

    for s in series:
        pts = " ".join(f"{xpix(n):.1f},{ypix(v):.1f}" for n, v in s["points"])
        color = s["color"]
        dash = ' stroke-dasharray="5 5"' if s.get("dash") else ""
        svg.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.2"{dash}/>')
        if s.get("markers", True):
            for n, v in s["points"]:
                svg.append(f'<circle cx="{xpix(n):.1f}" cy="{ypix(v):.1f}" r="4.2" fill="{color}"/>')

    # legend
    lx, ly = left + 8, top - 6
    for i, (label, color, dash) in enumerate(legend):
        cx = lx + i * 190
        if dash:
            svg.append(f'<line x1="{cx}" y1="{ly}" x2="{cx+18}" y2="{ly}" stroke="{color}" stroke-width="2" stroke-dasharray="5 5"/>')
        else:
            svg.append(f'<line x1="{cx}" y1="{ly}" x2="{cx+18}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<text class="leg" x="{cx+24}" y="{ly+4}">{label}</text>')

    svg.append('</svg>')
    return "\n".join(svg)


Ns = center_data["n_grid"]
cres = center_data["results_by_N"]
ares = angle_data["results_by_N"]

center_avg = [(n, cres[str(n)]["simple_average"]["rmse_2d"]) for n in Ns]
center_ls = [(n, cres[str(n)]["joint_least_squares"]["rmse_2d"]) for n in Ns]
baseline = center_ls[0][1]
center_theory = [(n, baseline / math.sqrt(n)) for n in Ns]

angle_std = [(n, ares[str(n)]["angle_deg"]["std"]) for n in Ns]
angle_baseline_std = angle_std[0][1]
angle_theory = [(n, angle_baseline_std / math.sqrt(n)) for n in Ns]

sameangle_center_rmse = [(n, ares[str(n)]["rmse_center_2d"]) for n in Ns]

chart1 = build_loglog_svg(
    series=[
        {"points": center_avg, "color": COLOR_AVG},
        {"points": center_ls, "color": COLOR_LS},
        {"points": center_theory, "color": COLOR_MUTED, "dash": True, "markers": False},
    ],
    x_ticks=Ns, y_ticks=[0.6, 0.8, 1.0, 1.5, 2.0, 2.5],
    x_label="사용한 영상 쌍 개수 N (서로 다른 각도 3~8°)",
    y_label="회전중심 RMSE [px]",
    legend=[("단순 평균", COLOR_AVG, False), ("공동 최소제곱(joint LS)", COLOR_LS, False), ("이론 1/√N", COLOR_MUTED, True)],
)

chart2 = build_loglog_svg(
    series=[
        {"points": angle_std, "color": COLOR_LS},
        {"points": angle_theory, "color": COLOR_MUTED, "dash": True, "markers": False},
    ],
    x_ticks=Ns, y_ticks=[0.010, 0.014, 0.018, 0.022, 0.026],
    x_label="같은 각도(5°) N회 반복촬영",
    y_label="각도 추정 표준편차 [deg]",
    legend=[("표준편차", COLOR_LS, False), ("이론 1/√N", COLOR_MUTED, True)],
)

chart3 = build_loglog_svg(
    series=[
        {"points": center_ls, "color": COLOR_LS},
        {"points": sameangle_center_rmse, "color": COLOR_AVG},
    ],
    x_ticks=Ns, y_ticks=[0.5, 1.0, 2.0, 4.0, 7.0],
    x_label="사용한 영상 장수 N",
    y_label="회전중심 RMSE [px]",
    legend=[("다른 각도 N장 (joint LS)", COLOR_LS, False), ("같은 각도(5°) N장 반복평균", COLOR_AVG, False)],
)

(OUT_DIR / "center_precision.svg").write_text(chart1, encoding="utf-8")
(OUT_DIR / "angle_precision.svg").write_text(chart2, encoding="utf-8")
(OUT_DIR / "comparison.svg").write_text(chart3, encoding="utf-8")
print("[OK] wrote charts/center_precision.svg, charts/angle_precision.svg, charts/comparison.svg")
