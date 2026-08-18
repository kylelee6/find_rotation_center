#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""리포트 HTML을 데이터로부터 조립한다 (숫자 하드코딩 대신 JSON을 직접 읽어 생성)."""
import base64
import json
import math
from pathlib import Path

HERE = Path(".")

center_data = json.loads((HERE / "multi_image_precision_result.json").read_text(encoding="utf-8"))
angle_data = json.loads((HERE / "angle_precision_result.json").read_text(encoding="utf-8"))


def b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


IMG_CLEAN_FULL = b64("images/noise_compare_clean_full.png")
IMG_NOISY_FULL = b64("images/noise_compare_noisy_full.png")
IMG_CLEAN_CROP = b64("images/noise_compare_clean_crop.png")
IMG_NOISY_CROP = b64("images/noise_compare_noisy_crop.png")


# ---------------- SVG log-log 라인차트 빌더 ----------------
def build_loglog_chart(series, x_ticks, y_ticks, x_label, y_label,
                        width=600, height=400, margin=(70, 30, 30, 50)):
    """series: list of dict(points=[(N, val), ...], color=..., dash=None)
    x_ticks: list of N tick values (log x-axis)
    y_ticks: list of y tick display values (log y-axis)
    margin: (left, right, top, bottom)
    """
    left, right, top, bottom = margin
    plot_w = width - left - right
    plot_h = height - top - bottom

    all_x = [n for s in series for n, v in s["points"]]
    all_y = [v for s in series for n, v in s["points"]]
    xmin_log, xmax_log = math.log10(min(x_ticks)), math.log10(max(x_ticks))
    ymin_log, ymax_log = math.log10(min(y_ticks)), math.log10(max(y_ticks))

    def xpix(n):
        return left + (math.log10(n) - xmin_log) / (xmax_log - xmin_log) * plot_w

    def ypix(v):
        return top + plot_h - (math.log10(v) - ymin_log) / (ymax_log - ymin_log) * plot_h

    svg = [f'<svg viewBox="0 0 {width} {height}" role="img">']

    # y gridlines
    for yt in y_ticks:
        yp = ypix(yt)
        svg.append(f'<line class="grid-line" x1="{left}" y1="{yp:.1f}" x2="{width-right}" y2="{yp:.1f}"/>')
        svg.append(f'<text class="tick-label" x="{left-8}" y="{yp+3:.1f}" text-anchor="end">{yt}</text>')

    # axes
    svg.append(f'<line class="axis-line" x1="{left}" y1="{top+plot_h}" x2="{width-right}" y2="{top+plot_h}"/>')
    svg.append(f'<line class="axis-line" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}"/>')

    # x ticks
    for xt in x_ticks:
        xp = xpix(xt)
        svg.append(f'<text class="tick-label" x="{xp:.1f}" y="{top+plot_h+16:.1f}" text-anchor="middle">{xt}</text>')

    svg.append(f'<text class="axis-label" x="{left+plot_w/2:.1f}" y="{height-8}" text-anchor="middle">{x_label}</text>')
    svg.append(f'<text class="axis-label" x="{-(top+plot_h/2):.1f}" y="18" transform="rotate(-90)" text-anchor="middle">{y_label}</text>')

    # series lines
    for s in series:
        pts = " ".join(f"{xpix(n):.1f},{ypix(v):.1f}" for n, v in s["points"])
        cls = s.get("cls", "line-generic")
        svg.append(f'<polyline class="{cls}" points="{pts}"/>')
        if s.get("markers", True):
            for n, v in s["points"]:
                svg.append(f'<circle class="{cls}-pt" cx="{xpix(n):.1f}" cy="{ypix(v):.1f}" r="4"/>')

    svg.append('</svg>')
    return "\n".join(svg)


# ---------------- 데이터 추출 ----------------
Ns = center_data["n_grid"]
cres = center_data["results_by_N"]
ares = angle_data["results_by_N"]

center_avg = [(n, cres[str(n)]["simple_average"]["rmse_2d"]) for n in Ns]
center_ls = [(n, cres[str(n)]["joint_least_squares"]["rmse_2d"]) for n in Ns]
baseline = center_ls[0][1]
center_theory = [(n, baseline / math.sqrt(n)) for n in Ns]

angle_std = [(n, ares[str(n)]["angle_deg"]["std"]) for n in Ns]
angle_rmse = [(n, math.sqrt(ares[str(n)]["angle_deg"]["bias"] ** 2 + ares[str(n)]["angle_deg"]["std"] ** 2)) for n in Ns]
angle_baseline_std = angle_std[0][1]
angle_theory = [(n, angle_baseline_std / math.sqrt(n)) for n in Ns]

sameangle_center_rmse = [(n, ares[str(n)]["rmse_center_2d"]) for n in Ns]

chart_center = build_loglog_chart(
    series=[
        {"points": center_avg, "cls": "line-avg"},
        {"points": center_ls, "cls": "line-ls"},
        {"points": center_theory, "cls": "line-theory", "markers": False},
    ],
    x_ticks=Ns, y_ticks=[0.6, 0.8, 1.0, 1.5, 2.0, 2.5],
    x_label="사용한 회전 영상 쌍 개수 N (log scale) — 서로 다른 각도(3~8°) 조합",
    y_label="회전중심 추정 RMSE [px] (log scale)",
)

chart_angle = build_loglog_chart(
    series=[
        {"points": angle_std, "cls": "line-ls"},
        {"points": angle_theory, "cls": "line-theory", "markers": False},
    ],
    x_ticks=Ns, y_ticks=[0.010, 0.014, 0.018, 0.022, 0.026],
    x_label="같은 각도(5°)를 N번 반복촬영",
    y_label="각도 추정 표준편차 [deg] (log scale)",
)

chart_compare = build_loglog_chart(
    series=[
        {"points": center_ls, "cls": "line-ls"},
        {"points": sameangle_center_rmse, "cls": "line-avg"},
    ],
    x_ticks=Ns, y_ticks=[0.5, 1.0, 2.0, 4.0, 7.0],
    x_label="사용한 영상 장수 N",
    y_label="회전중심 RMSE [px] (log scale)",
)


def fmt_table_rows_center():
    rows = []
    for n in Ns:
        a = cres[str(n)]["simple_average"]
        j = cres[str(n)]["joint_least_squares"]
        improve = baseline / j["rmse_2d"]
        rows.append(f'<tr><td>{n}</td><td>{a["cx"]["std"]:.3f}</td><td>{a["cy"]["std"]:.3f}</td>'
                    f'<td>{a["rmse_2d"]:.3f}</td><td>{j["cx"]["std"]:.3f}</td><td>{j["cy"]["std"]:.3f}</td>'
                    f'<td>{j["rmse_2d"]:.3f}</td><td>{improve:.2f}×</td></tr>')
    return "\n".join(rows)


def fmt_table_rows_angle():
    rows = []
    base_rmse = None
    for n in Ns:
        a = ares[str(n)]["angle_deg"]
        rmse = math.sqrt(a["bias"] ** 2 + a["std"] ** 2)
        if base_rmse is None:
            base_rmse = rmse
        improve = base_rmse / rmse
        crmse = ares[str(n)]["rmse_center_2d"]
        rows.append(f'<tr><td>{n}</td><td>{a["bias"]:+.5f}</td><td>{a["std"]:.5f}</td>'
                    f'<td>{rmse:.5f}</td><td>{improve:.2f}×</td><td>{crmse:.3f}</td></tr>')
    return "\n".join(rows)


def fmt_table_compare():
    rows = []
    for n in Ns:
        diverse = cres[str(n)]["joint_least_squares"]["rmse_2d"]
        same = ares[str(n)]["rmse_center_2d"]
        rows.append(f'<tr><td>{n}</td><td>{diverse:.3f}</td><td>{(center_ls[0][1]/diverse):.2f}×</td>'
                    f'<td>{same:.3f}</td><td>{(sameangle_center_rmse[0][1]/same):.2f}×</td></tr>')
    return "\n".join(rows)


html = f"""<title>영상 쌍 개수에 따른 회전중심·회전각 추정 정밀도</title>
<style>
  .viz-root {{
    color-scheme: light;
    --surface-1:      #fcfcfb;
    --page:           #f9f9f7;
    --text-primary:   #0b0b0b;
    --text-secondary: #52514e;
    --text-muted:     #898781;
    --grid:           #e1e0d9;
    --axis:           #c3c2b7;
    --series-1:       #2a78d6;
    --series-2:       #008300;
    --border:         rgba(11,11,11,0.10);
    --callout-bg:     #f0efec;
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    color: var(--text-primary);
    background: var(--page);
  }}
  @media (prefers-color-scheme: dark) {{
    :root:where(:not([data-theme="light"])) .viz-root {{
      color-scheme: dark;
      --surface-1:      #1a1a19;
      --page:           #0d0d0d;
      --text-primary:   #ffffff;
      --text-secondary: #c3c2b7;
      --text-muted:     #898781;
      --grid:           #2c2c2a;
      --axis:           #383835;
      --series-1:       #3987e5;
      --series-2:       #008300;
      --border:         rgba(255,255,255,0.10);
      --callout-bg:     #22221f;
    }}
  }}
  :root[data-theme="dark"] .viz-root {{
    color-scheme: dark;
    --surface-1:      #1a1a19;
    --page:           #0d0d0d;
    --text-primary:   #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted:     #898781;
    --grid:           #2c2c2a;
    --axis:           #383835;
    --series-1:       #3987e5;
    --series-2:       #008300;
    --border:         rgba(255,255,255,0.10);
    --callout-bg:     #22221f;
  }}

  .viz-root {{ max-width: 820px; margin: 0 auto; padding: 24px 20px 48px; }}
  h1 {{ font-size: 1.3rem; margin: 0 0 4px; }}
  h2 {{ font-size: 1.05rem; margin: 36px 0 8px; border-top: 1px solid var(--border); padding-top: 20px; }}
  .sub {{ color: var(--text-secondary); font-size: 0.88rem; margin: 0 0 20px; line-height: 1.55; }}
  .card {{
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 16px 8px;
    margin-bottom: 20px;
    overflow-x: auto;
  }}
  .callout {{
    background: var(--callout-bg);
    border-radius: 10px;
    padding: 14px 16px;
    font-size: 0.86rem;
    line-height: 1.6;
    color: var(--text-secondary);
    margin: 12px 0 20px;
  }}
  .callout .hl {{ color: var(--text-primary); font-weight: 600; }}
  svg {{ display: block; width: 100%; height: auto; min-width: 480px; }}
  .grid-line {{ stroke: var(--grid); stroke-width: 1; }}
  .axis-line {{ stroke: var(--axis); stroke-width: 1; }}
  .tick-label {{ fill: var(--text-muted); font-size: 11px; }}
  .axis-label {{ fill: var(--text-secondary); font-size: 12px; }}
  .line-avg {{ fill: none; stroke: var(--series-1); stroke-width: 2; }}
  .line-avg-pt {{ fill: var(--series-1); }}
  .line-ls {{ fill: none; stroke: var(--series-2); stroke-width: 2; }}
  .line-ls-pt {{ fill: var(--series-2); }}
  .line-theory {{ fill: none; stroke: var(--text-muted); stroke-width: 1.5; stroke-dasharray: 4 4; }}
  .legend {{ display: flex; gap: 18px; flex-wrap: wrap; margin: 4px 0 14px; font-size: 0.82rem; color: var(--text-secondary); }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; }}
  .swatch {{ width: 14px; height: 3px; border-radius: 2px; display: inline-block; }}
  .swatch.theory {{ background: none; border-top: 1.5px dashed var(--text-muted); width: 14px; height: 0; }}

  table {{ border-collapse: collapse; width: 100%; font-size: 0.83rem; }}
  caption {{ text-align: left; color: var(--text-secondary); font-size: 0.82rem; margin-bottom: 8px; }}
  th, td {{ text-align: right; padding: 7px 9px; border-bottom: 1px solid var(--grid); font-variant-numeric: tabular-nums; }}
  th:first-child, td:first-child {{ text-align: left; }}
  thead th {{ color: var(--text-secondary); font-weight: 600; font-size: 0.76rem; }}
  tbody tr:last-child td {{ border-bottom: none; font-weight: 600; color: var(--text-primary); }}
  .note {{ color: var(--text-secondary); font-size: 0.82rem; line-height: 1.6; margin-top: 4px; }}
  .hl {{ color: var(--text-primary); font-weight: 600; }}

  .img-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
  .img-grid figure {{ margin: 0; }}
  .img-grid img {{ width: 100%; border-radius: 8px; border: 1px solid var(--border); display: block; }}
  .img-grid figcaption {{ font-size: 0.78rem; color: var(--text-muted); margin-top: 6px; text-align: center; }}
</style>

<div class="viz-root">
  <h1>영상 쌍 개수에 따른 회전중심 · 회전각 추정 정밀도</h1>
  <p class="sub">
    test.bmp 1장을 참 장면으로 두고 참 회전중심 (1234.5, 980.2) px를 기준으로 시뮬레이션.
    <code>estimate_rotation_center.py</code>의 실제 angle_search()+refine() 함수를 그대로 재사용해
    각 (기준영상, 회전영상) 쌍을 추정. 아래 모든 실험은 픽셀노이즈 σ=0.01(0~1 스케일),
    밝기 gain 변동 σ=0.02, offset 변동 σ=0.01, 8bit 양자화를 가한 <span class="hl">노이즈 영상</span>에 대한 결과다.
  </p>

  <h2>0. 어느 정도의 노이즈를 가했는가</h2>
  <div class="card">
    <div class="img-grid">
      <figure><img src="data:image/png;base64,{IMG_CLEAN_FULL}" alt="노이즈 적용 전 원본"><figcaption>노이즈 적용 전 (원본 test.bmp)</figcaption></figure>
      <figure><img src="data:image/png;base64,{IMG_NOISY_FULL}" alt="노이즈 적용 후"><figcaption>노이즈 적용 후 (1회 촬영 시뮬레이션)</figcaption></figure>
      <figure><img src="data:image/png;base64,{IMG_CLEAN_CROP}" alt="확대: 노이즈 전"><figcaption>중앙부 확대(3배) — 노이즈 전</figcaption></figure>
      <figure><img src="data:image/png;base64,{IMG_NOISY_CROP}" alt="확대: 노이즈 후"><figcaption>중앙부 확대(3배) — 노이즈 후</figcaption></figure>
    </div>
    <p class="note" style="margin-top:14px">
      노이즈 모델: 가우시안 픽셀노이즈 σ=0.01(0~1 스케일, 8bit 기준 약 2.55 gray level) + 밝기 gain~N(1,0.02) +
      offset~N(0,0.01) + 8bit 양자화. 실측 결과 두 영상의 평균 절대 밝기차는 <span class="hl">2.07 gray level</span>
      (표준편차 1.79), 육안으로는 미세한 그레인 정도지만 서브픽셀 정합 알고리즘 입장에서는 상당한 교란이다.
    </p>
  </div>

  <div class="callout">
    <span class="hl">Q. 왜 이전 truth.json 테스트(서브픽셀 정확도)보다 정확도가 크게 떨어졌나?</span><br/>
    이전 truth.json 테스트는 노이즈가 전혀 없는 순수 수학적 워프(cv2.warpAffine)였다 — 두 영상 사이 유일한 오차원은
    큐빅 보간 리샘플링 오차뿐이라 photometric least-squares가 거의 완벽히 역산해 <span class="hl">0.1~0.2px</span> 수준이 나왔다.
    반면 이번 실험들은 위 노이즈를 실제로 주입한 <span class="hl">현실적 반복촬영 조건</span>이라 1장만으로는 px~10px대까지
    흔들린다. "영상을 여러 장 쓰면 좋아지는가"라는 질문 자체가 노이즈의 존재를 전제로 하므로, 이하의 모든 실험은 노이즈를
    켠 상태로 수행했다.
  </div>

  <h2>1. 회전중심 정밀도 — 서로 다른 각도(3~8°) N장을 삼각측량</h2>
  <div class="card">
    <div class="legend">
      <span class="legend-item"><span class="swatch" style="background:var(--series-1)"></span>단순 평균 (N개 개별 중심 평균)</span>
      <span class="legend-item"><span class="swatch" style="background:var(--series-2)"></span>공동 최소제곱 (joint LS)</span>
      <span class="legend-item"><span class="swatch theory"></span>이론: RMSE(N=1) / √N</span>
    </div>
    {chart_center}
  </div>
  <div class="card">
    <table>
      <caption>N별 회전중심 추정 정밀도 (30회 몬테카를로)</caption>
      <thead><tr><th>N</th><th>평균 std_x</th><th>평균 std_y</th><th>평균 RMSE</th><th>LS std_x</th><th>LS std_y</th><th>LS RMSE</th><th>개선배율</th></tr></thead>
      <tbody>{fmt_table_rows_center()}</tbody>
    </table>
    <p class="note">단위: px. 개선배율 = RMSE(N=1) / RMSE(joint LS, N).</p>
  </div>

  <h2>2. 회전각 정밀도 — 같은 각도(5°)를 N회 반복촬영 후 평균</h2>
  <p class="sub">
    위 실험은 매 영상마다 각도가 달라 "각도 자체의 반복정밀도"는 잴 수 없다. 그래서 여기서는 정확히 같은 5°를
    N번 독립적으로 촬영해 각도 추정치를 평균하는 고전적 반복측정 시나리오를 별도로 시뮬레이션했다.
  </p>
  <div class="card">
    <div class="legend">
      <span class="legend-item"><span class="swatch" style="background:var(--series-2)"></span>각도 표준편차 (N회 평균)</span>
      <span class="legend-item"><span class="swatch theory"></span>이론: std(N=1) / √N</span>
    </div>
    {chart_angle}
  </div>
  <div class="card">
    <table>
      <caption>N별 회전각(5° 고정) 추정 정밀도 (30회 몬테카를로)</caption>
      <thead><tr><th>N</th><th>편향(bias)[deg]</th><th>표준편차[deg]</th><th>RMSE[deg]</th><th>개선배율</th><th>동일조건 중심 RMSE[px]</th></tr></thead>
      <tbody>{fmt_table_rows_angle()}</tbody>
    </table>
    <p class="note">
      각도 표준편차는 N=1→12에서 0.0229°→0.0109°로 줄지만(약 2.1배), <span class="hl">편향은 0.010~0.013°에서 거의 줄지 않는다</span>.
      그 결과 RMSE는 N=8 이후 0.0166°에서 더 개선되지 않는 편향 바닥(bias floor)에 도달한다.
    </p>
  </div>

  <h2>3. 결정적 비교 — "같은 각도 반복" vs "다른 각도 삼각측량"</h2>
  <div class="card">
    <div class="legend">
      <span class="legend-item"><span class="swatch" style="background:var(--series-2)"></span>다른 각도 N장 (joint LS)</span>
      <span class="legend-item"><span class="swatch" style="background:var(--series-1)"></span>같은 각도(5°) N장 반복평균</span>
    </div>
    {chart_compare}
  </div>
  <div class="card">
    <table>
      <caption>동일한 N(영상 장수)에서 회전중심 RMSE 비교</caption>
      <thead><tr><th>N</th><th>다른 각도 RMSE[px]</th><th>개선배율</th><th>같은 각도 RMSE[px]</th><th>개선배율</th></tr></thead>
      <tbody>{fmt_table_compare()}</tbody>
    </table>
  </div>

  <p class="callout">
    <span class="hl">핵심 결론:</span> 같은 개수(N=12)의 영상을 쓰더라도 <span class="hl">어떤 각도로 찍느냐가 몇 장을 찍느냐만큼 중요하다</span>.
    같은 5°를 12번 반복 촬영하면 RMSE가 6.38→4.31px (1.48배 개선)에 그치는 반면, 3~8° 범위에서 서로 다른 각도로 12장을
    찍어 공동 최소제곱으로 결합하면 6.38(사실상 동일 N=1 기준)px가 아니라 2.56→0.56px (4.54배 개선)까지 좋아진다.
    이유는 <code>center_from_angle_translation</code>이 푸는 2×2 선형계 [[1-cosθ,-sinθ],[sinθ,1-cosθ]]가 <span class="hl">고정된 각도에서는 항상 같은 조건수를 갖는 계통오차(=편향)를 남기지만</span>,
    각도가 다양해지면 이 계통오차의 방향과 크기가 매 영상마다 달라져 평균 과정에서 서로 상쇄되기 때문이다.
    반대로 회전각 자체는 애초에 이런 행렬 역산이 필요 없는 직접 관측량이라, 같은 각도를 반복 촬영하는 것만으로도
    표준편차가 착실히 줄어든다(다만 편향 바닥은 여전히 존재).
  </p>
</div>
"""

out_path = HERE / "rotation_center_multi_image_report.html"
out_path.write_text(html, encoding="utf-8")
print(f"[OK] wrote {out_path} ({len(html)} chars)")
