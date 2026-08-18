# 회전중심 시뮬레이션 및 추정

🔗 **웹에서 바로 보기**: [프로젝트 소개](https://kylelee6.github.io/find_rotation_center/) ·
[웹 도구로 직접 실행](https://kylelee6.github.io/find_rotation_center/tool.html) ·
[정밀도 분석 리포트](https://kylelee6.github.io/find_rotation_center/report.html)

## 설치

```bash
pip install -r requirements.txt
```

## 1. 지정한 중심으로 영상 회전

```bash
python rotate_image_about_center.py input.png rotated.png ^
  --angle 5.0 ^
  --cx 1234.5 ^
  --cy 980.2 ^
  --save-matrix truth.json
```

PowerShell에서는 `^` 대신 한 줄로 실행하거나 백틱을 사용해도 됩니다.

권장 설정:
- `--interpolation cubic`
- `--border-mode reflect`

## 2. 두 영상에서 회전각과 중심 추정

```bash
python estimate_rotation_center.py input.png rotated.png --output-dir result_5deg
```

결과:
- `result.json`
- `result.csv`
- `aligned.png`
- `difference_x5.png`
- `overlay_reference_red_aligned_green.png`

## 주의

두 영상 사이에 회전과 무관한 자유 평행이동이 추가되면, 한 쌍만으로 회전중심과 평행이동을 분리할 수 없습니다.
이 프로그램은 두 영상의 차이를 순수한 2D 회전으로 가정합니다.

처음 시험할 때는 다음 순서가 좋습니다.

1. 원본 이미지에서 임의의 중심으로 정확히 5도 회전 영상을 생성
2. 생성 영상과 원본으로 추정 프로그램 실행
3. `truth.json`과 `result.json` 비교
4. +1, +2, +3, +5, -1, -2, -3, -5도 등으로 반복
5. 추정 중심의 평균과 표준편차 계산

회전각이 너무 작으면 중심 오차가 커집니다.
일반적으로 3~8도 범위가 단일 쌍 시험에 유리합니다.

## 3. 영상을 여러 장 쓰면 얼마나 정밀해지는가

위 "주의" 항목의 5번(반복 시험 후 평균/표준편차 계산)을 실제로 수치화한 몬테카를로 시뮬레이션이다.
카메라 노이즈가 섞인 조건에서 영상 1쌍만 쓸 때와 여러 장을 결합할 때의 회전중심·회전각 정밀도를
비교했다. 핵심 결론만 요약하면:

- 노이즈가 없으면 1쌍만으로 이미 서브픽셀(~0.05~0.2px)이다 — "여러 장이 도움되는가"라는
  질문 자체가 현실적 카메라 노이즈를 전제로 한다.
- **회전중심**은 몇 장을 찍느냐보다 **어떤 각도 조합으로 찍느냐**가 더 중요하다. 같은 각도를
  12번 반복하면 1.48배 개선에 그치지만, 3~8° 범위 서로 다른 각도 12장을 joint LS로 결합하면
  4.54배 개선된다.
- **회전각**은 반복촬영 평균으로 표준편차가 꾸준히 좋아지지만(N=12에서 ~2.1배), 편향은
  ~0.01~0.02° 근처에서 구조적으로 잘 줄지 않는다(gain/offset 동시 피팅 + 8bit 양자화의 영향).

전체 방법론, 노이즈 전/후 비교 이미지, 차트, 상세 수치는 **[REPORT.md](REPORT.md)** 참고.

스크립트: `simulate_multi_image_precision.py`(다른 각도 조합), `simulate_angle_precision.py`(같은 각도 반복),
`make_charts.py`(차트 생성), `build_report.py`(인터랙티브 HTML 리포트 생성)

## 4. 웹에서 바로 실행해보기

파이썬 설치 없이 브라우저에서 바로 두 영상을 올려 회전각·중심을 계산해볼 수 있다:
**[tool.html](https://kylelee6.github.io/find_rotation_center/tool.html)** — 모든 계산은
OpenCV.js(WebAssembly)로 브라우저 안에서 실행되며, 이미지는 서버로 전송되지 않는다.

웹 버전은 `estimate_rotation_center.py`의 핵심 아이디어(각도 탐색 → 평행이동 정합 → 중심 역산)를
JavaScript로 재현한 시연용 버전이다. `matchTemplate`(정규상관) 기반 평행이동 탐색을 쓰고
photometric least-squares 정밀화 단계는 생략했기 때문에, 파이썬 버전보다 정밀도가 낮다
(예제 기준 오차 약 0.05°/12px 수준). 연구용 정밀 분석에는 파이썬 스크립트를 사용할 것.
