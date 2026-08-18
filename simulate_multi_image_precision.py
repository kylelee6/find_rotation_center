#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
simulate_multi_image_precision.py

질문: 회전중심 추정에 영상 1쌍만 쓰는 것과 여러 쌍(N장)을 쓰는 것 중
      어느 쪽이 얼마나 더 정밀한가?

방법:
1. test.bmp를 "참 장면"으로 삼고, 알려진 참 회전중심 (cx0, cy0)을 정한다.
2. 매 트라이얼마다 기준영상(reference) 1장을 노이즈를 섞어 "촬영"하고,
   3~8도 / -3~-8도 범위에서 서로 다른 각도로 회전시킨 target 영상을
   최대 MAX_N장까지 각각 독립 노이즈로 "촬영"한다.
   노이즈 = 가우시안 픽셀노이즈 + 밝기 gain/offset 변동 + 8bit 양자화.
3. 각 (reference, target_i) 쌍에 대해 estimate_rotation_center.py의
   실제 angle_search()+refine() 함수를 그대로 호출해 회전중심을 추정한다
   (기존 스크립트 로직을 그대로 재사용 - 별도 알고리즘 아님).
4. N장을 사용할 때:
   (a) N개의 개별 추정 중심을 단순 평균
   (b) N개의 (angle, tx, ty)를 모아 공유 중심에 대해 선형최소제곱으로 공동추정
   두 방식의 표준편차(=정밀도)가 N에 따라 어떻게 줄어드는지 비교한다.
5. R회 반복(Monte Carlo)해서 각 N에서의 편향(bias)/표준편차(std)/RMSE를 구하고
   1/sqrt(N) 이론곡선과 비교한다.

주의: 여기서 만드는 "여러 장"은 서로 다른 회전각(3~8도 범위)으로 찍은
      독립된 관측치들이다. 즉 "같은 각도를 반복 촬영"이 아니라
      "다른 각도로 여러 번 촬영해서 정보를 합치는" 시나리오를 시뮬레이션한다.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

import estimate_rotation_center as erc

HERE = Path(".")  # Windows/Git-Bash 한글 경로 인코딩 문제 회피 (cwd 상대경로만 사용)

TRUE_CX = 1234.5
TRUE_CY = 980.2

ANGLE_POOL = [3, 4, 5, 6, 7, 8, -3, -4, -5, -6, -7, -8]  # len == MAX_N
MAX_N = len(ANGLE_POOL)
N_GRID = [1, 2, 3, 5, 8, 12]
R_TRIALS = 30

# 추정 파라미터 (속도를 위해 기본값보다 성기게 탐색하되, 참값을 참조하지 않음)
COARSE_STEP = 0.5
FINE_RANGE = 0.25
FINE_STEP = 0.05
MAX_POINTS = 3000
BLUR_SIGMA = 0.8

# 촬영 노이즈 모델
PIXEL_NOISE_SIGMA = 0.010   # 0~1 스케일, 8bit 기준 약 2.5 grayscale
GAIN_STD = 0.02             # 밝기 gain 변동
OFFSET_STD = 0.01           # 밝기 offset 변동


def make_shot(base: np.ndarray, M: np.ndarray | None, rng: np.random.Generator) -> np.ndarray:
    """base(참 장면)에서 카메라로 한 장 '촬영'한 것을 시뮬레이션."""
    h, w = base.shape
    img = base if M is None else erc.warp_image(base, M, (w, h))

    gain = float(rng.normal(1.0, GAIN_STD))
    offset = float(rng.normal(0.0, OFFSET_STD))
    noise = rng.normal(0.0, PIXEL_NOISE_SIGMA, size=img.shape).astype(np.float32)

    shot = img * gain + offset + noise
    shot = np.clip(shot, 0.0, 1.0)
    shot = np.round(shot * 255.0) / 255.0  # 8bit 양자화
    return shot.astype(np.float32)


def single_pair_estimate(ref_img: np.ndarray, target_img: np.ndarray, mask: np.ndarray):
    score, phase_resp, search_angle, M0 = erc.angle_search(
        ref_img, target_img, -10.0, 10.0, COARSE_STEP, FINE_RANGE, FINE_STEP
    )
    p, rmse, valid_fraction = erc.refine(ref_img, target_img, mask, M0, MAX_POINTS, seed=0)
    angle, tx, ty, gain, offset = map(float, p)
    cx, cy = erc.center_from_angle_translation(angle, tx, ty)
    return {"angle": angle, "tx": tx, "ty": ty, "cx": cx, "cy": cy, "rmse": rmse}


def joint_ls_center(triples):
    """여러 (angle, tx, ty)로부터 공유 회전중심 (cx,cy)을 선형최소제곱으로 공동추정."""
    rows = []
    rhs = []
    for angle_deg, tx, ty in triples:
        theta = np.radians(angle_deg)
        c, s = np.cos(theta), np.sin(theta)
        rows.append([1 - c, -s])
        rhs.append(tx)
        rows.append([s, 1 - c])
        rhs.append(ty)
    A = np.array(rows, dtype=np.float64)
    b = np.array(rhs, dtype=np.float64)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(sol[0]), float(sol[1])


def main():
    t_start = time.time()
    base = erc.preprocess(erc.read_gray(HERE / "test.bmp"), BLUR_SIGMA)
    h, w = base.shape
    mask = erc.read_mask(None, base.shape)

    M_true_by_angle = {
        a: cv2.getRotationMatrix2D((TRUE_CX, TRUE_CY), float(a), 1.0).astype(np.float64)
        for a in ANGLE_POOL
    }

    # per_trial[r] = list of per-image estimate dicts, in a random order of ANGLE_POOL
    per_trial = []

    rng_master = np.random.default_rng(12345)

    for r in range(R_TRIALS):
        trial_seed = int(rng_master.integers(0, 2**31 - 1))
        rng = np.random.default_rng(trial_seed)

        ref_shot = make_shot(base, None, rng)

        order = rng.permutation(len(ANGLE_POOL))
        angles_this_trial = [ANGLE_POOL[i] for i in order]

        estimates = []
        for a in angles_this_trial:
            target_shot = make_shot(base, M_true_by_angle[a], rng)
            est = single_pair_estimate(ref_shot, target_shot, mask)
            est["true_angle"] = a
            estimates.append(est)

        per_trial.append(estimates)
        elapsed = time.time() - t_start
        print(f"[trial {r+1}/{R_TRIALS}] done, elapsed={elapsed:.1f}s", flush=True)

    # ---- N별 결과 집계 ----
    results = {}
    for N in N_GRID:
        cx_avg, cy_avg = [], []
        cx_ls, cy_ls = [], []
        single_cx, single_cy = [], []  # N=1일 때 개별 추정값 분포(=참조용, N=1과 동일)

        for estimates in per_trial:
            subset = estimates[:N]
            cxs = [e["cx"] for e in subset]
            cys = [e["cy"] for e in subset]
            cx_avg.append(float(np.mean(cxs)))
            cy_avg.append(float(np.mean(cys)))

            triples = [(e["angle"], e["tx"], e["ty"]) for e in subset]
            if N == 1:
                cx_j, cy_j = cxs[0], cys[0]
            else:
                cx_j, cy_j = joint_ls_center(triples)
            cx_ls.append(cx_j)
            cy_ls.append(cy_j)

            if N == 1:
                single_cx.append(cxs[0])
                single_cy.append(cys[0])

        def stats(arr, true_val):
            arr = np.array(arr)
            err = arr - true_val
            return {
                "mean": float(np.mean(arr)),
                "bias": float(np.mean(err)),
                "std": float(np.std(arr, ddof=1)),
            }

        ex_avg = stats(cx_avg, TRUE_CX)
        ey_avg = stats(cy_avg, TRUE_CY)
        ex_ls = stats(cx_ls, TRUE_CX)
        ey_ls = stats(cy_ls, TRUE_CY)

        rmse_avg = float(np.sqrt(np.mean((np.array(cx_avg) - TRUE_CX) ** 2 + (np.array(cy_avg) - TRUE_CY) ** 2)))
        rmse_ls = float(np.sqrt(np.mean((np.array(cx_ls) - TRUE_CX) ** 2 + (np.array(cy_ls) - TRUE_CY) ** 2)))

        results[N] = {
            "simple_average": {"cx": ex_avg, "cy": ey_avg, "rmse_2d": rmse_avg},
            "joint_least_squares": {"cx": ex_ls, "cy": ey_ls, "rmse_2d": rmse_ls},
        }

    baseline_std_x = results[1]["joint_least_squares"]["cx"]["std"]
    baseline_std_y = results[1]["joint_least_squares"]["cy"]["std"]
    baseline_rmse = results[1]["joint_least_squares"]["rmse_2d"]

    print("\n===== N (사용 영상 쌍 수) 별 회전중심 추정 정밀도 =====")
    print(f"{'N':>3} | {'avg std_x':>10} {'avg std_y':>10} {'avg RMSE':>10} || "
          f"{'LS std_x':>10} {'LS std_y':>10} {'LS RMSE':>10} | {'1/sqrt(N) 예측':>14} | {'개선배율(RMSE)':>12}")
    for N in N_GRID:
        a = results[N]["simple_average"]
        j = results[N]["joint_least_squares"]
        pred = baseline_rmse / np.sqrt(N)
        improve = baseline_rmse / j["rmse_2d"] if j["rmse_2d"] > 0 else float("nan")
        print(f"{N:>3} | {a['cx']['std']:>10.4f} {a['cy']['std']:>10.4f} {a['rmse_2d']:>10.4f} || "
              f"{j['cx']['std']:>10.4f} {j['cy']['std']:>10.4f} {j['rmse_2d']:>10.4f} | "
              f"{pred:>14.4f} | {improve:>11.2f}x")

    out = {
        "true_center": {"cx": TRUE_CX, "cy": TRUE_CY},
        "angle_pool_deg": ANGLE_POOL,
        "n_grid": N_GRID,
        "r_trials": R_TRIALS,
        "noise_model": {
            "pixel_noise_sigma": PIXEL_NOISE_SIGMA,
            "gain_std": GAIN_STD,
            "offset_std": OFFSET_STD,
            "bit_depth": 8,
        },
        "estimator_params": {
            "coarse_step": COARSE_STEP,
            "fine_range": FINE_RANGE,
            "fine_step": FINE_STEP,
            "max_points": MAX_POINTS,
            "blur_sigma": BLUR_SIGMA,
        },
        "results_by_N": {str(k): v for k, v in results.items()},
    }
    out_path = HERE / "multi_image_precision_result.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    total_elapsed = time.time() - t_start
    print(f"\n[OK] 결과 저장: {out_path}")
    print(f"[OK] 총 소요시간: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
