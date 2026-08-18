#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
simulate_angle_precision.py

질문: 노이즈가 있는 영상에서, 같은 회전각(예: 5도)을 N번 반복 촬영해서
      평균 내면 각도 추정 정밀도(및 중심 정밀도)가 1장짜리 단일 쌍에 비해
      얼마나 좋아지는가?

simulate_multi_image_precision.py 와의 차이:
- 그 스크립트는 "서로 다른 각도(3~8도)로 N장 찍어서 회전중심을 삼각측량"하는
  실험이었다 (angle이 매 장마다 다름 -> 각도 자체의 반복정밀도는 잴 수 없음).
- 이 스크립트는 "정확히 같은 각도(5도)를 N번 반복 촬영"하는 고전적인
  반복측정 평균 시나리오다. 같은 참값을 N번 독립적으로 측정하므로
  각도(angle)와 중심(cx,cy) 모두에 대해 표준 반복측정 평균 효과(~1/sqrt(N))를
  직접 측정할 수 있다.

기준영상(reference)은 트라이얼마다 1장만 촬영하고, target은 같은 각도로
MAX_N장까지 독립 노이즈로 촬영한다. 노이즈 모델은
simulate_multi_image_precision.py와 동일하게 재사용한다.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

import estimate_rotation_center as erc
import simulate_multi_image_precision as base_sim  # 노이즈모델/추정 파라미터/함수 재사용

HERE = Path(".")

TRUE_CX = base_sim.TRUE_CX
TRUE_CY = base_sim.TRUE_CY
TRUE_ANGLE = 5.0  # README 권장 단일-쌍 기본 각도

MAX_N = 12
N_GRID = [1, 2, 3, 5, 8, 12]
R_TRIALS = 30


def main():
    t_start = time.time()
    base = erc.preprocess(erc.read_gray(HERE / "test.bmp"), base_sim.BLUR_SIGMA)
    mask = erc.read_mask(None, base.shape)

    M_true = cv2.getRotationMatrix2D((TRUE_CX, TRUE_CY), TRUE_ANGLE, 1.0).astype(np.float64)

    per_trial = []
    rng_master = np.random.default_rng(54321)

    for r in range(R_TRIALS):
        trial_seed = int(rng_master.integers(0, 2**31 - 1))
        rng = np.random.default_rng(trial_seed)

        ref_shot = base_sim.make_shot(base, None, rng)

        estimates = []
        for i in range(MAX_N):
            target_shot = base_sim.make_shot(base, M_true, rng)
            est = base_sim.single_pair_estimate(ref_shot, target_shot, mask)
            estimates.append(est)

        per_trial.append(estimates)
        elapsed = time.time() - t_start
        print(f"[trial {r+1}/{R_TRIALS}] done, elapsed={elapsed:.1f}s", flush=True)

    def stats(arr, true_val):
        arr = np.array(arr)
        err = arr - true_val
        return {
            "mean": float(np.mean(arr)),
            "bias": float(np.mean(err)),
            "std": float(np.std(arr, ddof=1)),
        }

    results = {}
    for N in N_GRID:
        angle_avg, cx_avg, cy_avg = [], [], []
        for estimates in per_trial:
            subset = estimates[:N]
            angle_avg.append(float(np.mean([e["angle"] for e in subset])))
            cx_avg.append(float(np.mean([e["cx"] for e in subset])))
            cy_avg.append(float(np.mean([e["cy"] for e in subset])))

        angle_stats = stats(angle_avg, TRUE_ANGLE)
        cx_stats = stats(cx_avg, TRUE_CX)
        cy_stats = stats(cy_avg, TRUE_CY)
        rmse_center = float(np.sqrt(np.mean((np.array(cx_avg) - TRUE_CX) ** 2 + (np.array(cy_avg) - TRUE_CY) ** 2)))

        results[N] = {
            "angle_deg": angle_stats,
            "cx": cx_stats,
            "cy": cy_stats,
            "rmse_center_2d": rmse_center,
        }

    baseline_angle_std = results[1]["angle_deg"]["std"]
    baseline_center_rmse = results[1]["rmse_center_2d"]

    print("\n===== 동일 각도(5도) N회 반복촬영 평균 정밀도 =====")
    print(f"{'N':>3} | {'angle std[deg]':>14} {'angle RMSE[deg]':>16} | "
          f"{'center std_x':>12} {'center std_y':>12} {'center RMSE':>12} | {'1/sqrt(N) 예측(angle)':>20}")
    for N in N_GRID:
        a = results[N]["angle_deg"]
        cx = results[N]["cx"]
        cy = results[N]["cy"]
        angle_rmse = float(np.sqrt(a["bias"] ** 2 + a["std"] ** 2))
        pred = baseline_angle_std / np.sqrt(N)
        print(f"{N:>3} | {a['std']:>14.6f} {angle_rmse:>16.6f} | "
              f"{cx['std']:>12.4f} {cy['std']:>12.4f} {results[N]['rmse_center_2d']:>12.4f} | {pred:>20.6f}")

    out = {
        "true_center": {"cx": TRUE_CX, "cy": TRUE_CY},
        "true_angle_deg": TRUE_ANGLE,
        "n_grid": N_GRID,
        "r_trials": R_TRIALS,
        "noise_model": {
            "pixel_noise_sigma": base_sim.PIXEL_NOISE_SIGMA,
            "gain_std": base_sim.GAIN_STD,
            "offset_std": base_sim.OFFSET_STD,
            "bit_depth": 8,
        },
        "results_by_N": {str(k): v for k, v in results.items()},
    }
    out_path = HERE / "angle_precision_result.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    total_elapsed = time.time() - t_start
    print(f"\n[OK] 결과 저장: {out_path}")
    print(f"[OK] 총 소요시간: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
