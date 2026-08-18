#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
estimate_rotation_center_robust.py

두 영상 사이의 순수 2D 회전을 다음 순서로 추정한다.

1. 각도 coarse-to-fine 탐색
2. 각 후보 각도마다 phase correlation로 평행이동 추정
3. overlap NCC로 최적 후보 선택
4. angle, tx, ty, gain, offset을 photometric least-squares로 정밀화
5. rigid transform의 angle/translation으로부터 회전중심 계산

영상 밖에 회전중심이 있어 큰 평행이동처럼 보이는 경우에도
단위행렬 초기값 하나만 쓰는 ECC보다 훨씬 안정적이다.

예:
python estimate_rotation_center_robust.py test.bmp rotated_5deg.png ^
    --output-dir result_5deg --angle-min -10 --angle-max 10
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("reference", type=Path)
    p.add_argument("rotated", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("rotation_result"))
    p.add_argument("--angle-min", type=float, default=-10.0)
    p.add_argument("--angle-max", type=float, default=10.0)
    p.add_argument("--coarse-step", type=float, default=0.25)
    p.add_argument("--fine-step", type=float, default=0.02)
    p.add_argument("--fine-range", type=float, default=0.5)
    p.add_argument("--max-points", type=int, default=30000)
    p.add_argument("--blur-sigma", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mask", type=Path, default=None)
    return p.parse_args()


def read_gray(path: Path) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(path)
    return im.astype(np.float32) / 255.0


def normalize(im: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(im, [1, 99])
    return np.clip((im - lo) / max(float(hi - lo), 1e-6), 0, 1).astype(np.float32)


def preprocess(im: np.ndarray, sigma: float) -> np.ndarray:
    im = normalize(im)
    if sigma > 0:
        im = cv2.GaussianBlur(im, (0, 0), sigma)
    return im


def read_mask(path: Path | None, shape):
    h, w = shape
    if path is None:
        m = np.ones(shape, np.uint8)
        b = max(5, int(round(min(h, w) * 0.01)))
        m[:b] = 0
        m[-b:] = 0
        m[:, :b] = 0
        m[:, -b:] = 0
        return m
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    if m.shape != shape:
        raise ValueError("mask size mismatch")
    return (m > 0).astype(np.uint8)


def rotation_matrix_about_image_center(angle_deg, w, h):
    return cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0).astype(np.float64)


def warp_image(im, M, size, interpolation=cv2.INTER_CUBIC):
    return cv2.warpAffine(
        im, M, size,
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def overlap_ncc(ref, target, M):
    h, w = ref.shape
    pred = warp_image(ref, M, (w, h))
    valid = warp_image(
        np.ones_like(ref, np.float32), M, (w, h), cv2.INTER_NEAREST
    ) > 0.5

    # 지나치게 어두운 zero padding은 점수에서 제외
    if valid.sum() < 1000:
        return -1.0

    a = pred[valid].astype(np.float64)
    b = target[valid].astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / max(den, 1e-12))


def estimate_shift_for_angle(ref, target, angle_deg):
    h, w = ref.shape
    M = rotation_matrix_about_image_center(angle_deg, w, h)
    rotated_ref = warp_image(ref, M, (w, h))

    # Hanning window로 경계 효과 완화
    window = cv2.createHanningWindow((w, h), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(
        rotated_ref * window,
        target * window
    )

    M[0, 2] += shift[0]
    M[1, 2] += shift[1]
    score = overlap_ncc(ref, target, M)
    return M, score, float(response)


def angle_search(ref, target, amin, amax, coarse_step, fine_range, fine_step):
    best = None

    coarse_angles = np.arange(amin, amax + coarse_step * 0.5, coarse_step)
    for a in coarse_angles:
        M, score, response = estimate_shift_for_angle(ref, target, float(a))
        candidate = (score, response, float(a), M)
        if best is None or candidate[0] > best[0]:
            best = candidate

    center_angle = best[2]
    fine_angles = np.arange(
        center_angle - fine_range,
        center_angle + fine_range + fine_step * 0.5,
        fine_step,
    )
    for a in fine_angles:
        M, score, response = estimate_shift_for_angle(ref, target, float(a))
        candidate = (score, response, float(a), M)
        if candidate[0] > best[0]:
            best = candidate

    return best


def transform_points(x, y, angle_deg, tx, ty):
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    xp = c * x + s * y + tx
    yp = -s * x + c * y + ty
    return xp, yp


def sample_remap(im, x, y, chunk=30000):
    x = np.asarray(x, np.float32).reshape(-1)
    y = np.asarray(y, np.float32).reshape(-1)
    out = np.empty(x.size, np.float64)
    for start in range(0, x.size, chunk):
        end = min(start + chunk, x.size)
        vals = cv2.remap(
            im,
            x[start:end].reshape(-1, 1),
            y[start:end].reshape(-1, 1),
            interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        out[start:end] = vals.reshape(-1)
    return out


def matrix_angle_translation(M):
    angle = math.degrees(math.atan2(float(M[0, 1]), float(M[0, 0])))
    return angle, float(M[0, 2]), float(M[1, 2])


def refine(ref, target, mask, M0, max_points, seed):
    h, w = ref.shape
    yy, xx = np.nonzero(mask > 0)
    rng = np.random.default_rng(seed)
    if len(xx) > max_points:
        ids = rng.choice(len(xx), max_points, replace=False)
        xx, yy = xx[ids], yy[ids]

    x = xx.astype(np.float64)
    y = yy.astype(np.float64)
    refv = ref[yy, xx].astype(np.float64)

    a0, tx0, ty0 = matrix_angle_translation(M0)
    p0 = np.array([a0, tx0, ty0, 1.0, 0.0], np.float64)

    lower = np.array([a0 - 1.0, tx0 - 20.0, ty0 - 20.0, 0.5, -0.5])
    upper = np.array([a0 + 1.0, tx0 + 20.0, ty0 + 20.0, 1.5, 0.5])

    def fun(p):
        angle, tx, ty, gain, offset = p
        xp, yp = transform_points(x, y, angle, tx, ty)
        sampled = sample_remap(target, xp, yp)

        valid = (
            (xp >= 2) & (xp <= w - 3) &
            (yp >= 2) & (yp <= h - 3)
        )
        r = gain * sampled + offset - refv
        # 유효영역 밖 표본에는 고정 penalty
        r[~valid] = 0.5
        return r

    result = least_squares(
        fun,
        p0,
        bounds=(lower, upper),
        method="trf",
        loss="huber",
        f_scale=0.03,
        x_scale=np.array([1.0, 100.0, 100.0, 1.0, 0.1]),
        max_nfev=150,
    )

    p = result.x
    xp, yp = transform_points(x, y, p[0], p[1], p[2])
    valid = (
        (xp >= 2) & (xp <= w - 3) &
        (yp >= 2) & (yp <= h - 3)
    )
    sampled = sample_remap(target, xp, yp)
    err = p[3] * sampled + p[4] - refv
    rmse = float(np.sqrt(np.mean(err[valid] ** 2)))
    return p, rmse, float(valid.mean())


def center_from_angle_translation(angle_deg, tx, ty):
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)

    # OpenCV 형식:
    # x' = c*x + s*y + tx
    # y' = -s*x + c*y + ty
    # tx = (1-c)cx - s*cy
    # ty = s*cx + (1-c)cy
    A = np.array([[1-c, -s], [s, 1-c]], np.float64)
    if abs(np.linalg.det(A)) < 1e-12:
        raise ValueError("angle too small to determine center")
    center = np.linalg.solve(A, np.array([tx, ty], np.float64))
    return float(center[0]), float(center[1])


def matrix_from_parameters(angle, tx, ty):
    t = math.radians(angle)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, s, tx], [-s, c, ty]], np.float64)


def save_diagnostics(ref, target, M, gain, offset, cx, cy, outdir):
    h, w = ref.shape

    # target -> reference 정렬은 M의 역변환
    Minv = cv2.invertAffineTransform(M)
    aligned = warp_image(target, Minv, (w, h))
    aligned = np.clip(gain * aligned + offset, 0, 1)

    diff = np.abs(ref - aligned)
    ref8 = np.clip(ref * 255, 0, 255).astype(np.uint8)
    tar8 = np.clip(target * 255, 0, 255).astype(np.uint8)
    ali8 = np.clip(aligned * 255, 0, 255).astype(np.uint8)
    dif8 = np.clip(diff * 5 * 255, 0, 255).astype(np.uint8)

    overlay = np.zeros((h, w, 3), np.uint8)
    overlay[:, :, 2] = ref8
    overlay[:, :, 1] = ali8

    # 중심이 화면 안/근처일 때만 표시
    if -10000 < cx < 10000 and -10000 < cy < 10000:
        cv2.drawMarker(
            overlay, (int(round(cx)), int(round(cy))),
            (255, 255, 255), cv2.MARKER_CROSS, 25, 2
        )

    cv2.imwrite(str(outdir / "reference.png"), ref8)
    cv2.imwrite(str(outdir / "rotated.png"), tar8)
    cv2.imwrite(str(outdir / "aligned.png"), ali8)
    cv2.imwrite(str(outdir / "difference_x5.png"), dif8)
    cv2.imwrite(str(outdir / "overlay_reference_red_aligned_green.png"), overlay)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ref = preprocess(read_gray(args.reference), args.blur_sigma)
    target = preprocess(read_gray(args.rotated), args.blur_sigma)
    if ref.shape != target.shape:
        raise ValueError("image size mismatch")

    mask = read_mask(args.mask, ref.shape)

    score, phase_response, search_angle, M0 = angle_search(
        ref, target,
        args.angle_min, args.angle_max,
        args.coarse_step, args.fine_range, args.fine_step
    )
    a0, tx0, ty0 = matrix_angle_translation(M0)
    cx0, cy0 = center_from_angle_translation(a0, tx0, ty0)

    print("[Global initialization]")
    print(f"  angle       : {a0:.9f} deg")
    print(f"  translation : ({tx0:.9f}, {ty0:.9f}) px")
    print(f"  center      : ({cx0:.9f}, {cy0:.9f}) px")
    print(f"  overlap NCC : {score:.12f}")
    print(f"  phase resp. : {phase_response:.12f}")

    p, rmse, valid_fraction = refine(
        ref, target, mask, M0, args.max_points, args.seed
    )
    angle, tx, ty, gain, offset = map(float, p)
    cx, cy = center_from_angle_translation(angle, tx, ty)
    M = matrix_from_parameters(angle, tx, ty)

    print("\n[Final result]")
    print(f"  angle       : {angle:.9f} deg")
    print(f"  translation : ({tx:.9f}, {ty:.9f}) px")
    print(f"  center_x    : {cx:.9f} px")
    print(f"  center_y    : {cy:.9f} px")
    print(f"  gain        : {gain:.9f}")
    print(f"  offset      : {offset:.9f}")
    print(f"  RMSE        : {rmse:.12f}")
    print(f"  valid ratio : {valid_fraction:.6f}")

    result = {
        "reference": str(args.reference),
        "rotated": str(args.rotated),
        "angle_deg": angle,
        "translation_x_px": tx,
        "translation_y_px": ty,
        "center_x_px": cx,
        "center_y_px": cy,
        "gain": gain,
        "offset": offset,
        "rmse": rmse,
        "valid_fraction": valid_fraction,
        "initial_overlap_ncc": score,
        "initial_phase_response": phase_response,
        "matrix_2x3": M.tolist(),
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    with (args.output_dir / "result.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        wr = csv.writer(f)
        wr.writerow(result.keys())
        wr.writerow(result.values())

    save_diagnostics(
        ref, target, M, gain, offset, cx, cy, args.output_dir
    )
    print(f"\n[OK] 결과 폴더: {args.output_dir}")


if __name__ == "__main__":
    main()