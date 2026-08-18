#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
rotate_image_about_center.py

입력 이미지 한 장을 사용자가 지정한 이미지 좌표상의 회전중심을 기준으로 회전한다.

좌표 규칙:
- x: 왼쪽에서 오른쪽
- y: 위에서 아래
- OpenCV 좌표계 사용
- 양의 angle_deg는 OpenCV 기준 반시계방향(CCW)

예:
python rotate_image_about_center.py input.png rotated.png --angle 5.0 --cx 1234.5 --cy 980.2

경계 처리 예:
python rotate_image_about_center.py input.png rotated.png --angle -5 --cx 1000 --cy 800 \
    --border-mode reflect --interpolation cubic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


INTERPOLATIONS = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}

BORDER_MODES = {
    "constant": cv2.BORDER_CONSTANT,
    "reflect": cv2.BORDER_REFLECT_101,
    "replicate": cv2.BORDER_REPLICATE,
    "wrap": cv2.BORDER_WRAP,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="지정한 이미지 좌표의 회전중심을 기준으로 영상을 회전합니다."
    )
    parser.add_argument("input", type=Path, help="입력 이미지 경로")
    parser.add_argument("output", type=Path, help="출력 이미지 경로")
    parser.add_argument("--angle", type=float, required=True, help="회전각 [deg], 양수=반시계방향")
    parser.add_argument("--cx", type=float, required=True, help="회전중심 x [pixel]")
    parser.add_argument("--cy", type=float, required=True, help="회전중심 y [pixel]")
    parser.add_argument(
        "--interpolation",
        choices=INTERPOLATIONS,
        default="cubic",
        help="보간법 (기본: cubic)",
    )
    parser.add_argument(
        "--border-mode",
        choices=BORDER_MODES,
        default="reflect",
        help="영상 밖 경계 처리 (기본: reflect)",
    )
    parser.add_argument(
        "--border-value",
        type=float,
        default=0.0,
        help="border-mode=constant일 때 채울 값",
    )
    parser.add_argument(
        "--save-matrix",
        type=Path,
        default=None,
        help="사용한 2x3 변환행렬과 입력값을 JSON으로 저장",
    )
    return parser.parse_args()


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {path}")
    return image


def main() -> None:
    args = parse_args()
    image = read_image(args.input)
    h, w = image.shape[:2]

    matrix = cv2.getRotationMatrix2D(
        center=(args.cx, args.cy),
        angle=args.angle,
        scale=1.0,
    )

    rotated = cv2.warpAffine(
        image,
        matrix,
        dsize=(w, h),
        flags=INTERPOLATIONS[args.interpolation],
        borderMode=BORDER_MODES[args.border_mode],
        borderValue=args.border_value,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(args.output), rotated)
    if not ok:
        raise OSError(f"출력 이미지를 저장하지 못했습니다: {args.output}")

    print(f"[OK] input      : {args.input}")
    print(f"[OK] output     : {args.output}")
    print(f"[OK] image size : {w} x {h}")
    print(f"[OK] angle      : {args.angle:.9f} deg")
    print(f"[OK] center     : ({args.cx:.9f}, {args.cy:.9f}) px")
    print("[OK] affine matrix:")
    print(matrix)

    if args.save_matrix is not None:
        payload = {
            "input": str(args.input),
            "output": str(args.output),
            "width": int(w),
            "height": int(h),
            "angle_deg": float(args.angle),
            "center_x_px": float(args.cx),
            "center_y_px": float(args.cy),
            "matrix_2x3": matrix.tolist(),
            "interpolation": args.interpolation,
            "border_mode": args.border_mode,
        }
        args.save_matrix.parent.mkdir(parents=True, exist_ok=True)
        args.save_matrix.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[OK] metadata    : {args.save_matrix}")


if __name__ == "__main__":
    main()
