import os

import cv2
import numpy as np
import torch


# =========================
# Config
# =========================
PT_PATH = "res/pick_place/hybrikx_output.pt"
OUTPUT_PATH = "res/pick_place/arm_overlay.mp4"

FPS = 30

# 关节点索引
LEFT_SHOULDER = 16
RIGHT_SHOULDER = 17

LEFT_ELBOW = 18
RIGHT_ELBOW = 19

LEFT_WRIST = 20
RIGHT_WRIST = 21


# =========================
# Load prediction
# =========================
data = torch.load(
    PT_PATH,
    map_location="cpu",
    # weights_only=False,
)

pred_uvd = np.asarray(data["pred_uvd_jts"])   # [N, 71, 3]
img_paths = data["img_path"]
bboxes = np.asarray(data["bbox"])

num_frames = len(pred_uvd)

print("Loaded:", PT_PATH)
print("pred_uvd_jts shape:", pred_uvd.shape)
print("num_frames:", num_frames)

if pred_uvd.shape[1:] != (71, 3):
    raise ValueError(
        f"Expected pred_uvd_jts shape [N, 71, 3], got {pred_uvd.shape}"
    )


# =========================
# Read first frame
# =========================
first_img = cv2.imread(img_paths[0])

if first_img is None:
    raise RuntimeError(
        f"Cannot read image: {img_paths[0]}"
    )

height, width = first_img.shape[:2]

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True,
)


# =========================
# Video writer
# =========================
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    FPS,
    (width, height),
)

if not writer.isOpened():
    raise RuntimeError(
        f"Cannot open video writer: {OUTPUT_PATH}"
    )


# =========================
# Drawing utilities
# =========================
def draw_joint(img, point, label):
    """
    point: (u, v), image pixel coordinates
    """

    x = int(round(point[0]))
    y = int(round(point[1]))

    cv2.circle(
        img,
        (x, y),
        7,
        (0, 0, 255),
        -1,
    )

    cv2.putText(
        img,
        label,
        (x + 8, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )


def draw_bone(img, p1, p2):
    x1 = int(round(p1[0]))
    y1 = int(round(p1[1]))

    x2 = int(round(p2[0]))
    y2 = int(round(p2[1]))

    cv2.line(
        img,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3,
    )


# =========================
# Process frames
# =========================
for frame_idx in range(num_frames):

    img = cv2.imread(img_paths[frame_idx])

    if img is None:
        print(
            f"Warning: cannot read frame "
            f"{frame_idx}: {img_paths[frame_idx]}"
        )
        continue

    bbox = bboxes[frame_idx]

    # bbox = [x1, y1, x2, y2]
    x1, y1, x2, y2 = bbox

    bbox_w = x2 - x1
    bbox_h = y2 - y1

    bbox_cx = (x1 + x2) / 2.0
    bbox_cy = (y1 + y2) / 2.0

    joints = pred_uvd[frame_idx]

    # -------------------------------------------------
    # HybrIK-X pred_uvd 的 u/v 是 bbox-relative normalized coordinate
    #
    # demo_video_x.py 中原始可视化采用：
    #
    # pts = uv_jts * bbox_xywh[2]
    # pts[:, 0] += bbox_xywh[0]
    # pts[:, 1] += bbox_xywh[1]
    #
    # 所以这里保持完全一致
    # -------------------------------------------------

    uv = joints[:, :2].copy()

    uv[:, 0] = uv[:, 0] * bbox_w + bbox_cx
    uv[:, 1] = uv[:, 1] * bbox_w + bbox_cy

    # 6 个关键点
    ls = uv[LEFT_SHOULDER]
    rs = uv[RIGHT_SHOULDER]

    le = uv[LEFT_ELBOW]
    re = uv[RIGHT_ELBOW]

    lw = uv[LEFT_WRIST]
    rw = uv[RIGHT_WRIST]

    # -------------------------
    # draw bones
    # -------------------------
    draw_bone(img, ls, le)
    draw_bone(img, le, lw)

    draw_bone(img, rs, re)
    draw_bone(img, re, rw)

    # shoulder line，方便观察左右
    draw_bone(img, ls, rs)

    # -------------------------
    # draw joints
    # -------------------------
    draw_joint(img, ls, "LS")
    draw_joint(img, le, "LE")
    draw_joint(img, lw, "LW")

    draw_joint(img, rs, "RS")
    draw_joint(img, re, "RE")
    draw_joint(img, rw, "RW")

    # -------------------------
    # frame info
    # -------------------------
    cv2.putText(
        img,
        f"Frame {frame_idx}/{num_frames - 1}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    writer.write(img)

    if frame_idx % 100 == 0:
        print(
            f"Processed "
            f"{frame_idx}/{num_frames}"
        )


# =========================
# Finish
# =========================
writer.release()

print()
print("Done.")
print("Saved to:", OUTPUT_PATH)