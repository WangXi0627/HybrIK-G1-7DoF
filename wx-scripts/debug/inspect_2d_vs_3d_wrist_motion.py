import torch
import numpy as np


PT_PATH = "res/pick_place/hybrikx_output.pt"
RETARGET_PATH = "res/pick_place/retarget_input.npz"


LW = 20
LS = 16
SPINE3 = 9
NECK = 12


# ============================================================
# Load
# ============================================================

data = torch.load(
    PT_PATH,
    map_location="cpu",
    # weights_only=False,
)

pred_uvd = np.asarray(
    data["pred_uvd_jts"]
)

xyz = np.asarray(
    data["pred_xyz_hybrik"]
)

bboxes = np.asarray(
    data["bbox"]
)

retarget = np.load(
    RETARGET_PATH
)

R_torso = retarget[
    "torso_rotation"
]


# ============================================================
# Reshape UV
# ============================================================

pred_uvd = pred_uvd.reshape(
    len(pred_uvd),
    71,
    3,
)


# ============================================================
# Recover left wrist image coordinates
#
# Same convention as demo_video_x.py
# ============================================================

wrist_uv = []

for i in range(len(pred_uvd)):

    x1, y1, x2, y2 = bboxes[i]

    bbox_w = x2 - x1

    bbox_cx = (
        x1 + x2
    ) / 2.0

    bbox_cy = (
        y1 + y2
    ) / 2.0

    u = (
        pred_uvd[i, LW, 0]
        * bbox_w
        + bbox_cx
    )

    v = (
        pred_uvd[i, LW, 1]
        * bbox_w
        + bbox_cy
    )

    wrist_uv.append([
        u,
        v,
    ])


wrist_uv = np.asarray(
    wrist_uv
)


# ============================================================
# 2D displacement relative to frame 0
# ============================================================

uv_disp = (
    wrist_uv
    -
    wrist_uv[0]
)

uv_disp_norm = np.linalg.norm(
    uv_disp,
    axis=1,
)


print(
    "========== 2D left wrist =========="
)

print(
    "frame0:",
    wrist_uv[0]
)

print(
    "u range:",
    wrist_uv[:, 0].min(),
    "~",
    wrist_uv[:, 0].max(),
)

print(
    "v range:",
    wrist_uv[:, 1].min(),
    "~",
    wrist_uv[:, 1].max(),
)

print(
    "max displacement from frame0:",
    uv_disp_norm.max(),
    "pixels",
)


# ============================================================
# 3D wrist torso-local
# ============================================================

torso_center = (
    xyz[:, SPINE3]
    +
    xyz[:, NECK]
) / 2.0


wrist_global = (
    xyz[:, LW]
    -
    torso_center
)


wrist_local = np.einsum(
    "nij,nj->ni",
    np.transpose(
        R_torso,
        (0, 2, 1),
    ),
    wrist_global,
)


xyz_disp = (
    wrist_local
    -
    wrist_local[0]
)

xyz_disp_norm = np.linalg.norm(
    xyz_disp,
    axis=1,
)


print()
print(
    "========== 3D torso-local left wrist =========="
)

print(
    "frame0:",
    wrist_local[0]
)

print(
    "x range:",
    wrist_local[:, 0].min(),
    "~",
    wrist_local[:, 0].max(),
)

print(
    "y range:",
    wrist_local[:, 1].min(),
    "~",
    wrist_local[:, 1].max(),
)

print(
    "z range:",
    wrist_local[:, 2].min(),
    "~",
    wrist_local[:, 2].max(),
)

print(
    "max displacement from frame0:",
    xyz_disp_norm.max(),
)


# ============================================================
# 2D shoulder-relative wrist
#
# This is useful because camera/body motion is partially removed.
# ============================================================

shoulder_uv = []

for i in range(len(pred_uvd)):

    x1, y1, x2, y2 = bboxes[i]

    bbox_w = x2 - x1

    bbox_cx = (
        x1 + x2
    ) / 2.0

    bbox_cy = (
        y1 + y2
    ) / 2.0

    u = (
        pred_uvd[i, LS, 0]
        * bbox_w
        + bbox_cx
    )

    v = (
        pred_uvd[i, LS, 1]
        * bbox_w
        + bbox_cy
    )

    shoulder_uv.append([
        u,
        v,
    ])


shoulder_uv = np.asarray(
    shoulder_uv
)

wrist_rel_shoulder_2d = (
    wrist_uv
    -
    shoulder_uv
)

rel_disp = (
    wrist_rel_shoulder_2d
    -
    wrist_rel_shoulder_2d[0]
)

rel_disp_norm = np.linalg.norm(
    rel_disp,
    axis=1,
)


print()
print(
    "========== 2D wrist relative to shoulder =========="
)

print(
    "max displacement:",
    rel_disp_norm.max(),
    "pixels",
)