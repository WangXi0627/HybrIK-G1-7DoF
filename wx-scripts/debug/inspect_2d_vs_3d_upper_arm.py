import torch
import numpy as np


PT_PATH = "res/pick_place/hybrikx_output.pt"
RETARGET_PATH = "res/pick_place/retarget_input.npz"


LS = 16
LE = 18


def normalize(v):
    return v / (
        np.linalg.norm(
            v,
            axis=-1,
            keepdims=True,
        ) + 1e-8
    )


def angle_from_frame0(v):
    v = normalize(v)

    ref = v[0]

    dots = np.sum(
        v * ref[None, :],
        axis=1,
    )

    dots = np.clip(
        dots,
        -1.0,
        1.0,
    )

    return np.degrees(
        np.arccos(dots)
    )


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
).reshape(-1, 71, 3)

bboxes = np.asarray(
    data["bbox"]
)

retarget = np.load(
    RETARGET_PATH
)


# ============================================================
# Recover 2D shoulder / elbow
# ============================================================

shoulder_uv = []
elbow_uv = []

for i in range(len(pred_uvd)):

    x1, y1, x2, y2 = bboxes[i]

    w = x2 - x1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    def recover(jid):
        u = pred_uvd[i, jid, 0] * w + cx
        v = pred_uvd[i, jid, 1] * w + cy
        return np.array([u, v])

    shoulder_uv.append(
        recover(LS)
    )

    elbow_uv.append(
        recover(LE)
    )


shoulder_uv = np.asarray(
    shoulder_uv
)

elbow_uv = np.asarray(
    elbow_uv
)


# ============================================================
# 2D upper-arm direction
# ============================================================

upper_2d = (
    elbow_uv
    -
    shoulder_uv
)

upper_2d_change = (
    angle_from_frame0(
        upper_2d
    )
)


# ============================================================
# 3D upper-arm direction
# ============================================================

upper_3d = retarget[
    "left_upper_arm_dir"
]

upper_3d_change = (
    angle_from_frame0(
        upper_3d
    )
)


# ============================================================
# Results
# ============================================================

print(
    "========== 2D upper-arm direction =========="
)

print(
    "max change:",
    upper_2d_change.max(),
    "deg"
)

print()
print(
    "========== 3D torso-local upper-arm direction =========="
)

print(
    "max change:",
    upper_3d_change.max(),
    "deg"
)


print()
print(
    "========== Frames of maximum change =========="
)

print(
    "2D max frame:",
    np.argmax(
        upper_2d_change
    )
)

print(
    "3D max frame:",
    np.argmax(
        upper_3d_change
    )
)