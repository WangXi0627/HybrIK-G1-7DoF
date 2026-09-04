import torch
import numpy as np


PT_PATH = "res/pick_place/hybrikx_output.pt"
RETARGET_PATH = "res/pick_place/retarget_input.npz"


# SMPL-X
SPINE3 = 9
NECK = 12

LS = 16
LE = 18
LW = 20


def normalize(v):
    return v / (
        np.linalg.norm(
            v,
            axis=-1,
            keepdims=True,
        )
        + 1e-8
    )


def angular_range(v):
    """
    Angle of every vector relative to frame 0.
    """
    ref = v[0]

    dots = np.sum(
        v * ref[None, :],
        axis=1,
    )

    dots = np.clip(
        dots,
        -1,
        1,
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

xyz = np.asarray(
    data["pred_xyz_hybrik"]
)

retarget = np.load(
    RETARGET_PATH
)

upper = retarget[
    "left_upper_arm_dir"
]

fore = retarget[
    "left_forearm_dir"
]

R_torso = retarget[
    "torso_rotation"
]


# ============================================================
# Direction change relative to frame 0
# ============================================================

upper_change = angular_range(
    upper
)

fore_change = angular_range(
    fore
)


print(
    "========== Direction range =========="
)

print(
    "upper relative frame0:"
)

print(
    "  min :",
    upper_change.min(),
    "deg"
)

print(
    "  max :",
    upper_change.max(),
    "deg"
)


print()

print(
    "fore relative frame0:"
)

print(
    "  min :",
    fore_change.min(),
    "deg"
)

print(
    "  max :",
    fore_change.max(),
    "deg"
)


# ============================================================
# Wrist position relative to torso
# ============================================================

# Use spine3/neck midpoint as torso reference
torso_center = (
    xyz[:, SPINE3]
    +
    xyz[:, NECK]
) / 2.0


# wrist displacement in original HybrIK coordinates
wrist_global = (
    xyz[:, LW]
    -
    torso_center
)


# Convert to torso-local frame
wrist_local = np.einsum(
    "nij,nj->ni",
    np.transpose(
        R_torso,
        (0, 2, 1),
    ),
    wrist_global,
)


# Position relative to frame 0
disp = (
    wrist_local
    -
    wrist_local[0]
)

disp_norm = np.linalg.norm(
    disp,
    axis=1,
)


print()
print(
    "========== Wrist motion in torso frame =========="
)

print(
    "frame 0 wrist:",
    wrist_local[0]
)

print(
    "min x / max x:",
    wrist_local[:, 0].min(),
    wrist_local[:, 0].max(),
)

print(
    "min y / max y:",
    wrist_local[:, 1].min(),
    wrist_local[:, 1].max(),
)

print(
    "min z / max z:",
    wrist_local[:, 2].min(),
    wrist_local[:, 2].max(),
)

print()

print(
    "max displacement from frame0:",
    disp_norm.max()
)


# ============================================================
# Shoulder position relative to torso
# ============================================================

shoulder_global = (
    xyz[:, LS]
    -
    torso_center
)

shoulder_local = np.einsum(
    "nij,nj->ni",
    np.transpose(
        R_torso,
        (0, 2, 1),
    ),
    shoulder_global,
)

shoulder_disp = (
    shoulder_local
    -
    shoulder_local[0]
)

shoulder_disp_norm = (
    np.linalg.norm(
        shoulder_disp,
        axis=1,
    )
)


print()
print(
    "========== Shoulder motion in torso frame =========="
)

print(
    "max shoulder displacement:",
    shoulder_disp_norm.max()
)