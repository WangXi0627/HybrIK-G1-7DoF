#!/usr/bin/env python3

import argparse

import numpy as np
import torch


parser = argparse.ArgumentParser(
    description=(
        "Inspect human left/right arm motion range "
        "in torso-local coordinates."
    )
)

parser.add_argument(
    "--pt-path",
    default="res/pick_banana/hybrikx_output.pt",
    help="Path to hybrikx_output.pt",
)

parser.add_argument(
    "--retarget-path",
    default="res/pick_banana/retarget_input.npz",
    help="Path to retarget_input.npz",
)

parser.add_argument(
    "--side",
    choices=["left", "right", "both"],
    default="left",
    help="Which human arm to inspect.",
)

args = parser.parse_args()

PT_PATH = args.pt_path
RETARGET_PATH = args.retarget_path
SIDE = args.side


# ============================================================
# SMPL-X / HybrIK joint IDs
# ============================================================

SPINE3 = 9
NECK = 12

JOINTS = {
    "left": {
        "shoulder": 16,
        "elbow": 18,
        "wrist": 20,
    },
    "right": {
        "shoulder": 17,
        "elbow": 19,
        "wrist": 21,
    },
}


# ============================================================
# Utilities
# ============================================================

def angular_range(v):
    """
    Angle of every normalized vector relative to frame 0.
    """
    v = np.asarray(
        v,
        dtype=np.float64,
    )

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


def to_torso_local(
    points_world,
    torso_center,
    R_torso,
):
    relative = (
        points_world
        -
        torso_center
    )

    return np.einsum(
        "nij,nj->ni",
        np.transpose(
            R_torso,
            (0, 2, 1),
        ),
        relative,
    )


# ============================================================
# Load
# ============================================================

data = torch.load(
    PT_PATH,
    map_location="cpu",
)

xyz = np.asarray(
    data["pred_xyz_hybrik"],
    dtype=np.float64,
)

retarget = np.load(
    RETARGET_PATH
)

R_torso = np.asarray(
    retarget["torso_rotation"],
    dtype=np.float64,
)


if len(xyz) != len(R_torso):
    raise ValueError(
        "Frame mismatch between HybrIK output and retarget_input."
    )


torso_center = (
    xyz[:, SPINE3]
    +
    xyz[:, NECK]
) / 2.0


print(
    "HybrIK       :",
    PT_PATH,
)
print(
    "Retarget     :",
    RETARGET_PATH,
)
print(
    "Frames       :",
    len(xyz),
)
print(
    "Side         :",
    SIDE,
)


# ============================================================
# Per-side inspection
# ============================================================

def inspect_side(side):
    ids = JOINTS[side]

    shoulder_id = ids["shoulder"]
    elbow_id = ids["elbow"]
    wrist_id = ids["wrist"]

    upper_key = (
        "{}_upper_arm_dir".format(side)
    )
    fore_key = (
        "{}_forearm_dir".format(side)
    )

    if upper_key not in retarget:
        raise KeyError(
            "Missing key: {}".format(
                upper_key
            )
        )

    if fore_key not in retarget:
        raise KeyError(
            "Missing key: {}".format(
                fore_key
            )
        )

    upper = np.asarray(
        retarget[upper_key],
        dtype=np.float64,
    )

    fore = np.asarray(
        retarget[fore_key],
        dtype=np.float64,
    )


    # --------------------------------------------------------
    # Direction change relative to frame 0
    # --------------------------------------------------------

    upper_change = angular_range(
        upper
    )

    fore_change = angular_range(
        fore
    )


    print()
    print(
        "=========================================="
    )
    print(
        "{} ARM".format(
            side.upper()
        )
    )
    print(
        "=========================================="
    )

    print()
    print(
        "========== Direction range =========="
    )

    print(
        "upper relative frame0:"
    )
    print(
        "  min :",
        upper_change.min(),
        "deg",
    )
    print(
        "  max :",
        upper_change.max(),
        "deg",
    )

    print()
    print(
        "fore relative frame0:"
    )
    print(
        "  min :",
        fore_change.min(),
        "deg",
    )
    print(
        "  max :",
        fore_change.max(),
        "deg",
    )


    # --------------------------------------------------------
    # Wrist motion relative to torso
    # --------------------------------------------------------

    wrist_local = to_torso_local(
        xyz[:, wrist_id],
        torso_center,
        R_torso,
    )

    wrist_disp = (
        wrist_local
        -
        wrist_local[0]
    )

    wrist_disp_norm = np.linalg.norm(
        wrist_disp,
        axis=1,
    )


    print()
    print(
        "========== Wrist motion in torso frame =========="
    )

    print(
        "frame 0 wrist:",
        wrist_local[0],
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

    print(
        "max displacement from frame0:",
        wrist_disp_norm.max(),
    )


    # --------------------------------------------------------
    # Shoulder motion relative to torso
    # --------------------------------------------------------

    shoulder_local = to_torso_local(
        xyz[:, shoulder_id],
        torso_center,
        R_torso,
    )

    shoulder_disp = (
        shoulder_local
        -
        shoulder_local[0]
    )

    shoulder_disp_norm = np.linalg.norm(
        shoulder_disp,
        axis=1,
    )


    print()
    print(
        "========== Shoulder motion in torso frame =========="
    )

    print(
        "frame 0 shoulder:",
        shoulder_local[0],
    )

    print(
        "max shoulder displacement:",
        shoulder_disp_norm.max(),
    )


    # --------------------------------------------------------
    # Optional elbow motion diagnostics
    # --------------------------------------------------------

    elbow_local = to_torso_local(
        xyz[:, elbow_id],
        torso_center,
        R_torso,
    )

    elbow_disp = (
        elbow_local
        -
        elbow_local[0]
    )

    elbow_disp_norm = np.linalg.norm(
        elbow_disp,
        axis=1,
    )


    print()
    print(
        "========== Elbow motion in torso frame =========="
    )

    print(
        "frame 0 elbow:",
        elbow_local[0],
    )

    print(
        "max elbow displacement:",
        elbow_disp_norm.max(),
    )


# ============================================================
# Run
# ============================================================

if SIDE == "both":
    inspect_side("left")
    inspect_side("right")
else:
    inspect_side(SIDE)
