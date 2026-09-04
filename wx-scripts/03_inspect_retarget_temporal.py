#!/usr/bin/env python3

import argparse
import numpy as np


parser = argparse.ArgumentParser(
    description=(
        "Inspect frame-to-frame temporal changes in left/right "
        "retarget arm direction signals."
    )
)

parser.add_argument(
    "--path",
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

PATH = args.path
SIDE = args.side


# ============================================================
# Load
# ============================================================

d = np.load(PATH)

print("Loaded:", PATH)
print("Side  :", SIDE)


# ============================================================
# Utilities
# ============================================================

def angle_between_frames(x):
    """
    Frame-to-frame angle change for a sequence of normalized vectors.

    x: [N, 3]
    return: [N-1], degrees
    """
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    dots = np.sum(
        x[1:] * x[:-1],
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


def report_signal(key):
    if key not in d:
        raise KeyError(
            "Missing key in retarget_input.npz: {}".format(key)
        )

    a = angle_between_frames(
        d[key]
    )

    max_idx = int(
        np.argmax(a)
    )

    print()
    print(
        "========== {} ==========".format(key)
    )

    print(
        "  mean step:",
        a.mean(),
        "deg",
    )

    print(
        "  max step :",
        a.max(),
        "deg",
    )

    print(
        "  max jump :",
        "{} -> {}".format(
            max_idx,
            max_idx + 1,
        ),
    )

    print(
        "  >10 deg  :",
        int(
            np.sum(
                a > 10
            )
        ),
    )

    print(
        "  >20 deg  :",
        int(
            np.sum(
                a > 20
            )
        ),
    )

    print(
        "  >30 deg  :",
        int(
            np.sum(
                a > 30
            )
        ),
    )

    print(
        "  >60 deg  :",
        int(
            np.sum(
                a > 60
            )
        ),
    )


def inspect_side(side):
    print()
    print(
        "=========================================="
    )
    print(
        "{} arm temporal inspection".format(
            side.upper()
        )
    )
    print(
        "=========================================="
    )

    report_signal(
        "{}_upper_arm_dir".format(side)
    )

    report_signal(
        "{}_forearm_dir".format(side)
    )


# ============================================================
# Run
# ============================================================

if SIDE == "both":
    inspect_side("left")
    inspect_side("right")
else:
    inspect_side(SIDE)
