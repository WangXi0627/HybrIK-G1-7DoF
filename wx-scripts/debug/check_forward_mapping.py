import torch
import numpy as np


# ============================================================
# Config
# ============================================================

PT_PATH = "res/pick_place/hybrikx_output.pt"
RETARGET_PATH = "res/pick_place/retarget_input.npz"

# 你说“一开始就在向前伸”
# 先检查 0 -> 60 帧
START_FRAME = 0
END_FRAME = 60


# HybrIK joints
SPINE3 = 9
NECK = 12
LW = 20


# ============================================================
# Load
# ============================================================

data = torch.load(
    PT_PATH,
    map_location="cpu",
    # weights_only=False,
)

xyz = np.asarray(
    data["pred_xyz_hybrik"],
    dtype=np.float64,
)

retarget = np.load(
    RETARGET_PATH,
)

R_torso = np.asarray(
    retarget["torso_rotation"],
    dtype=np.float64,
)


# ============================================================
# Human wrist position relative to torso
# ============================================================

torso_center = (
    xyz[:, SPINE3]
    + xyz[:, NECK]
) / 2.0

wrist_world_relative = (
    xyz[:, LW]
    - torso_center
)

wrist_local = np.einsum(
    "nij,nj->ni",
    np.transpose(
        R_torso,
        (0, 2, 1),
    ),
    wrist_world_relative,
)


# ============================================================
# Compare start/end
# ============================================================

p0 = wrist_local[START_FRAME]
p1 = wrist_local[END_FRAME]

delta = p1 - p0


print("==========================================")
print("Human torso-local wrist movement")
print("==========================================")

print("start frame:", START_FRAME)
print("end frame  :", END_FRAME)

print()
print("start wrist:", p0)
print("end wrist  :", p1)

print()
print("delta:")
print(" dx =", delta[0])
print(" dy =", delta[1])
print(" dz =", delta[2])


# ============================================================
# Two candidate mappings
#
# G1 +x = forward
#
# Mapping A (original):
#   G1_x = +Human_z
#
# Mapping B (flip_forward):
#   G1_x = -Human_z
# ============================================================

g1_forward_delta_original = (
    delta[2]
)

g1_forward_delta_flip = (
    -delta[2]
)


print()
print("==========================================")
print("Predicted G1 forward displacement")
print("==========================================")

print(
    "ORIGINAL mapping "
    "(G1_x = +Human_z):",
    g1_forward_delta_original
)

print(
    "FLIP_FORWARD mapping "
    "(G1_x = -Human_z):",
    g1_forward_delta_flip
)


print()
print("==========================================")
print("Interpretation")
print("==========================================")

print(
    "For a known HUMAN FORWARD reach, "
    "the correct mapping should give "
    "POSITIVE G1 forward displacement."
)