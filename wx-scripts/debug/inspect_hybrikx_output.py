import torch
import numpy as np

PT_PATH = "res/pick_place/hybrikx_output.pt"

data = torch.load(
    PT_PATH,
    map_location="cpu",
    # weights_only=False,
)

print("========== keys ==========")
print(data.keys())

xyz = np.asarray(data["pred_xyz_hybrik"])

print("\n========== xyz ==========")
print("shape:", xyz.shape)

# --------------------------------------------------
# HybrIK-X / SMPL-X joints used for upper-body retargeting
# --------------------------------------------------

JOINTS = {
    # torso
    "pelvis": 0,
    "spine1": 3,
    "spine2": 6,
    "spine3": 9,
    "neck": 12,

    # shoulder / arm
    "left_collar": 13,
    "right_collar": 14,

    "left_shoulder": 16,
    "right_shoulder": 17,

    "left_elbow": 18,
    "right_elbow": 19,

    "left_wrist": 20,
    "right_wrist": 21,

    # left hand
    "left_index1": 25,
    "left_index2": 26,
    "left_index3": 27,

    "left_middle1": 28,
    "left_middle2": 29,
    "left_middle3": 30,

    "left_pinky1": 31,
    "left_pinky2": 32,
    "left_pinky3": 33,

    "left_ring1": 34,
    "left_ring2": 35,
    "left_ring3": 36,

    "left_thumb1": 37,
    "left_thumb2": 38,
    "left_thumb3": 39,

    # right hand
    "right_index1": 40,
    "right_index2": 41,
    "right_index3": 42,

    "right_middle1": 43,
    "right_middle2": 44,
    "right_middle3": 45,

    "right_pinky1": 46,
    "right_pinky2": 47,
    "right_pinky3": 48,

    "right_ring1": 49,
    "right_ring2": 50,
    "right_ring3": 51,

    "right_thumb1": 52,
    "right_thumb2": 53,
    "right_thumb3": 54,
}

# 检查第一帧
print("\n========== frame 0 ==========")

for name, idx in JOINTS.items():
    print(
        f"{idx:2d} {name:20s}: "
        f"{xyz[0, idx]}"
    )