import torch
import numpy as np


PT_PATH = "res/pick_place/hybrikx_output.pt"
OUT_PATH = "res/pick_place/retarget_input.npz"


def normalize(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)


data = torch.load(
    PT_PATH,
    map_location="cpu",
    # weights_only=False,
)

xyz = np.asarray(data["pred_xyz_hybrik"])   # [N, 71, 3]

# joints
SPINE3 = 9
NECK = 12

LS = 16
RS = 17
LE = 18
RE = 19
LW = 20
RW = 21


# =========================
# torso frame
# =========================

# 人体右方向
right_axis = normalize(
    xyz[:, RS] - xyz[:, LS]
)

# 人体向上方向
up_raw = normalize(
    xyz[:, NECK] - xyz[:, SPINE3]
)

# 前向方向
forward_axis = normalize(
    np.cross(right_axis, up_raw)
)

# 重新正交化 up
up_axis = normalize(
    np.cross(forward_axis, right_axis)
)

# rotation matrix:
# columns = [right, up, forward]
R_torso = np.stack(
    [right_axis, up_axis, forward_axis],
    axis=-1,
)   # [N, 3, 3]


# =========================
# arm vectors in camera frame
# =========================

left_upper = normalize(
    xyz[:, LE] - xyz[:, LS]
)

left_fore = normalize(
    xyz[:, LW] - xyz[:, LE]
)

right_upper = normalize(
    xyz[:, RE] - xyz[:, RS]
)

right_fore = normalize(
    xyz[:, RW] - xyz[:, RE]
)


# =========================
# convert to torso frame
# =========================

# local = R^T * world
left_upper_local = np.einsum(
    "nij,nj->ni",
    np.transpose(R_torso, (0, 2, 1)),
    left_upper,
)

left_fore_local = np.einsum(
    "nij,nj->ni",
    np.transpose(R_torso, (0, 2, 1)),
    left_fore,
)

right_upper_local = np.einsum(
    "nij,nj->ni",
    np.transpose(R_torso, (0, 2, 1)),
    right_upper,
)

right_fore_local = np.einsum(
    "nij,nj->ni",
    np.transpose(R_torso, (0, 2, 1)),
    right_fore,
)


# =========================
# elbow angle
# =========================

def joint_angle(a, b, c):
    """
    Angle ABC.
    """
    v1 = normalize(a - b)
    v2 = normalize(c - b)

    cos_angle = np.sum(v1 * v2, axis=-1)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return np.arccos(cos_angle)


left_elbow_angle = joint_angle(
    xyz[:, LS],
    xyz[:, LE],
    xyz[:, LW],
)

right_elbow_angle = joint_angle(
    xyz[:, RS],
    xyz[:, RE],
    xyz[:, RW],
)


# =========================
# save
# =========================

np.savez(
    OUT_PATH,

    torso_rotation=R_torso,

    left_upper_arm_dir=left_upper_local,
    left_forearm_dir=left_fore_local,

    right_upper_arm_dir=right_upper_local,
    right_forearm_dir=right_fore_local,

    left_elbow_angle=left_elbow_angle,
    right_elbow_angle=right_elbow_angle,
)

print("Saved:", OUT_PATH)
print("Frames:", len(xyz))