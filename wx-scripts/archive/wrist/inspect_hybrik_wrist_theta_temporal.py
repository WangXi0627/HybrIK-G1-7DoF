import torch
import numpy as np
from scipy.spatial.transform import Rotation


# ============================================================
# Config
# ============================================================

# 改成你当前视频对应的 HybrIK-X 输出
PT_PATH = "res/pick_place/hybrikx_output.pt"


# ============================================================
# SMPL-X joint IDs
#
# 16 = left_shoulder
# 18 = left_elbow
# 20 = left_wrist
# ============================================================

LEFT_SHOULDER = 16
LEFT_ELBOW = 18
LEFT_WRIST = 20


# ============================================================
# Utilities
# ============================================================

def project_to_so3(M):
    """
    将可能存在少量数值误差的 3x3 矩阵投影到合法旋转矩阵 SO(3)。
    """

    U, _, Vt = np.linalg.svd(M)

    R = U @ Vt

    # 防止出现 det = -1 的反射矩阵
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    return R


def temporal_rotation_step_deg(R_all):
    """
    计算相邻帧旋转矩阵之间的旋转角度。

    input:
        R_all: [N, 3, 3]

    output:
        step_deg: [N-1]
    """

    steps = []

    for i in range(1, len(R_all)):

        # frame i-1 -> frame i 的相对旋转
        R_rel = (
            R_all[i - 1].T
            @ R_all[i]
        )

        rotvec = (
            Rotation
            .from_matrix(R_rel)
            .as_rotvec()
        )

        angle_deg = np.degrees(
            np.linalg.norm(rotvec)
        )

        steps.append(
            angle_deg
        )

    return np.asarray(
        steps
    )


def print_statistics(
    name,
    step_deg,
):

    print()
    print(
        f"========== {name} =========="
    )

    print(
        "mean step:",
        step_deg.mean(),
        "deg"
    )

    print(
        "max step :",
        step_deg.max(),
        "deg"
    )

    print(
        ">10 deg  :",
        np.sum(step_deg > 10)
    )

    print(
        ">30 deg  :",
        np.sum(step_deg > 30)
    )

    print(
        ">60 deg  :",
        np.sum(step_deg > 60)
    )

    print(
        ">90 deg  :",
        np.sum(step_deg > 90)
    )


# ============================================================
# Load HybrIK-X result
# ============================================================

data = torch.load(
    PT_PATH,
    map_location="cpu",
    # weights_only=False,
)


if "pred_theta_mat" not in data:
    raise KeyError(
        "pred_theta_mat not found in HybrIK-X output."
    )


theta = np.asarray(
    data["pred_theta_mat"]
)


print(
    "========== Input =========="
)

print(
    "PT path:",
    PT_PATH
)

print(
    "raw pred_theta_mat shape:",
    theta.shape
)


# ============================================================
# Reshape
#
# HybrIK-X output:
#
#   pred_theta_mat = [N, 495]
#
# 495 = 55 * 3 * 3
#
# therefore:
#
#   [N, 55, 3, 3]
# ============================================================

if theta.ndim == 2:

    expected_dim = (
        55 * 3 * 3
    )

    if theta.shape[1] != expected_dim:

        raise ValueError(
            f"Expected second dim {expected_dim}, "
            f"got {theta.shape}"
        )

    theta = theta.reshape(
        theta.shape[0],
        55,
        3,
        3,
    )

elif theta.ndim == 4:

    if theta.shape[1:] != (
        55,
        3,
        3,
    ):
        raise ValueError(
            f"Unexpected theta shape: {theta.shape}"
        )

else:

    raise ValueError(
        f"Unexpected pred_theta_mat shape: {theta.shape}"
    )


print(
    "reshaped theta:",
    theta.shape
)

num_frames = theta.shape[0]

print(
    "frames:",
    num_frames
)


# ============================================================
# Check rotation matrices
# ============================================================

JOINTS_TO_CHECK = [
    (
        LEFT_SHOULDER,
        "left_shoulder",
    ),
    (
        LEFT_ELBOW,
        "left_elbow",
    ),
    (
        LEFT_WRIST,
        "left_wrist",
    ),
]


for joint_id, joint_name in JOINTS_TO_CHECK:

    R_all = []

    for frame_idx in range(
        num_frames
    ):

        R_raw = theta[
            frame_idx,
            joint_id,
        ]

        R_valid = (
            project_to_so3(
                R_raw
            )
        )

        R_all.append(
            R_valid
        )


    R_all = np.asarray(
        R_all
    )


    step_deg = (
        temporal_rotation_step_deg(
            R_all
        )
    )


    print_statistics(
        joint_name,
        step_deg,
    )


    # 额外打印最大跳变位置
    max_idx = int(
        np.argmax(
            step_deg
        )
    )

    print(
        "max jump between frames:",
        max_idx,
        "->",
        max_idx + 1,
    )


# ============================================================
# Inspect frame 0 wrist rotation
# ============================================================

R_wrist_0 = (
    project_to_so3(
        theta[
            0,
            LEFT_WRIST,
        ]
    )
)


print()
print(
    "========== Left wrist frame 0 =========="
)

print(
    R_wrist_0
)


print()
print(
    "det:",
    np.linalg.det(
        R_wrist_0
    )
)


# ============================================================
# Extra: overall wrist motion relative to frame 0
# ============================================================

R_wrist_all = np.asarray([
    project_to_so3(
        theta[i, LEFT_WRIST]
    )
    for i in range(
        num_frames
    )
])


R_ref = (
    R_wrist_all[0]
)


relative_angle_deg = []

for i in range(
    num_frames
):

    R_rel = (
        R_ref.T
        @ R_wrist_all[i]
    )

    rotvec = (
        Rotation
        .from_matrix(
            R_rel
        )
        .as_rotvec()
    )

    angle = np.degrees(
        np.linalg.norm(
            rotvec
        )
    )

    relative_angle_deg.append(
        angle
    )


relative_angle_deg = (
    np.asarray(
        relative_angle_deg
    )
)


print()
print(
    "========== Left wrist motion relative to frame 0 =========="
)

print(
    "min:",
    relative_angle_deg.min(),
    "deg"
)

print(
    "max:",
    relative_angle_deg.max(),
    "deg"
)

print(
    "mean:",
    relative_angle_deg.mean(),
    "deg"
)

print(
    "max frame:",
    np.argmax(
        relative_angle_deg
    )
)