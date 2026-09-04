import mujoco
import numpy as np
import torch

from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


# ============================================================
# User config
# ============================================================

XML_PATH = (
    "/data/wx/code-IK/unitree_ros/"
    "robots/g1_description/g1_29dof.xml"
)

# 当前视频的 HybrIK-X 输出
HYBRIK_PT_PATH = (
    "res/pick_place/hybrikx_output.pt"
)

# 当前视频对应的 4DoF 左臂轨迹
ARM_TRAJ_PATH = (
    "res/pick_place/g1_left_arm_trajectory.npz"
)

# 最终 7DoF 输出
OUT_PATH = (
    "res/pick_place/g1_left_arm_7dof_trajectory.npz"
)


# Wrist temporal regularization
#
# 越大：
#   wrist 越平滑
#   但姿态跟踪误差可能更大
#
# 当前先用较小值
TEMPORAL_WEIGHT = 0.02

MAX_NFEV = 150


# ============================================================
# SMPL-X joint IDs
# ============================================================

PELVIS = 0

SPINE1 = 3
SPINE2 = 6
SPINE3 = 9

LEFT_COLLAR = 13
LEFT_SHOULDER = 16
LEFT_ELBOW = 18
LEFT_WRIST = 20


# ============================================================
# Human -> G1 coordinate convention
#
# Human torso:
#
#   +x = right
#   +y = up
#   +z = forward
#
# G1:
#
#   +x = forward
#   +y = left
#   +z = up
#
# ============================================================

R_HUMAN_TO_G1 = np.array(
    [
        [0.0,  0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0,  1.0, 0.0],
    ],
    dtype=np.float64,
)


# ============================================================
# Utilities
# ============================================================

def project_to_so3(M):
    """
    Project a noisy 3x3 matrix to SO(3).
    """

    U, _, Vt = np.linalg.svd(M)

    R = U @ Vt

    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    return R


def rotation_error_vector(
    R_target,
    R_actual,
):
    """
    SO(3) geodesic error represented as rotvec.
    """

    R_err = (
        R_target.T
        @ R_actual
    )

    return (
        Rotation
        .from_matrix(R_err)
        .as_rotvec()
    )


def rotation_error_deg(
    R_target,
    R_actual,
):
    rotvec = rotation_error_vector(
        R_target,
        R_actual,
    )

    return np.degrees(
        np.linalg.norm(rotvec)
    )


def relative_rotation_step_deg(
    rotations,
):
    """
    Frame-to-frame SO(3) rotation step.
    """

    result = []

    for i in range(
        1,
        len(rotations),
    ):

        R_rel = (
            rotations[i - 1].T
            @ rotations[i]
        )

        angle = np.linalg.norm(
            Rotation
            .from_matrix(R_rel)
            .as_rotvec()
        )

        result.append(
            np.degrees(angle)
        )

    return np.asarray(result)


# ============================================================
# Load HybrIK-X output
# ============================================================

hybrik_data = torch.load(
    HYBRIK_PT_PATH,
    map_location="cpu",
    # weights_only=False,
)


if "pred_theta_mat" not in hybrik_data:
    raise KeyError(
        "pred_theta_mat is missing from HybrIK-X output."
    )


theta = np.asarray(
    hybrik_data["pred_theta_mat"],
    dtype=np.float64,
)


print(
    "========== HybrIK =========="
)

print(
    "raw pred_theta_mat:",
    theta.shape
)


if theta.ndim == 2:

    if theta.shape[1] != (
        55 * 3 * 3
    ):
        raise ValueError(
            f"Unexpected theta shape: {theta.shape}"
        )

    theta = theta.reshape(
        theta.shape[0],
        55,
        3,
        3,
    )

elif theta.shape[1:] != (
    55,
    3,
    3,
):
    raise ValueError(
        f"Unexpected theta shape: {theta.shape}"
    )


num_frames = theta.shape[0]

print(
    "theta:",
    theta.shape
)

print(
    "frames:",
    num_frames
)


# ============================================================
# Project every rotation to SO(3)
# ============================================================

theta_so3 = np.empty_like(
    theta
)

for f in range(
    num_frames
):

    for j in range(
        55
    ):

        theta_so3[
            f,
            j,
        ] = project_to_so3(
            theta[f, j]
        )


# ============================================================
# Load existing stable 4DoF trajectory
# ============================================================

arm_data = np.load(
    ARM_TRAJ_PATH,
    allow_pickle=True,
)

q_arm = np.asarray(
    arm_data["q"],
    dtype=np.float64,
)

arm_joint_names = [
    str(x)
    for x in arm_data[
        "joint_names"
    ]
]


print()
print(
    "========== Existing 4DoF arm =========="
)

print(
    "trajectory:",
    ARM_TRAJ_PATH
)

print(
    "q:",
    q_arm.shape
)

print(
    "joints:",
    arm_joint_names
)


if q_arm.ndim != 2:
    raise ValueError(
        f"q_arm must be [N,4], got {q_arm.shape}"
    )

if q_arm.shape[1] != 4:
    raise ValueError(
        f"Expected 4DoF arm, got {q_arm.shape}"
    )


# IMPORTANT:
# 不静默截断！
#
# 防止误用之前 290 帧 trajectory
# 配当前 362 帧 HybrIK 输出。
if len(q_arm) != num_frames:

    raise ValueError(
        "\nFrame number mismatch!\n"
        f"HybrIK frames = {num_frames}\n"
        f"4DoF trajectory frames = {len(q_arm)}\n\n"
        "Please regenerate "
        "g1_left_arm_trajectory.npz "
        "for the CURRENT video."
    )


# ============================================================
# Construct SMPL-X global rotations
#
# The left-wrist chain is:
#
# pelvis
#   -> spine1
#   -> spine2
#   -> spine3
#   -> left_collar
#   -> left_shoulder
#   -> left_elbow
#   -> left_wrist
#
# pred_theta_mat contains local joint rotations.
# ============================================================

LEFT_WRIST_CHAIN = [
    PELVIS,
    SPINE1,
    SPINE2,
    SPINE3,
    LEFT_COLLAR,
    LEFT_SHOULDER,
    LEFT_ELBOW,
    LEFT_WRIST,
]


TORSO_CHAIN = [
    PELVIS,
    SPINE1,
    SPINE2,
    SPINE3,
]


def compose_chain_rotation(
    frame_theta,
    chain,
):
    """
    Compose local SMPL-X rotations
    along a kinematic chain.
    """

    R = np.eye(
        3,
        dtype=np.float64,
    )

    for jid in chain:

        R = (
            R
            @ frame_theta[jid]
        )

    return project_to_so3(
        R
    )


# ============================================================
# Human wrist orientation relative to torso
# ============================================================

human_wrist_torso = []


for f in range(
    num_frames
):

    R_torso_global = (
        compose_chain_rotation(
            theta_so3[f],
            TORSO_CHAIN,
        )
    )

    R_wrist_global = (
        compose_chain_rotation(
            theta_so3[f],
            LEFT_WRIST_CHAIN,
        )
    )

    # Wrist orientation expressed relative to torso
    R_wrist_rel = (
        R_torso_global.T
        @ R_wrist_global
    )

    R_wrist_rel = (
        project_to_so3(
            R_wrist_rel
        )
    )


    # Convert human torso coordinate convention
    # into G1 torso convention.
    #
    # For an orientation matrix:
    #
    #   R_G1 = A R_H A^T
    #
    A = R_HUMAN_TO_G1

    R_wrist_g1_coords = (
        A
        @ R_wrist_rel
        @ A.T
    )

    R_wrist_g1_coords = (
        project_to_so3(
            R_wrist_g1_coords
        )
    )

    human_wrist_torso.append(
        R_wrist_g1_coords
    )


human_wrist_torso = np.asarray(
    human_wrist_torso
)


# ============================================================
# Human wrist temporal sanity check
# ============================================================

human_steps = (
    relative_rotation_step_deg(
        human_wrist_torso
    )
)


print()
print(
    "========== Human wrist orientation =========="
)

print(
    "mean frame step:",
    human_steps.mean(),
    "deg"
)

print(
    "max frame step :",
    human_steps.max(),
    "deg"
)

print(
    ">10 deg:",
    np.sum(
        human_steps > 10
    )
)

print(
    ">30 deg:",
    np.sum(
        human_steps > 30
    )
)


# ============================================================
# Load G1 MuJoCo model
# ============================================================

model = mujoco.MjModel.from_xml_path(
    XML_PATH
)

data = mujoco.MjData(
    model
)


ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
]

WRIST_JOINT_NAMES = [
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]

ALL_JOINT_NAMES = (
    ARM_JOINT_NAMES
    +
    WRIST_JOINT_NAMES
)


# ============================================================
# Resolve joint information
# ============================================================

def get_joint_info(name):

    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    if jid < 0:
        raise RuntimeError(
            f"Joint not found: {name}"
        )

    qadr = model.jnt_qposadr[
        jid
    ]

    low = model.jnt_range[
        jid,
        0
    ]

    high = model.jnt_range[
        jid,
        1
    ]

    return (
        jid,
        qadr,
        low,
        high,
    )


arm_qpos_adrs = []

for name in ARM_JOINT_NAMES:

    _, qadr, _, _ = (
        get_joint_info(name)
    )

    arm_qpos_adrs.append(
        qadr
    )


wrist_qpos_adrs = []

wrist_lower = []
wrist_upper = []


for name in WRIST_JOINT_NAMES:

    _, qadr, low, high = (
        get_joint_info(name)
    )

    wrist_qpos_adrs.append(
        qadr
    )

    wrist_lower.append(
        low
    )

    wrist_upper.append(
        high
    )


wrist_lower = np.asarray(
    wrist_lower
)

wrist_upper = np.asarray(
    wrist_upper
)


print()
print(
    "========== G1 wrist limits =========="
)

for i, name in enumerate(
    WRIST_JOINT_NAMES
):

    print(
        f"{name:28s}: "
        f"{np.degrees(wrist_lower[i]):+.2f}"
        " ~ "
        f"{np.degrees(wrist_upper[i]):+.2f}"
        " deg"
    )


# ============================================================
# Body IDs
#
# We compare orientation RELATIVE TO TORSO,
# not absolute world orientation.
# ============================================================

TORSO_BODY = (
    "torso_link"
)

HAND_BODY = (
    "left_wrist_yaw_link"
)


torso_bid = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    TORSO_BODY,
)

hand_bid = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    HAND_BODY,
)


if torso_bid < 0:
    raise RuntimeError(
        f"Body not found: {TORSO_BODY}"
    )

if hand_bid < 0:
    raise RuntimeError(
        f"Body not found: {HAND_BODY}"
    )


# ============================================================
# Initialize G1 state
# ============================================================

data.qpos[:] = 0.0


# Floating base quaternion = identity
if model.nq >= 7:

    data.qpos[3] = 1.0
    data.qpos[4] = 0.0
    data.qpos[5] = 0.0
    data.qpos[6] = 0.0


mujoco.mj_forward(
    model,
    data,
)


# ============================================================
# G1 FK
# ============================================================

def set_g1_pose(
    q_arm_frame,
    q_wrist_frame,
):

    for adr, value in zip(
        arm_qpos_adrs,
        q_arm_frame,
    ):

        data.qpos[adr] = value


    for adr, value in zip(
        wrist_qpos_adrs,
        q_wrist_frame,
    ):

        data.qpos[adr] = value


    mujoco.mj_forward(
        model,
        data,
    )


def get_g1_hand_torso_rotation(
    q_arm_frame,
    q_wrist_frame,
):
    """
    Get wrist-yaw-link orientation
    relative to G1 torso.
    """

    set_g1_pose(
        q_arm_frame,
        q_wrist_frame,
    )


    R_torso_world = (
        data.xmat[
            torso_bid
        ]
        .reshape(
            3,
            3,
        )
        .copy()
    )


    R_hand_world = (
        data.xmat[
            hand_bid
        ]
        .reshape(
            3,
            3,
        )
        .copy()
    )


    R_rel = (
        R_torso_world.T
        @ R_hand_world
    )


    return project_to_so3(
        R_rel
    )


# ============================================================
# Calibration at frame 0
#
# We do NOT assume:
#
# Human wrist frame == G1 wrist frame.
#
# Instead:
#
# 1. Obtain human frame-0 wrist orientation.
# 2. Obtain G1 frame-0 wrist orientation with wrist q=0.
# 3. Transfer only the HUMAN RELATIVE ROTATION
#    after frame 0.
# ============================================================

q_wrist_zero = np.zeros(
    3,
    dtype=np.float64,
)


R_human_0 = (
    human_wrist_torso[0]
)


R_g1_0 = (
    get_g1_hand_torso_rotation(
        q_arm[0],
        q_wrist_zero,
    )
)


print()
print(
    "========== Frame-0 calibration =========="
)

print(
    "Human wrist torso R:"
)
print(
    R_human_0
)

print()

print(
    "G1 hand torso R:"
)
print(
    R_g1_0
)


# ============================================================
# Build G1 orientation targets
#
# Human relative change:
#
#   ΔR_H(t) = R_H(0)^T R_H(t)
#
# Apply the same relative change to the
# G1 frame-0 wrist orientation:
#
#   R_target(t) = R_G1(0) ΔR_H(t)
#
# ============================================================

target_rotations = []


for f in range(
    num_frames
):

    delta_human = (
        R_human_0.T
        @ human_wrist_torso[f]
    )

    delta_human = (
        project_to_so3(
            delta_human
        )
    )


    R_target = (
        R_g1_0
        @ delta_human
    )


    R_target = (
        project_to_so3(
            R_target
        )
    )


    target_rotations.append(
        R_target
    )


target_rotations = np.asarray(
    target_rotations
)


# ============================================================
# Sequential wrist IK
# ============================================================

q_wrist_traj = np.zeros(
    (
        num_frames,
        3,
    ),
    dtype=np.float64,
)


orientation_errors = np.zeros(
    num_frames,
    dtype=np.float64,
)


success_flags = np.zeros(
    num_frames,
    dtype=bool,
)


costs = np.zeros(
    num_frames,
    dtype=np.float64,
)


q_prev = np.zeros(
    3,
    dtype=np.float64,
)


for frame_idx in range(
    num_frames
):

    q_arm_frame = (
        q_arm[frame_idx]
    )

    R_target = (
        target_rotations[
            frame_idx
        ]
    )


    def residual(
        q_wrist,
    ):

        R_actual = (
            get_g1_hand_torso_rotation(
                q_arm_frame,
                q_wrist,
            )
        )


        r_orientation = (
            rotation_error_vector(
                R_target,
                R_actual,
            )
        )


        # Important:
        #
        # Frame 0 should be exactly calibrated.
        # Do not pull it toward any previous state.
        if frame_idx == 0:

            return (
                r_orientation
            )


        r_temporal = (
            np.sqrt(
                TEMPORAL_WEIGHT
            )
            *
            (
                q_wrist
                -
                q_prev
            )
        )


        return np.concatenate(
            [
                r_orientation,
                r_temporal,
            ]
        )


    result = least_squares(
        residual,
        q_prev,
        bounds=(
            wrist_lower,
            wrist_upper,
        ),
        max_nfev=MAX_NFEV,
        xtol=1e-9,
        ftol=1e-9,
        gtol=1e-9,
    )


    q_opt = result.x


    q_wrist_traj[
        frame_idx
    ] = q_opt


    success_flags[
        frame_idx
    ] = result.success


    costs[
        frame_idx
    ] = result.cost


    R_actual = (
        get_g1_hand_torso_rotation(
            q_arm_frame,
            q_opt,
        )
    )


    orientation_errors[
        frame_idx
    ] = (
        rotation_error_deg(
            R_target,
            R_actual,
        )
    )


    q_prev = (
        q_opt.copy()
    )


    if (
        frame_idx % 50 == 0
        or frame_idx
        == num_frames - 1
    ):

        print(
            f"[{frame_idx:4d}/{num_frames}] "
            f"wrist_err="
            f"{orientation_errors[frame_idx]:6.2f} deg "
            f"q_deg="
            f"{np.degrees(q_opt)}"
        )


# ============================================================
# Combine into final 7DoF trajectory
# ============================================================

q7 = np.concatenate(
    [
        q_arm,
        q_wrist_traj,
    ],
    axis=1,
)


# ============================================================
# Temporal statistics
# ============================================================

dq = np.diff(
    q7,
    axis=0,
)


max_step_deg = np.degrees(
    np.max(
        np.abs(dq),
        axis=0,
    )
)


mean_step_deg = np.degrees(
    np.mean(
        np.abs(dq),
        axis=0,
    )
)


# ============================================================
# Wrist limit statistics
# ============================================================

LIMIT_MARGIN = np.deg2rad(
    1.0
)


near_limit = (
    (
        q_wrist_traj
        <= wrist_lower[
            None,
            :
        ]
        + LIMIT_MARGIN
    )
    |
    (
        q_wrist_traj
        >= wrist_upper[
            None,
            :
        ]
        - LIMIT_MARGIN
    )
)


# ============================================================
# Save
# ============================================================

np.savez(
    OUT_PATH,

    q=q7,

    joint_names=np.asarray(
        ALL_JOINT_NAMES
    ),

    q_arm=q_arm,

    q_wrist=q_wrist_traj,

    human_wrist_rotation=(
        human_wrist_torso
    ),

    target_wrist_rotation=(
        target_rotations
    ),

    wrist_orientation_error_deg=(
        orientation_errors
    ),

    wrist_success=(
        success_flags
    ),

    wrist_cost=costs,

    human_wrist_step_deg=(
        human_steps
    ),

    wrist_lower_limit=(
        wrist_lower
    ),

    wrist_upper_limit=(
        wrist_upper
    ),
)


# ============================================================
# Summary
# ============================================================

print()
print(
    "========== Done =========="
)

print(
    "Saved:",
    OUT_PATH
)

print(
    "q shape:",
    q7.shape
)


print()
print(
    "========== Wrist IK =========="
)

print(
    "success:",
    np.sum(success_flags),
    "/",
    num_frames
)


print()
print(
    "========== Wrist orientation error =========="
)

print(
    "mean:",
    orientation_errors.mean(),
    "deg"
)

print(
    "max :",
    orientation_errors.max(),
    "deg"
)

print(
    ">5 deg:",
    np.sum(
        orientation_errors > 5
    )
)

print(
    ">10 deg:",
    np.sum(
        orientation_errors > 10
    )
)


print()
print(
    "========== Wrist ranges =========="
)

for i, name in enumerate(
    WRIST_JOINT_NAMES
):

    print(
        name
    )

    print(
        "  min:",
        np.degrees(
            q_wrist_traj[
                :,
                i
            ].min()
        ),
        "deg"
    )

    print(
        "  max:",
        np.degrees(
            q_wrist_traj[
                :,
                i
            ].max()
        ),
        "deg"
    )

    print(
        "  model limit:",
        np.degrees(
            wrist_lower[i]
        ),
        "~",
        np.degrees(
            wrist_upper[i]
        ),
        "deg"
    )


print()
print(
    "========== Mean frame-to-frame step =========="
)

for name, value in zip(
    ALL_JOINT_NAMES,
    mean_step_deg,
):

    print(
        f"{name:30s}: "
        f"{value:.3f} deg/frame"
    )


print()
print(
    "========== Max frame-to-frame step =========="
)

for name, value in zip(
    ALL_JOINT_NAMES,
    max_step_deg,
):

    print(
        f"{name:30s}: "
        f"{value:.3f} deg/frame"
    )


print()
print(
    "========== Wrist near joint limits =========="
)

for i, name in enumerate(
    WRIST_JOINT_NAMES
):

    count = int(
        near_limit[
            :,
            i
        ].sum()
    )

    print(
        f"{name:30s}: "
        f"{count}/{num_frames}"
    )