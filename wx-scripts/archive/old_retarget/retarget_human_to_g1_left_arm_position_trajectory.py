import os

import mujoco
import numpy as np
import torch

from scipy.optimize import least_squares


# ============================================================
# Config
# ============================================================

HYBRIK_PT_PATH = (
    "res/pick_place/hybrikx_output.pt"
)

RETARGET_PATH = (
    "res/pick_place/retarget_input.npz"
)

XML_PATH = (
    "/data/wx/code-IK/unitree_ros/"
    "robots/g1_description/g1_29dof.xml"
)

OUT_PATH = (
    "res/pick_place/"
    "g1_left_arm_4dof_position_trajectory.npz"
)


# ============================================================
# IK weights
#
# position residual 的单位是 meter，
# direction residual 是无量纲。
#
# 因此 position 权重需要明显大一些。
#
# 第一版先用下面这组，不建议立刻乱调。
# ============================================================

ELBOW_POS_WEIGHT = 100.0
WRIST_POS_WEIGHT = 150.0

UPPER_DIR_WEIGHT = 0.5
FORE_DIR_WEIGHT = 0.5

TEMPORAL_WEIGHT = 0.02


MAX_NFEV = 200


# ============================================================
# Optional target smoothing
#
# 当前先关闭。
#
# 我们首先想验证真实 position trajectory 的方向是否正确，
# 不希望 smoothing 掩盖问题。
# ============================================================

SMOOTH_TARGETS = False

SMOOTH_WINDOW = 5


# ============================================================
# HybrIK / SMPL-X joint IDs
# ============================================================

SPINE3 = 9
NECK = 12

LEFT_SHOULDER = 16
LEFT_ELBOW = 18
LEFT_WRIST = 20


# ============================================================
# Human -> G1 coordinate mapping
#
# 我们目前保留已经验证的 flip_forward。
#
# Human torso-local:
#
#   +x = anatomical right
#   +y = up
#   -z = forward
#
# G1:
#
#   +x = forward
#   +y = left
#   +z = up
#
# therefore:
#
#   G1_x = -Human_z
#   G1_y = -Human_x
#   G1_z =  Human_y
#
# det(R) = +1
# ============================================================

R_HUMAN_TO_G1 = np.array(
    [
        [0.0,  0.0, -1.0],
        [-1.0, 0.0,  0.0],
        [0.0,  1.0,  0.0],
    ],
    dtype=np.float64,
)


# ============================================================
# Utilities
# ============================================================

def normalize(v):
    """
    Normalize vectors.

    Supports:
        [3]
        [N,3]
    """

    v = np.asarray(
        v,
        dtype=np.float64,
    )

    norm = np.linalg.norm(
        v,
        axis=-1,
        keepdims=True,
    )

    return (
        v
        /
        (
            norm
            + 1e-8
        )
    )


def angle_deg(a, b):
    """
    Angle between two 3D vectors.
    """

    a = normalize(a)
    b = normalize(b)

    dot = np.sum(
        a * b,
        axis=-1,
    )

    dot = np.clip(
        dot,
        -1.0,
        1.0,
    )

    return np.degrees(
        np.arccos(dot)
    )


def moving_average(x, window):
    """
    Simple centered temporal moving average.

    x: [N,D]
    """

    if window <= 1:
        return x.copy()

    if window % 2 == 0:
        raise ValueError(
            "SMOOTH_WINDOW must be odd."
        )

    pad = window // 2

    x_pad = np.pad(
        x,
        (
            (pad, pad),
            (0, 0),
        ),
        mode="edge",
    )

    output = np.zeros_like(
        x,
        dtype=np.float64,
    )

    for i in range(
        len(x)
    ):

        output[i] = np.mean(
            x_pad[
                i:
                i + window
            ],
            axis=0,
        )

    return output


# ============================================================
# Load HybrIK-X
# ============================================================

hybrik_data = torch.load(
    HYBRIK_PT_PATH,
    map_location="cpu",
    # weights_only=False,
)


if "pred_xyz_hybrik" not in hybrik_data:

    raise KeyError(
        "pred_xyz_hybrik is missing."
    )


xyz = np.asarray(
    hybrik_data[
        "pred_xyz_hybrik"
    ],
    dtype=np.float64,
)


if xyz.ndim == 2:

    # In case data was saved as [N, 213]
    if xyz.shape[1] != 71 * 3:

        raise ValueError(
            f"Unexpected xyz shape: {xyz.shape}"
        )

    xyz = xyz.reshape(
        xyz.shape[0],
        71,
        3,
    )


if (
    xyz.ndim != 3
    or xyz.shape[1:] != (71, 3)
):

    raise ValueError(
        f"Expected [N,71,3], got {xyz.shape}"
    )


num_frames = xyz.shape[0]


print(
    "=========================================="
)

print(
    "Input"
)

print(
    "=========================================="
)

print(
    "HybrIK:",
    HYBRIK_PT_PATH
)

print(
    "xyz shape:",
    xyz.shape
)

print(
    "frames:",
    num_frames
)


# ============================================================
# Load torso coordinate frame
# ============================================================

retarget = np.load(
    RETARGET_PATH,
    allow_pickle=True,
)


if "torso_rotation" not in retarget:

    raise KeyError(
        "torso_rotation missing from "
        "retarget_input.npz"
    )


R_torso = np.asarray(
    retarget[
        "torso_rotation"
    ],
    dtype=np.float64,
)


if R_torso.shape != (
    num_frames,
    3,
    3,
):

    raise ValueError(
        "Unexpected torso_rotation shape: "
        f"{R_torso.shape}"
    )


# ============================================================
# Human joints in torso-local frame
#
# First obtain a torso center.
#
# Note:
# shoulder-relative coordinates below make torso-center
# translation cancel out, but converting all joints through
# the same torso frame keeps the implementation explicit.
# ============================================================

torso_center_world = (
    xyz[:, SPINE3]
    +
    xyz[:, NECK]
) / 2.0


def world_to_torso_local(
    points_world,
):

    relative = (
        points_world
        -
        torso_center_world
    )

    return np.einsum(
        "nij,nj->ni",
        np.transpose(
            R_torso,
            (0, 2, 1),
        ),
        relative,
    )


shoulder_h = world_to_torso_local(
    xyz[:, LEFT_SHOULDER]
)

elbow_h = world_to_torso_local(
    xyz[:, LEFT_ELBOW]
)

wrist_h = world_to_torso_local(
    xyz[:, LEFT_WRIST]
)


# ============================================================
# Human shoulder-relative position trajectories
#
# THIS is the key difference from the previous script.
#
# We preserve the actual position vectors from HybrIK:
#
#   shoulder -> elbow
#   shoulder -> wrist
#
# We do NOT reconstruct them from normalized directions.
# ============================================================

human_elbow_rel = (
    elbow_h
    -
    shoulder_h
)

human_wrist_rel = (
    wrist_h
    -
    shoulder_h
)


human_upper_vec = (
    elbow_h
    -
    shoulder_h
)

human_fore_vec = (
    wrist_h
    -
    elbow_h
)


human_upper_len = np.linalg.norm(
    human_upper_vec,
    axis=1,
)

human_fore_len = np.linalg.norm(
    human_fore_vec,
    axis=1,
)


human_upper_dir = normalize(
    human_upper_vec
)

human_fore_dir = normalize(
    human_fore_vec
)


# ============================================================
# Human geometry statistics
#
# Use MEDIAN rather than a single frame.
#
# This gives a robust constant scale and prevents scale jitter
# from propagating frame-by-frame into the robot.
# ============================================================

HUMAN_UPPER_LEN = np.median(
    human_upper_len
)

HUMAN_FORE_LEN = np.median(
    human_fore_len
)

HUMAN_ARM_LEN = (
    HUMAN_UPPER_LEN
    +
    HUMAN_FORE_LEN
)


print()
print(
    "=========================================="
)

print(
    "Human arm geometry"
)

print(
    "=========================================="
)

print(
    "upper median:",
    HUMAN_UPPER_LEN
)

print(
    "fore median :",
    HUMAN_FORE_LEN
)

print(
    "total       :",
    HUMAN_ARM_LEN
)

print()

print(
    "upper min/max:",
    human_upper_len.min(),
    "~",
    human_upper_len.max(),
)

print(
    "fore min/max :",
    human_fore_len.min(),
    "~",
    human_fore_len.max(),
)


# ============================================================
# MuJoCo G1 model
# ============================================================

model = mujoco.MjModel.from_xml_path(
    XML_PATH
)

data = mujoco.MjData(
    model
)


JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
]


# ============================================================
# Joint mapping and limits
# ============================================================

qpos_adrs = []

lower = []
upper = []


for name in JOINT_NAMES:

    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    if jid < 0:

        raise RuntimeError(
            f"G1 joint not found: {name}"
        )


    qpos_adrs.append(
        int(
            model.jnt_qposadr[
                jid
            ]
        )
    )

    lower.append(
        model.jnt_range[
            jid,
            0
        ]
    )

    upper.append(
        model.jnt_range[
            jid,
            1
        ]
    )


lower = np.asarray(
    lower,
    dtype=np.float64,
)

upper = np.asarray(
    upper,
    dtype=np.float64,
)


print()
print(
    "=========================================="
)

print(
    "G1 joints"
)

print(
    "=========================================="
)


for i, name in enumerate(
    JOINT_NAMES
):

    print(
        f"{name:30s}: "
        f"{np.degrees(lower[i]):+.2f}"
        " ~ "
        f"{np.degrees(upper[i]):+.2f}"
        " deg"
    )


# ============================================================
# G1 body IDs
#
# Keep exactly the same geometric convention as the
# previously validated IK scripts.
# ============================================================

SHOULDER_BODY = (
    "left_shoulder_yaw_link"
)

ELBOW_BODY = (
    "left_elbow_link"
)

WRIST_BODY = (
    "left_wrist_roll_link"
)


shoulder_bid = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    SHOULDER_BODY,
)

elbow_bid = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    ELBOW_BODY,
)

wrist_bid = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    WRIST_BODY,
)


if min(
    shoulder_bid,
    elbow_bid,
    wrist_bid,
) < 0:

    raise RuntimeError(
        "Failed to find G1 left-arm bodies."
    )


# ============================================================
# G1 FK
#
# IMPORTANT:
#
# Always start from model.qpos0.
#
# Do NOT use:
#
#     data.qpos[:] = 0
#
# Otherwise the floating base is put into the ground.
# ============================================================

def set_robot_pose(q4):

    data.qpos[:] = (
        model.qpos0.copy()
    )

    for adr, value in zip(
        qpos_adrs,
        q4,
    ):

        data.qpos[
            adr
        ] = value


    mujoco.mj_forward(
        model,
        data,
    )


def get_robot_geometry(q4):

    set_robot_pose(
        q4
    )


    shoulder = (
        data.xpos[
            shoulder_bid
        ].copy()
    )

    elbow = (
        data.xpos[
            elbow_bid
        ].copy()
    )

    wrist = (
        data.xpos[
            wrist_bid
        ].copy()
    )


    robot_upper_vec = (
        elbow
        -
        shoulder
    )

    robot_fore_vec = (
        wrist
        -
        elbow
    )


    robot_upper_dir = normalize(
        robot_upper_vec
    )

    robot_fore_dir = normalize(
        robot_fore_vec
    )


    robot_elbow_rel = (
        elbow
        -
        shoulder
    )

    robot_wrist_rel = (
        wrist
        -
        shoulder
    )


    return (
        shoulder,
        elbow,
        wrist,
        robot_elbow_rel,
        robot_wrist_rel,
        robot_upper_dir,
        robot_fore_dir,
    )


# ============================================================
# Measure G1 arm lengths from FK
# ============================================================

q_zero = np.zeros(
    4,
    dtype=np.float64,
)


(
    shoulder0,
    elbow0,
    wrist0,
    _,
    _,
    _,
    _,
) = get_robot_geometry(
    q_zero
)


G1_UPPER_LEN = np.linalg.norm(
    elbow0
    -
    shoulder0
)

G1_FORE_LEN = np.linalg.norm(
    wrist0
    -
    elbow0
)

G1_ARM_LEN = (
    G1_UPPER_LEN
    +
    G1_FORE_LEN
)


print()
print(
    "=========================================="
)

print(
    "G1 arm geometry"
)

print(
    "=========================================="
)

print(
    "upper:",
    G1_UPPER_LEN
)

print(
    "fore :",
    G1_FORE_LEN
)

print(
    "total:",
    G1_ARM_LEN
)


# ============================================================
# Constant Human -> G1 scale
#
# IMPORTANT:
#
# one SINGLE scale for the entire arm.
#
# Unlike the previous implementation,
# we do NOT separately force:
#
# human upper direction * G1 upper length
# human fore  direction * G1 fore length
#
# A single global scale preserves human elbow/wrist trajectory
# geometry as much as possible.
# ============================================================

SCALE = (
    G1_ARM_LEN
    /
    HUMAN_ARM_LEN
)


print()
print(
    "=========================================="
)

print(
    "Human -> G1 scale"
)

print(
    "=========================================="
)

print(
    "scale:",
    SCALE
)


# ============================================================
# Map the REAL human position trajectories
#
# human_elbow_rel:
#
#     shoulder -> elbow
#
# human_wrist_rel:
#
#     shoulder -> wrist
#
# Both retain their original HybrIK 3D geometry.
# ============================================================

target_elbow_rel = (
    SCALE
    *
    (
        R_HUMAN_TO_G1
        @ human_elbow_rel.T
    ).T
)


target_wrist_rel = (
    SCALE
    *
    (
        R_HUMAN_TO_G1
        @ human_wrist_rel.T
    ).T
)


# Direction targets still come from the human skeleton,
# but they are now only secondary constraints.

target_upper_dir = (
    R_HUMAN_TO_G1
    @ human_upper_dir.T
).T


target_fore_dir = (
    R_HUMAN_TO_G1
    @ human_fore_dir.T
).T


target_upper_dir = normalize(
    target_upper_dir
)

target_fore_dir = normalize(
    target_fore_dir
)


# ============================================================
# Optional temporal smoothing of TARGET POSITION only
# ============================================================

if SMOOTH_TARGETS:

    print()
    print(
        "Smoothing targets with window:",
        SMOOTH_WINDOW
    )

    target_elbow_rel = (
        moving_average(
            target_elbow_rel,
            SMOOTH_WINDOW,
        )
    )

    target_wrist_rel = (
        moving_average(
            target_wrist_rel,
            SMOOTH_WINDOW,
        )
    )


# ============================================================
# Target sanity check
# ============================================================

print()
print(
    "=========================================="
)

print(
    "Target trajectory sanity check"
)

print(
    "=========================================="
)


CHECK_FRAMES = [
    0,
    20,
    40,
    60,
    100,
    140,
    180,
    220,
    260,
    300,
    num_frames - 1,
]


for frame in CHECK_FRAMES:

    if frame >= num_frames:
        continue

    print()
    print(
        f"frame {frame}"
    )

    print(
        " elbow target:",
        np.round(
            target_elbow_rel[
                frame
            ],
            5,
        )
    )

    print(
        " wrist target:",
        np.round(
            target_wrist_rel[
                frame
            ],
            5,
        )
    )


# ============================================================
# Explicit frame 0 -> 60 displacement
#
# Since the original video starts with a forward reach,
# this is useful for checking semantic direction.
# ============================================================

if num_frames > 60:

    print()
    print(
        "=========================================="
    )

    print(
        "Target frame 0 -> 60"
    )

    print(
        "=========================================="
    )

    print(
        "elbow delta:",
        target_elbow_rel[60]
        -
        target_elbow_rel[0]
    )

    print(
        "wrist delta:",
        target_wrist_rel[60]
        -
        target_wrist_rel[0]
    )

    print()

    print(
        "Remember:"
    )

    print(
        "G1 +x = forward"
    )

    print(
        "G1 +y = left"
    )

    print(
        "G1 +z = up"
    )


# ============================================================
# Sequential IK
# ============================================================

q_traj = np.zeros(
    (
        num_frames,
        4,
    ),
    dtype=np.float64,
)


success_flags = np.zeros(
    num_frames,
    dtype=bool,
)


upper_errors = np.zeros(
    num_frames,
    dtype=np.float64,
)

fore_errors = np.zeros(
    num_frames,
    dtype=np.float64,
)

elbow_pos_errors = np.zeros(
    num_frames,
    dtype=np.float64,
)

wrist_pos_errors = np.zeros(
    num_frames,
    dtype=np.float64,
)


actual_elbow_rel = np.zeros(
    (
        num_frames,
        3,
    ),
    dtype=np.float64,
)

actual_wrist_rel = np.zeros(
    (
        num_frames,
        3,
    ),
    dtype=np.float64,
)


q_prev = np.zeros(
    4,
    dtype=np.float64,
)


for frame_idx in range(
    num_frames
):

    target_elbow = (
        target_elbow_rel[
            frame_idx
        ]
    )

    target_wrist = (
        target_wrist_rel[
            frame_idx
        ]
    )

    target_upper = (
        target_upper_dir[
            frame_idx
        ]
    )

    target_fore = (
        target_fore_dir[
            frame_idx
        ]
    )


    def residual(q4):

        (
            shoulder,
            elbow,
            wrist,
            robot_elbow,
            robot_wrist,
            robot_upper,
            robot_fore,
        ) = get_robot_geometry(
            q4
        )


        # ====================================================
        # True position trajectory constraints
        # ====================================================

        r_elbow_pos = (
            np.sqrt(
                ELBOW_POS_WEIGHT
            )
            *
            (
                robot_elbow
                -
                target_elbow
            )
        )


        r_wrist_pos = (
            np.sqrt(
                WRIST_POS_WEIGHT
            )
            *
            (
                robot_wrist
                -
                target_wrist
            )
        )


        # ====================================================
        # Secondary direction constraints
        # ====================================================

        r_upper_dir = (
            np.sqrt(
                UPPER_DIR_WEIGHT
            )
            *
            (
                robot_upper
                -
                target_upper
            )
        )


        r_fore_dir = (
            np.sqrt(
                FORE_DIR_WEIGHT
            )
            *
            (
                robot_fore
                -
                target_fore
            )
        )


        residual_list = [
            r_elbow_pos,
            r_wrist_pos,
            r_upper_dir,
            r_fore_dir,
        ]


        # ====================================================
        # Temporal regularization
        #
        # Do NOT regularize frame 0 against zero pose.
        # ====================================================

        if frame_idx > 0:

            r_temporal = (
                np.sqrt(
                    TEMPORAL_WEIGHT
                )
                *
                (
                    q4
                    -
                    q_prev
                )
            )

            residual_list.append(
                r_temporal
            )


        return np.concatenate(
            residual_list
        )


    result = least_squares(
        residual,
        q_prev,
        bounds=(
            lower,
            upper,
        ),
        max_nfev=MAX_NFEV,
        xtol=1e-9,
        ftol=1e-9,
        gtol=1e-9,
    )


    q_opt = result.x


    q_traj[
        frame_idx
    ] = q_opt


    success_flags[
        frame_idx
    ] = result.success


    (
        shoulder,
        elbow,
        wrist,
        robot_elbow,
        robot_wrist,
        robot_upper,
        robot_fore,
    ) = get_robot_geometry(
        q_opt
    )


    actual_elbow_rel[
        frame_idx
    ] = robot_elbow


    actual_wrist_rel[
        frame_idx
    ] = robot_wrist


    upper_errors[
        frame_idx
    ] = angle_deg(
        robot_upper,
        target_upper,
    )


    fore_errors[
        frame_idx
    ] = angle_deg(
        robot_fore,
        target_fore,
    )


    elbow_pos_errors[
        frame_idx
    ] = np.linalg.norm(
        robot_elbow
        -
        target_elbow
    )


    wrist_pos_errors[
        frame_idx
    ] = np.linalg.norm(
        robot_wrist
        -
        target_wrist
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
            f"upper={upper_errors[frame_idx]:6.2f} deg  "
            f"fore={fore_errors[frame_idx]:6.2f} deg  "
            f"elbow_pos="
            f"{elbow_pos_errors[frame_idx]:.5f} m  "
            f"wrist_pos="
            f"{wrist_pos_errors[frame_idx]:.5f} m"
        )


# ============================================================
# Temporal statistics
# ============================================================

dq = np.diff(
    q_traj,
    axis=0,
)


mean_step_deg = np.degrees(
    np.mean(
        np.abs(dq),
        axis=0,
    )
)


max_step_deg = np.degrees(
    np.max(
        np.abs(dq),
        axis=0,
    )
)


# ============================================================
# Joint-limit statistics
# ============================================================

LIMIT_MARGIN = np.deg2rad(
    1.0
)


near_limit = (
    (
        q_traj
        <= lower[
            None,
            :
        ]
        + LIMIT_MARGIN
    )
    |
    (
        q_traj
        >= upper[
            None,
            :
        ]
        - LIMIT_MARGIN
    )
)


# ============================================================
# Position trajectory displacement diagnostics
# ============================================================

target_elbow_delta = (
    target_elbow_rel
    -
    target_elbow_rel[0]
)

target_wrist_delta = (
    target_wrist_rel
    -
    target_wrist_rel[0]
)


actual_elbow_delta = (
    actual_elbow_rel
    -
    actual_elbow_rel[0]
)

actual_wrist_delta = (
    actual_wrist_rel
    -
    actual_wrist_rel[0]
)


# ============================================================
# Save
# ============================================================

os.makedirs(
    os.path.dirname(
        OUT_PATH
    ),
    exist_ok=True,
)


np.savez(
    OUT_PATH,

    q=q_traj,

    joint_names=np.asarray(
        JOINT_NAMES
    ),

    success=success_flags,

    scale=np.asarray(
        SCALE
    ),

    human_upper_length=np.asarray(
        HUMAN_UPPER_LEN
    ),

    human_fore_length=np.asarray(
        HUMAN_FORE_LEN
    ),

    g1_upper_length=np.asarray(
        G1_UPPER_LEN
    ),

    g1_fore_length=np.asarray(
        G1_FORE_LEN
    ),

    target_elbow_rel=(
        target_elbow_rel
    ),

    target_wrist_rel=(
        target_wrist_rel
    ),

    actual_elbow_rel=(
        actual_elbow_rel
    ),

    actual_wrist_rel=(
        actual_wrist_rel
    ),

    target_elbow_delta=(
        target_elbow_delta
    ),

    target_wrist_delta=(
        target_wrist_delta
    ),

    actual_elbow_delta=(
        actual_elbow_delta
    ),

    actual_wrist_delta=(
        actual_wrist_delta
    ),

    target_upper_dir=(
        target_upper_dir
    ),

    target_fore_dir=(
        target_fore_dir
    ),

    upper_error_deg=(
        upper_errors
    ),

    fore_error_deg=(
        fore_errors
    ),

    elbow_position_error=(
        elbow_pos_errors
    ),

    wrist_position_error=(
        wrist_pos_errors
    ),

    lower_limit=lower,

    upper_limit=upper,

    mapping=(
        R_HUMAN_TO_G1
    ),
)


# ============================================================
# Summary
# ============================================================

print()
print(
    "=========================================="
)

print(
    "Done"
)

print(
    "=========================================="
)

print(
    "Saved:",
    OUT_PATH
)

print(
    "q shape:",
    q_traj.shape
)


print()
print(
    "IK success:"
)

print(
    np.sum(
        success_flags
    ),
    "/",
    num_frames
)


# ============================================================
# Errors
# ============================================================

print()
print(
    "=========================================="
)

print(
    "Position error"
)

print(
    "=========================================="
)

print(
    "Elbow mean:",
    elbow_pos_errors.mean(),
    "m"
)

print(
    "Elbow max :",
    elbow_pos_errors.max(),
    "m"
)

print()

print(
    "Wrist mean:",
    wrist_pos_errors.mean(),
    "m"
)

print(
    "Wrist max :",
    wrist_pos_errors.max(),
    "m"
)


print()
print(
    "=========================================="
)

print(
    "Angular error"
)

print(
    "=========================================="
)

print(
    "Upper mean:",
    upper_errors.mean(),
    "deg"
)

print(
    "Upper max :",
    upper_errors.max(),
    "deg"
)

print()

print(
    "Fore mean:",
    fore_errors.mean(),
    "deg"
)

print(
    "Fore max :",
    fore_errors.max(),
    "deg"
)


# ============================================================
# Joint ranges
# ============================================================

print()
print(
    "=========================================="
)

print(
    "Joint ranges"
)

print(
    "=========================================="
)


for i, name in enumerate(
    JOINT_NAMES
):

    print()
    print(
        name
    )

    print(
        "  min:",
        np.degrees(
            q_traj[:, i].min()
        ),
        "deg"
    )

    print(
        "  max:",
        np.degrees(
            q_traj[:, i].max()
        ),
        "deg"
    )

    print(
        "  limit:",
        np.degrees(
            lower[i]
        ),
        "~",
        np.degrees(
            upper[i]
        ),
        "deg"
    )


# ============================================================
# Temporal motion
# ============================================================

print()
print(
    "=========================================="
)

print(
    "Mean frame-to-frame step"
)

print(
    "=========================================="
)


for name, value in zip(
    JOINT_NAMES,
    mean_step_deg,
):

    print(
        f"{name:30s}: "
        f"{value:.3f} deg/frame"
    )


print()
print(
    "=========================================="
)

print(
    "Max frame-to-frame step"
)

print(
    "=========================================="
)


for name, value in zip(
    JOINT_NAMES,
    max_step_deg,
):

    print(
        f"{name:30s}: "
        f"{value:.3f} deg/frame"
    )


# ============================================================
# Limits
# ============================================================

print()
print(
    "=========================================="
)

print(
    "Near joint limits"
)

print(
    "=========================================="
)


for i, name in enumerate(
    JOINT_NAMES
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


# ============================================================
# 0 -> 60 actual-vs-target trajectory check
# ============================================================

if num_frames > 60:

    print()
    print(
        "=========================================="
    )

    print(
        "FRAME 0 -> 60 position trajectory"
    )

    print(
        "=========================================="
    )

    print(
        "Target elbow delta:"
    )

    print(
        target_elbow_delta[60]
    )

    print(
        "Actual G1 elbow delta:"
    )

    print(
        actual_elbow_delta[60]
    )

    print()

    print(
        "Target wrist delta:"
    )

    print(
        target_wrist_delta[60]
    )

    print(
        "Actual G1 wrist delta:"
    )

    print(
        actual_wrist_delta[60]
    )

    print()

    print(
        "Coordinate reminder:"
    )

    print(
        "  +x = G1 forward"
    )

    print(
        "  +y = G1 left"
    )

    print(
        "  +z = G1 up"
    )