import argparse
import os

import mujoco
import numpy as np
import torch

from scipy.optimize import least_squares


# ============================================================
# Command-line arguments
# ============================================================

parser = argparse.ArgumentParser(
    description="Retarget HybrIK left-arm motion to G1 left-arm 4DoF."
)

# Paths
parser.add_argument(
    "--hybrik-pt-path",
    default="res/pick_place/hybrikx_output.pt",
    help="Path to hybrikx_output.pt",
)
parser.add_argument(
    "--retarget-path",
    default="res/pick_place/retarget_input.npz",
    help="Path to retarget_input.npz",
)
parser.add_argument(
    "--xml-path",
    default=(
        "/data/wx/code-IK/unitree_ros/"
        "robots/g1_description/g1_29dof.xml"
    ),
    help="Path to G1 MuJoCo MJCF/XML",
)
parser.add_argument(
    "--out-path",
    default="res/pick_place/g1_left_arm_4dof.npz",
    help="Output trajectory NPZ path",
)

# IK weights
parser.add_argument(
    "--elbow-pos-weight",
    type=float,
    default=20.0,
)
parser.add_argument(
    "--wrist-pos-weight",
    type=float,
    default=25.0,
)
parser.add_argument(
    "--upper-dir-weight",
    type=float,
    default=1.0,
)
parser.add_argument(
    "--fore-dir-weight",
    type=float,
    default=1.0,
)
parser.add_argument(
    "--temporal-weight",
    type=float,
    default=0.02,
)
parser.add_argument(
    "--max-nfev",
    type=int,
    default=150,
)

args = parser.parse_args()

HYBRIK_PT_PATH = args.hybrik_pt_path
RETARGET_PATH = args.retarget_path
XML_PATH = args.xml_path
OUT_PATH = args.out_path

ELBOW_POS_WEIGHT = args.elbow_pos_weight
WRIST_POS_WEIGHT = args.wrist_pos_weight
UPPER_DIR_WEIGHT = args.upper_dir_weight
FORE_DIR_WEIGHT = args.fore_dir_weight
TEMPORAL_WEIGHT = args.temporal_weight
MAX_NFEV = args.max_nfev


# ============================================================
# HybrIK joints
# ============================================================

SPINE3 = 9
NECK = 12

LEFT_SHOULDER = 16
LEFT_ELBOW = 18
LEFT_WRIST = 20


# ============================================================
# Human -> G1 mapping
#
# 当前保留已经验证过的 flip_forward：
#
# Human:
#   +x = right
#   +y = up
#   -z = forward
#
# G1:
#   +x = forward
#   +y = left
#   +z = up
#
# therefore:
#
#   G1_x = -Human_z
#   G1_y = -Human_x
#   G1_z =  Human_y
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

    v = np.asarray(
        v,
        dtype=np.float64,
    )

    return (
        v
        /
        (
            np.linalg.norm(v)
            + 1e-8
        )
    )


def angle_deg(a, b):

    a = normalize(a)
    b = normalize(b)

    dot = np.clip(
        np.dot(a, b),
        -1.0,
        1.0,
    )

    return np.degrees(
        np.arccos(dot)
    )


# ============================================================
# Load HybrIK
# ============================================================

hybrik_data = torch.load(
    HYBRIK_PT_PATH,
    map_location="cpu",
    # weights_only=False,
)

xyz = np.asarray(
    hybrik_data["pred_xyz_hybrik"],
    dtype=np.float64,
)


if xyz.ndim != 3:

    raise ValueError(
        f"Expected [N,71,3], got {xyz.shape}"
    )


num_frames = xyz.shape[0]


print(
    "========== Input =========="
)

print(
    "HybrIK xyz:",
    xyz.shape
)

print(
    "Frames:",
    num_frames
)


# ============================================================
# Load torso frame
# ============================================================

retarget = np.load(
    RETARGET_PATH
)

R_torso = np.asarray(
    retarget["torso_rotation"],
    dtype=np.float64,
)


if len(R_torso) != num_frames:

    raise ValueError(
        "Frame mismatch between HybrIK and retarget_input."
    )


# ============================================================
# Human joints -> torso-local coordinates
# ============================================================

torso_center_world = (
    xyz[:, SPINE3]
    +
    xyz[:, NECK]
) / 2.0


def world_to_torso_local(
    point_world,
):

    relative = (
        point_world
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
# Use shoulder-relative geometry
#
# 机器人肩膀不会跟人体肩膀整体平移，
# 所以这里重点保留：
#
# shoulder -> elbow
# shoulder -> wrist
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


human_upper_dir = (
    human_upper_vec
    /
    (
        human_upper_len[:, None]
        + 1e-8
    )
)

human_fore_dir = (
    human_fore_vec
    /
    (
        human_fore_len[:, None]
        + 1e-8
    )
)


# ============================================================
# Map Human coordinates to G1 coordinates
# ============================================================

mapped_upper_dir = (
    R_HUMAN_TO_G1
    @ human_upper_dir.T
).T

mapped_fore_dir = (
    R_HUMAN_TO_G1
    @ human_fore_dir.T
).T


# ============================================================
# MuJoCo model
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
# Resolve joint addresses / limits
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
            f"Joint not found: {name}"
        )


    qpos_adrs.append(
        model.jnt_qposadr[jid]
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


# ============================================================
# Resolve bodies
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
        "Failed to resolve G1 arm bodies."
    )


# ============================================================
# Correct default robot pose
#
# IMPORTANT:
# use model.qpos0 instead of zeroing floating base.
# ============================================================

def set_pose(q4):

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

    set_pose(q4)


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


    upper_vec = (
        elbow
        -
        shoulder
    )

    fore_vec = (
        wrist
        -
        elbow
    )


    upper_dir = normalize(
        upper_vec
    )

    fore_dir = normalize(
        fore_vec
    )


    return (
        shoulder,
        elbow,
        wrist,
        upper_dir,
        fore_dir,
    )


# ============================================================
# Determine G1 segment lengths
#
# Use model FK instead of hard-coding.
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


print()
print(
    "========== Robot arm geometry =========="
)

print(
    "G1 upper length:",
    G1_UPPER_LEN
)

print(
    "G1 forearm length:",
    G1_FORE_LEN
)


# ============================================================
# Human geometry -> Robot-scale position targets
#
# Important:
#
# 不直接使用 HybrIK 的绝对尺度，
# 而是使用人体方向 + G1 自身臂长。
#
# elbow target:
#
#   shoulder_robot
#   + human_upper_dir * robot_upper_len
#
# wrist target:
#
#   elbow_target
#   + human_fore_dir * robot_fore_len
#
# ============================================================

target_elbow_rel = (
    mapped_upper_dir
    *
    G1_UPPER_LEN
)


target_wrist_rel = (
    target_elbow_rel
    +
    mapped_fore_dir
    *
    G1_FORE_LEN
)


# ============================================================
# Inspect human target motion
# ============================================================

print()
print(
    "========== Target motion =========="
)


target_elbow_delta = (
    target_elbow_rel[-1]
    -
    target_elbow_rel[0]
)

target_wrist_delta = (
    target_wrist_rel[-1]
    -
    target_wrist_rel[0]
)


print(
    "elbow target frame0:",
    target_elbow_rel[0]
)

print(
    "wrist target frame0:",
    target_wrist_rel[0]
)


print()
print(
    "elbow target total delta:",
    target_elbow_delta
)

print(
    "wrist target total delta:",
    target_wrist_delta
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
)

fore_errors = np.zeros(
    num_frames,
)

elbow_pos_errors = np.zeros(
    num_frames,
)

wrist_pos_errors = np.zeros(
    num_frames,
)


q_prev = np.zeros(
    4,
    dtype=np.float64,
)


for frame_idx in range(
    num_frames
):


    target_upper = (
        mapped_upper_dir[
            frame_idx
        ]
    )

    target_fore = (
        mapped_fore_dir[
            frame_idx
        ]
    )


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


    def residual(q4):

        (
            shoulder,
            elbow,
            wrist,
            robot_upper,
            robot_fore,
        ) = get_robot_geometry(
            q4
        )


        # Robot positions relative to shoulder
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


        # ---------------------------------------------
        # Position loss
        # ---------------------------------------------

        r_elbow_pos = (
            np.sqrt(
                ELBOW_POS_WEIGHT
            )
            *
            (
                robot_elbow_rel
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
                robot_wrist_rel
                -
                target_wrist
            )
        )


        # ---------------------------------------------
        # Direction loss
        # ---------------------------------------------

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


        residuals = [
            r_elbow_pos,
            r_wrist_pos,
            r_upper_dir,
            r_fore_dir,
        ]


        # ---------------------------------------------
        # Temporal regularization
        #
        # Frame 0:
        # 不强制拉向 zero pose，
        # 避免第一帧出现额外误差。
        # ---------------------------------------------

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

            residuals.append(
                r_temporal
            )


        return np.concatenate(
            residuals
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
        robot_upper,
        robot_fore,
    ) = get_robot_geometry(
        q_opt
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
        robot_elbow_rel
        -
        target_elbow
    )


    wrist_pos_errors[
        frame_idx
    ] = np.linalg.norm(
        robot_wrist_rel
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
            f"upper={upper_errors[frame_idx]:6.2f} deg, "
            f"fore={fore_errors[frame_idx]:6.2f} deg, "
            f"elbow_pos={elbow_pos_errors[frame_idx]:.5f}, "
            f"wrist_pos={wrist_pos_errors[frame_idx]:.5f}"
        )


# ============================================================
# Joint temporal statistics
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
        <= lower[None, :]
        + LIMIT_MARGIN
    )
    |
    (
        q_traj
        >= upper[None, :]
        - LIMIT_MARGIN
    )
)


# ============================================================
# Save
# ============================================================

out_dir = os.path.dirname(OUT_PATH)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)

np.savez(
    OUT_PATH,

    q=q_traj,

    joint_names=np.asarray(
        JOINT_NAMES
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

    target_elbow_rel=(
        target_elbow_rel
    ),

    target_wrist_rel=(
        target_wrist_rel
    ),

    mapped_upper_dir=(
        mapped_upper_dir
    ),

    mapped_fore_dir=(
        mapped_fore_dir
    ),

    success=(
        success_flags
    ),

    lower_limit=lower,

    upper_limit=upper,
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


print()
print(
    "========== Angular error =========="
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


print()
print(
    "========== Position error =========="
)

print(
    "Elbow mean:",
    elbow_pos_errors.mean()
)

print(
    "Elbow max :",
    elbow_pos_errors.max()
)

print()

print(
    "Wrist mean:",
    wrist_pos_errors.mean()
)

print(
    "Wrist max :",
    wrist_pos_errors.max()
)


print()
print(
    "========== Joint ranges =========="
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
        "  model limit:",
        np.degrees(
            lower[i]
        ),
        "~",
        np.degrees(
            upper[i]
        ),
        "deg"
    )


print()
print(
    "========== Mean frame-to-frame step =========="
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
    "========== Max frame-to-frame step =========="
)

for name, value in zip(
    JOINT_NAMES,
    max_step_deg,
):

    print(
        f"{name:30s}: "
        f"{value:.3f} deg/frame"
    )


print()
print(
    "========== Near joint limits =========="
)

for i, name in enumerate(
    JOINT_NAMES
):

    count = int(
        near_limit[:, i].sum()
    )

    print(
        f"{name:30s}: "
        f"{count}/{num_frames}"
    )