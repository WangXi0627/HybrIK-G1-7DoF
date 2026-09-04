import argparse
import os

import mujoco
import numpy as np
import torch


# ============================================================
# Command-line arguments
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Inspect shoulder-elbow-wrist arm-frame stability and "
        "Human-to-G1 frame-0 calibration."
    )
)

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
    "--g1-traj-path",
    default="res/pick_place/g1_left_arm_4dof.npz",
    help="Path to the verified G1 4DoF trajectory NPZ",
)

parser.add_argument(
    "--xml-path",
    default=(
        "/data/wx/code-IK/unitree_ros/"
        "robots/g1_description/g1_29dof.xml"
    ),
    help="Path to G1 MJCF/XML",
)

parser.add_argument(
    "--out-path",
    default="res/pick_place/left_arm_frame_calibration.npz",
    help="Output diagnostic/calibration NPZ",
)

parser.add_argument(
    "--degenerate-threshold",
    type=float,
    default=1e-4,
    help=(
        "Minimum projected elbow vector norm when constructing "
        "the arm frame"
    ),
)

args = parser.parse_args()


HYBRIK_PT_PATH = args.hybrik_pt_path
RETARGET_PATH = args.retarget_path
G1_TRAJ_PATH = args.g1_traj_path
XML_PATH = args.xml_path
OUT_PATH = args.out_path

DEGENERATE_THRESHOLD = args.degenerate_threshold


# ============================================================
# Joint IDs
# ============================================================

SPINE3 = 9
NECK = 12

LEFT_SHOULDER = 16
LEFT_ELBOW = 18
LEFT_WRIST = 20


# ============================================================
# Human -> G1 coordinate mapping
#
# Verified 4DoF mapping:
#
# Human torso:
#   +x = right
#   +y = up
#   -z = forward
#
# G1:
#   +x = forward
#   +y = left
#   +z = up
#
# G1_x = -Human_z
# G1_y = -Human_x
# G1_z =  Human_y
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
# Utils
# ============================================================

def normalize(v, eps=1e-8):

    v = np.asarray(
        v,
        dtype=np.float64,
    )

    n = np.linalg.norm(v)

    if n < eps:
        return None

    return v / n


def rotation_angle_deg(R):
    """
    Geodesic rotation angle of R in SO(3).
    """

    cos_theta = (
        np.trace(R) - 1.0
    ) / 2.0

    cos_theta = np.clip(
        cos_theta,
        -1.0,
        1.0,
    )

    return np.degrees(
        np.arccos(cos_theta)
    )


def rotation_sequence_step_deg(R_seq):
    """
    Frame-to-frame rotation magnitude.

    R_seq: [N,3,3]
    """

    values = []

    for i in range(
        1,
        len(R_seq),
    ):

        R_rel = (
            R_seq[i - 1].T
            @ R_seq[i]
        )

        values.append(
            rotation_angle_deg(
                R_rel
            )
        )

    return np.asarray(
        values,
        dtype=np.float64,
    )


def rotation_relative_frame0_deg(
    R_seq
):
    """
    Rotation magnitude relative to frame 0.
    """

    R0 = R_seq[0]

    values = []

    for R in R_seq:

        R_rel = (
            R0.T
            @ R
        )

        values.append(
            rotation_angle_deg(
                R_rel
            )
        )

    return np.asarray(
        values,
        dtype=np.float64,
    )


# ============================================================
# Arm frame construction
#
# shoulder -> wrist = primary forward axis z
#
# shoulder -> elbow is projected onto the plane
# perpendicular to z to determine the bending direction.
#
#
#            elbow
#              *
#             /
#            /
# shoulder *------------* wrist
#
#
# z = shoulder -> wrist
#
# x = elbow's perpendicular direction relative to z
#
# y = z x x
#
# columns of R:
#
#     [x, y, z]
#
# ============================================================

def build_arm_frame(
    shoulder,
    elbow,
    wrist,
    previous_R=None,
):

    sw = (
        wrist
        -
        shoulder
    )

    se = (
        elbow
        -
        shoulder
    )


    # --------------------------------------------------------
    # z: shoulder -> wrist
    # --------------------------------------------------------

    z_axis = normalize(
        sw
    )

    if z_axis is None:
        return None, np.nan


    # --------------------------------------------------------
    # remove component of shoulder->elbow along z
    #
    # This leaves elbow bending direction.
    # --------------------------------------------------------

    x_raw = (
        se
        -
        np.dot(
            se,
            z_axis,
        )
        *
        z_axis
    )

    proj_norm = np.linalg.norm(
        x_raw
    )


    if (
        proj_norm
        <
        DEGENERATE_THRESHOLD
    ):
        return None, proj_norm


    x_axis = (
        x_raw
        /
        proj_norm
    )


    # --------------------------------------------------------
    # y = z cross x
    # --------------------------------------------------------

    y_axis = normalize(
        np.cross(
            z_axis,
            x_axis,
        )
    )

    if y_axis is None:
        return None, proj_norm


    # --------------------------------------------------------
    # re-orthogonalize x
    # --------------------------------------------------------

    x_axis = normalize(
        np.cross(
            y_axis,
            z_axis,
        )
    )


    R = np.stack(
        [
            x_axis,
            y_axis,
            z_axis,
        ],
        axis=-1,
    )


    # --------------------------------------------------------
    # Numerical sanity
    # --------------------------------------------------------

    if (
        not np.all(
            np.isfinite(R)
        )
    ):
        return None, proj_norm


    # --------------------------------------------------------
    # Optional continuity check.
    #
    # With this construction we normally should NOT need
    # arbitrary sign flipping.
    #
    # We leave the frame untouched here, because changing
    # x/y signs would also change its physical meaning.
    # --------------------------------------------------------

    return R, proj_norm


def build_sequence(
    shoulders,
    elbows,
    wrists,
    name,
):

    frames = []

    valid = np.ones(
        len(shoulders),
        dtype=bool,
    )

    projection_norms = np.zeros(
        len(shoulders),
        dtype=np.float64,
    )


    previous_R = None


    for i in range(
        len(shoulders)
    ):

        R, proj_norm = (
            build_arm_frame(
                shoulders[i],
                elbows[i],
                wrists[i],
                previous_R=previous_R,
            )
        )

        projection_norms[i] = (
            proj_norm
        )


        if R is None:

            valid[i] = False

            # If one isolated frame degenerates,
            # use previous frame only for diagnostic continuity.
            if previous_R is None:

                raise RuntimeError(
                    f"{name}: frame {i} "
                    "cannot construct arm frame "
                    "and no previous valid frame exists."
                )

            R = previous_R.copy()


        frames.append(
            R
        )

        previous_R = R


    return (
        np.asarray(
            frames
        ),
        valid,
        projection_norms,
    )


# ============================================================
# Load HybrIK
# ============================================================

hybrik_data = torch.load(
    HYBRIK_PT_PATH,
    map_location="cpu",
)

xyz = np.asarray(
    hybrik_data[
        "pred_xyz_hybrik"
    ],
    dtype=np.float64,
)


if xyz.ndim != 3:

    raise ValueError(
        f"Expected [N,71,3], got {xyz.shape}"
    )


num_frames = len(xyz)


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
    "retarget:",
    RETARGET_PATH
)

print(
    "G1 trajectory:",
    G1_TRAJ_PATH
)

print(
    "MJCF:",
    XML_PATH
)

print(
    "frames:",
    num_frames
)


# ============================================================
# Human torso frame
# ============================================================

retarget = np.load(
    RETARGET_PATH,
    allow_pickle=True,
)

R_torso = np.asarray(
    retarget[
        "torso_rotation"
    ],
    dtype=np.float64,
)


if len(R_torso) != num_frames:

    raise ValueError(
        "HybrIK / retarget frame count mismatch."
    )


torso_center = (
    xyz[:, SPINE3]
    +
    xyz[:, NECK]
) / 2.0


def to_torso_local(
    points_world
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
            (
                0,
                2,
                1,
            ),
        ),
        relative,
    )


human_shoulder = (
    to_torso_local(
        xyz[
            :,
            LEFT_SHOULDER
        ]
    )
)

human_elbow = (
    to_torso_local(
        xyz[
            :,
            LEFT_ELBOW
        ]
    )
)

human_wrist = (
    to_torso_local(
        xyz[
            :,
            LEFT_WRIST
        ]
    )
)


# ============================================================
# Human arm frame
# ============================================================

(
    R_human,
    human_valid,
    human_proj_norm,
) = build_sequence(
    human_shoulder,
    human_elbow,
    human_wrist,
    name="Human",
)


# ============================================================
# Convert Human arm-frame orientation into G1 coordinates
#
# Every axis is transformed by R_HUMAN_TO_G1:
#
#     R_H_mapped = A @ R_H
# ============================================================

R_human_mapped = np.einsum(
    "ij,njk->nik",
    R_HUMAN_TO_G1,
    R_human,
)


# ============================================================
# Load G1 4DoF trajectory
# ============================================================

traj = np.load(
    G1_TRAJ_PATH,
    allow_pickle=True,
)

q = np.asarray(
    traj["q"],
    dtype=np.float64,
)

joint_names = [
    str(x)
    for x in traj[
        "joint_names"
    ]
]


if len(q) != num_frames:

    raise ValueError(
        f"Frame mismatch: "
        f"Human={num_frames}, G1={len(q)}"
    )


expected_names = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
]


if joint_names != expected_names:

    raise ValueError(
        "Unexpected G1 joint order:\n"
        f"{joint_names}"
    )


# ============================================================
# MuJoCo
# ============================================================

model = mujoco.MjModel.from_xml_path(
    XML_PATH
)

data = mujoco.MjData(
    model
)


qpos_adrs = []

for name in joint_names:

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
        model.jnt_qposadr[
            jid
        ]
    )


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
        "Failed to resolve G1 arm body IDs."
    )


# ============================================================
# G1 positions for every frame
# ============================================================

g1_shoulder = np.zeros(
    (
        num_frames,
        3,
    ),
    dtype=np.float64,
)

g1_elbow = np.zeros_like(
    g1_shoulder
)

g1_wrist = np.zeros_like(
    g1_shoulder
)


for frame_idx in range(
    num_frames
):

    # IMPORTANT:
    # preserve correct floating-base standing pose
    data.qpos[:] = (
        model.qpos0.copy()
    )


    for adr, value in zip(
        qpos_adrs,
        q[
            frame_idx
        ],
    ):

        data.qpos[
            adr
        ] = value


    mujoco.mj_forward(
        model,
        data,
    )


    g1_shoulder[
        frame_idx
    ] = data.xpos[
        shoulder_bid
    ]

    g1_elbow[
        frame_idx
    ] = data.xpos[
        elbow_bid
    ]

    g1_wrist[
        frame_idx
    ] = data.xpos[
        wrist_bid
    ]


# ============================================================
# G1 arm frame
# ============================================================

(
    R_g1,
    g1_valid,
    g1_proj_norm,
) = build_sequence(
    g1_shoulder,
    g1_elbow,
    g1_wrist,
    name="G1",
)


# ============================================================
# Frame-0 calibration
#
# We already converted Human arm frame into G1 world axes:
#
#       R_Hm(t)
#
# But Human and G1 arm frames can still have a constant
# zero-pose orientation offset.
#
# Find C such that:
#
#       C @ R_Hm(0) = R_G1(0)
#
# therefore:
#
#       C = R_G1(0) @ R_Hm(0)^T
#
# Then:
#
#       R_target(t) = C @ R_Hm(t)
#
# ============================================================

R_calib = (
    R_g1[0]
    @ R_human_mapped[0].T
)


print()
print(
    "=========================================="
)

print(
    "Frame-0 calibration"
)

print(
    "=========================================="
)

print(
    "R_calib:"
)

print(
    R_calib
)

print(
    "det(R_calib):",
    np.linalg.det(
        R_calib
    )
)

print(
    "calibration angle:",
    rotation_angle_deg(
        R_calib
    ),
    "deg"
)


# ============================================================
# Calibrated Human arm-frame targets
# ============================================================

R_target = np.einsum(
    "ij,njk->nik",
    R_calib,
    R_human_mapped,
)


# ============================================================
# Compare calibrated target vs actual G1 frame
# ============================================================

target_vs_g1_error = np.zeros(
    num_frames,
    dtype=np.float64,
)


for i in range(
    num_frames
):

    R_err = (
        R_target[i].T
        @ R_g1[i]
    )

    target_vs_g1_error[i] = (
        rotation_angle_deg(
            R_err
        )
    )


# ============================================================
# Temporal statistics
# ============================================================

human_step = (
    rotation_sequence_step_deg(
        R_human
    )
)

human_mapped_step = (
    rotation_sequence_step_deg(
        R_human_mapped
    )
)

target_step = (
    rotation_sequence_step_deg(
        R_target
    )
)

g1_step = (
    rotation_sequence_step_deg(
        R_g1
    )
)


human_rel0 = (
    rotation_relative_frame0_deg(
        R_human
    )
)

target_rel0 = (
    rotation_relative_frame0_deg(
        R_target
    )
)

g1_rel0 = (
    rotation_relative_frame0_deg(
        R_g1
    )
)


# ============================================================
# Report helper
# ============================================================

def report_steps(
    name,
    values,
):

    print()
    print(
        f"========== {name} =========="
    )

    print(
        "mean step:",
        values.mean(),
        "deg"
    )

    print(
        "max step :",
        values.max(),
        "deg"
    )

    print(
        ">10 deg  :",
        np.sum(
            values > 10
        )
    )

    print(
        ">20 deg  :",
        np.sum(
            values > 20
        )
    )

    print(
        ">30 deg  :",
        np.sum(
            values > 30
        )
    )

    print(
        ">60 deg  :",
        np.sum(
            values > 60
        )
    )

    if len(values) > 0:

        idx = int(
            np.argmax(
                values
            )
        )

        print(
            "max jump:",
            idx,
            "->",
            idx + 1,
        )


# ============================================================
# Print statistics
# ============================================================

report_steps(
    "Human arm frame temporal",
    human_step,
)

report_steps(
    "Calibrated Human target temporal",
    target_step,
)

report_steps(
    "G1 arm frame temporal",
    g1_step,
)


print()
print(
    "========== Arm-frame motion relative to frame 0 =========="
)

print(
    "Human max:",
    human_rel0.max(),
    "deg"
)

print(
    "Target max:",
    target_rel0.max(),
    "deg"
)

print(
    "G1 max:",
    g1_rel0.max(),
    "deg"
)


print()
print(
    "========== Calibrated Human target vs G1 =========="
)

print(
    "mean error:",
    target_vs_g1_error.mean(),
    "deg"
)

print(
    "max error :",
    target_vs_g1_error.max(),
    "deg"
)

print(
    ">10 deg   :",
    np.sum(
        target_vs_g1_error
        >
        10
    )
)

print(
    ">20 deg   :",
    np.sum(
        target_vs_g1_error
        >
        20
    )
)

print(
    ">30 deg   :",
    np.sum(
        target_vs_g1_error
        >
        30
    )
)


max_error_frame = int(
    np.argmax(
        target_vs_g1_error
    )
)

print(
    "max error frame:",
    max_error_frame
)


# ============================================================
# Degeneracy diagnostics
# ============================================================

print()
print(
    "========== Frame construction health =========="
)

print(
    "Human valid:",
    np.sum(
        human_valid
    ),
    "/",
    num_frames
)

print(
    "Human projected elbow norm:"
)

print(
    "  min :",
    np.nanmin(
        human_proj_norm
    )
)

print(
    "  mean:",
    np.nanmean(
        human_proj_norm
    )
)


print()
print(
    "G1 valid:",
    np.sum(
        g1_valid
    ),
    "/",
    num_frames
)

print(
    "G1 projected elbow norm:"
)

print(
    "  min :",
    np.nanmin(
        g1_proj_norm
    )
)

print(
    "  mean:",
    np.nanmean(
        g1_proj_norm
    )
)


# ============================================================
# Print selected frames
# ============================================================

check_frames = [
    0,
    20,
    40,
    60,
    100,
    150,
    200,
    250,
    300,
    num_frames - 1,
]


print()
print(
    "========== Selected frames =========="
)


for frame_idx in check_frames:

    if (
        frame_idx < 0
        or
        frame_idx >= num_frames
    ):
        continue


    print()
    print(
        "------------------------------------------"
    )

    print(
        f"Frame {frame_idx}"
    )

    print(
        "target-vs-G1 error:",
        target_vs_g1_error[
            frame_idx
        ],
        "deg"
    )

    print(
        "Human mapped arm z:",
        np.round(
            R_human_mapped[
                frame_idx,
                :,
                2,
            ],
            4,
        )
    )

    print(
        "Target arm z:",
        np.round(
            R_target[
                frame_idx,
                :,
                2,
            ],
            4,
        )
    )

    print(
        "G1 arm z:",
        np.round(
            R_g1[
                frame_idx,
                :,
                2,
            ],
            4,
        )
    )


# ============================================================
# Save
# ============================================================

output_dir = os.path.dirname(
    OUT_PATH
)

if output_dir:

    os.makedirs(
        output_dir,
        exist_ok=True,
    )


np.savez(
    OUT_PATH,

    # human arm frame in Human torso-local axes
    human_arm_frame=R_human,

    # Human arm frame after global Human -> G1 axis mapping
    human_arm_frame_mapped=(
        R_human_mapped
    ),

    # G1 arm frame reconstructed from the verified 4DoF motion
    g1_arm_frame=R_g1,

    # frame-0 orientation calibration
    calibration_rotation=R_calib,

    # calibrated Human target orientation in G1 coordinates
    calibrated_target_frame=(
        R_target
    ),

    # diagnostics
    target_vs_g1_error_deg=(
        target_vs_g1_error
    ),

    human_step_deg=(
        human_step
    ),

    g1_step_deg=(
        g1_step
    ),

    target_step_deg=(
        target_step
    ),

    human_relative_frame0_deg=(
        human_rel0
    ),

    g1_relative_frame0_deg=(
        g1_rel0
    ),

    target_relative_frame0_deg=(
        target_rel0
    ),

    human_valid=(
        human_valid
    ),

    g1_valid=(
        g1_valid
    ),

    human_projection_norm=(
        human_proj_norm
    ),

    g1_projection_norm=(
        g1_proj_norm
    ),

    human_to_g1_mapping=(
        R_HUMAN_TO_G1
    ),
)


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