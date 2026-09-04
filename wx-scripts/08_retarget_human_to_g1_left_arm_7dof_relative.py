import argparse
import os

import mujoco
import numpy as np
import torch

from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


# ============================================================
# Command-line arguments
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Unified 7DoF relative-pose retargeting for the G1 left arm. "
        "The first 4DoF are softly anchored to a verified 4DoF trajectory; "
        "the wrist 3DoF tracks Human hand orientation relative to a "
        "shoulder-elbow-wrist arm frame."
    )
)

parser.add_argument(
    "--hybrik-pt-path",
    default="res/pick_place/hybrikx_output.pt",
    help="Path to HybrIK-X output .pt",
)
parser.add_argument(
    "--retarget-path",
    default="res/pick_place/retarget_input.npz",
    help="Path to retarget_input.npz",
)
parser.add_argument(
    "--g1-4dof-path",
    default="res/pick_place/g1_left_arm_4dof.npz",
    help="Path to the verified G1 4DoF trajectory",
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
    default="res/pick_place/g1_left_arm_7dof_relative.npz",
    help="Output 7DoF trajectory NPZ",
)

# 4DoF spatial target weights
parser.add_argument("--elbow-pos-weight", type=float, default=20.0)
parser.add_argument("--wrist-pos-weight", type=float, default=25.0)
parser.add_argument("--upper-dir-weight", type=float, default=1.0)
parser.add_argument("--fore-dir-weight", type=float, default=1.0)

# Relative-frame / wrist weights
parser.add_argument(
    "--arm-frame-weight",
    type=float,
    default=2.0,
    help="SO(3) weight for the calibrated shoulder-elbow-wrist arm frame",
)
parser.add_argument(
    "--hand-rel-rot-weight",
    type=float,
    default=3.0,
    help="SO(3) weight for hand orientation relative to the arm frame",
)
parser.add_argument(
    "--arm-anchor-weight",
    type=float,
    default=25.0,
    help="Keeps shoulder3+elbow close to the verified 4DoF trajectory",
)
parser.add_argument(
    "--temporal-weight",
    type=float,
    default=0.02,
    help="Frame-to-frame joint regularization",
)
parser.add_argument(
    "--wrist-rotation-scale",
    type=float,
    default=1.0,
    help=(
        "Scale Human hand-relative rotation from frame 0. "
        "1.0=full, 0.5=half, 0.0=no residual wrist motion."
    ),
)
parser.add_argument("--max-nfev", type=int, default=200)
parser.add_argument("--print-freq", type=int, default=50)

args = parser.parse_args()

HYBRIK_PT_PATH = args.hybrik_pt_path
RETARGET_PATH = args.retarget_path
G1_4DOF_PATH = args.g1_4dof_path
XML_PATH = args.xml_path
OUT_PATH = args.out_path

ELBOW_POS_WEIGHT = args.elbow_pos_weight
WRIST_POS_WEIGHT = args.wrist_pos_weight
UPPER_DIR_WEIGHT = args.upper_dir_weight
FORE_DIR_WEIGHT = args.fore_dir_weight
ARM_FRAME_WEIGHT = args.arm_frame_weight
HAND_REL_ROT_WEIGHT = args.hand_rel_rot_weight
ARM_ANCHOR_WEIGHT = args.arm_anchor_weight
TEMPORAL_WEIGHT = args.temporal_weight
WRIST_ROTATION_SCALE = args.wrist_rotation_scale
MAX_NFEV = args.max_nfev
PRINT_FREQ = args.print_freq


# ============================================================
# HybrIK / SMPL-X joint IDs
# ============================================================

PELVIS = 0
SPINE1 = 3
SPINE2 = 6
SPINE3 = 9
NECK = 12
LEFT_COLLAR = 13
LEFT_SHOULDER = 16
LEFT_ELBOW = 18
LEFT_WRIST = 20

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


# ============================================================
# Human -> G1 coordinate mapping
# ============================================================

R_HUMAN_TO_G1 = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


# ============================================================
# Utilities
# ============================================================

def normalize(v, eps=1e-8):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < eps:
        return None
    return v / n


def project_to_so3(R):
    U, _, Vt = np.linalg.svd(R)
    R_proj = U @ Vt
    if np.linalg.det(R_proj) < 0:
        U[:, -1] *= -1.0
        R_proj = U @ Vt
    return R_proj


def rotation_error_rotvec(R_target, R_current):
    R_err = project_to_so3(R_target.T @ R_current)
    return Rotation.from_matrix(R_err).as_rotvec()


def rotation_angle_deg(R):
    R = project_to_so3(R)
    return np.degrees(
        np.linalg.norm(Rotation.from_matrix(R).as_rotvec())
    )


def scale_rotation(R, scale):
    R = project_to_so3(R)
    rotvec = Rotation.from_matrix(R).as_rotvec()
    return Rotation.from_rotvec(rotvec * scale).as_matrix()


def angle_deg(a, b):
    a = normalize(a)
    b = normalize(b)
    if a is None or b is None:
        return np.nan
    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def arm_frame_from_points(shoulder, elbow, wrist, eps=1e-8):
    """
    Build the same geometric arm frame for Human and G1.

    z: shoulder -> wrist
    x: shoulder -> elbow projected onto plane perpendicular to z
    y: z x x
    """
    sw = wrist - shoulder
    se = elbow - shoulder

    z_axis = normalize(sw, eps=eps)
    if z_axis is None:
        return None

    x_raw = se - np.dot(se, z_axis) * z_axis
    x_norm = np.linalg.norm(x_raw)

    if x_norm < eps:
        # Numerical fallback only for near-collinear configurations.
        candidates = np.eye(3)
        helper = candidates[np.argmin(np.abs(candidates @ z_axis))]
        x_raw = helper - np.dot(helper, z_axis) * z_axis
        x_norm = np.linalg.norm(x_raw)
        if x_norm < eps:
            return None

    x_axis = x_raw / x_norm
    y_axis = normalize(np.cross(z_axis, x_axis), eps=eps)
    if y_axis is None:
        return None

    x_axis = normalize(np.cross(y_axis, z_axis), eps=eps)
    if x_axis is None:
        return None

    return project_to_so3(
        np.stack([x_axis, y_axis, z_axis], axis=-1)
    )


def temporal_rotation_steps_deg(R_seq):
    values = []
    for i in range(1, len(R_seq)):
        values.append(
            rotation_angle_deg(R_seq[i - 1].T @ R_seq[i])
        )
    return np.asarray(values, dtype=np.float64)


def relative_frame0_rotation_deg(R_seq):
    R0 = R_seq[0]
    return np.asarray(
        [rotation_angle_deg(R0.T @ R) for R in R_seq],
        dtype=np.float64,
    )


# ============================================================
# Load HybrIK
# ============================================================

hybrik_data = torch.load(HYBRIK_PT_PATH, map_location="cpu")

xyz = np.asarray(
    hybrik_data["pred_xyz_hybrik"],
    dtype=np.float64,
)

if xyz.ndim != 3:
    raise ValueError(
        f"Expected pred_xyz_hybrik [N,71,3], got {xyz.shape}"
    )

num_frames = xyz.shape[0]

if "pred_theta_mat" not in hybrik_data:
    raise KeyError("HybrIK output does not contain pred_theta_mat.")

theta = np.asarray(
    hybrik_data["pred_theta_mat"],
    dtype=np.float64,
)

if theta.ndim == 2:
    if theta.shape[1] != 55 * 3 * 3:
        raise ValueError(
            "Expected flattened pred_theta_mat [N,495], "
            f"got {theta.shape}"
        )
    theta = theta.reshape(theta.shape[0], 55, 3, 3)
elif theta.ndim == 4:
    if theta.shape[1:] != (55, 3, 3):
        raise ValueError(
            "Expected pred_theta_mat [N,55,3,3], "
            f"got {theta.shape}"
        )
else:
    raise ValueError(f"Unexpected pred_theta_mat shape: {theta.shape}")

if len(theta) != num_frames:
    raise ValueError(
        "pred_xyz_hybrik and pred_theta_mat frame mismatch."
    )

for frame_idx in range(num_frames):
    for jid in LEFT_WRIST_CHAIN:
        theta[frame_idx, jid] = project_to_so3(
            theta[frame_idx, jid]
        )

print("==========================================")
print("Input")
print("==========================================")
print("HybrIK       :", HYBRIK_PT_PATH)
print("Retarget     :", RETARGET_PATH)
print("Verified 4DoF:", G1_4DOF_PATH)
print("MJCF         :", XML_PATH)
print("Output       :", OUT_PATH)
print("Frames       :", num_frames)
print("theta shape  :", theta.shape)
print("wrist rotation scale:", WRIST_ROTATION_SCALE)


# ============================================================
# Human torso-local geometry
# ============================================================

retarget = np.load(RETARGET_PATH, allow_pickle=True)
R_torso = np.asarray(
    retarget["torso_rotation"],
    dtype=np.float64,
)

if len(R_torso) != num_frames:
    raise ValueError("HybrIK and retarget_input frame mismatch.")

torso_center_world = (
    xyz[:, SPINE3] + xyz[:, NECK]
) / 2.0


def world_to_torso_local(points_world):
    relative = points_world - torso_center_world
    return np.einsum(
        "nij,nj->ni",
        np.transpose(R_torso, (0, 2, 1)),
        relative,
    )


human_shoulder = world_to_torso_local(xyz[:, LEFT_SHOULDER])
human_elbow = world_to_torso_local(xyz[:, LEFT_ELBOW])
human_wrist = world_to_torso_local(xyz[:, LEFT_WRIST])

human_upper_vec = human_elbow - human_shoulder
human_fore_vec = human_wrist - human_elbow

human_upper_len = np.linalg.norm(human_upper_vec, axis=1)
human_fore_len = np.linalg.norm(human_fore_vec, axis=1)

human_upper_dir = (
    human_upper_vec / (human_upper_len[:, None] + 1e-8)
)
human_fore_dir = (
    human_fore_vec / (human_fore_len[:, None] + 1e-8)
)


# ============================================================
# Human geometric arm frame
# ============================================================

R_human_arm = np.zeros((num_frames, 3, 3), dtype=np.float64)

for frame_idx in range(num_frames):
    R = arm_frame_from_points(
        human_shoulder[frame_idx],
        human_elbow[frame_idx],
        human_wrist[frame_idx],
    )

    if R is None:
        if frame_idx == 0:
            raise RuntimeError(
                "Human frame 0 cannot construct arm frame."
            )
        print(
            f"Warning: Human arm-frame degenerate at frame "
            f"{frame_idx}; using previous frame."
        )
        R = R_human_arm[frame_idx - 1].copy()

    R_human_arm[frame_idx] = R


# ============================================================
# Map Human geometry into G1 axes
# ============================================================

mapped_upper_dir = (R_HUMAN_TO_G1 @ human_upper_dir.T).T
mapped_fore_dir = (R_HUMAN_TO_G1 @ human_fore_dir.T).T

R_human_arm_mapped = np.einsum(
    "ij,njk->nik",
    R_HUMAN_TO_G1,
    R_human_arm,
)


# ============================================================
# Human hand/wrist orientation
# ============================================================

R_human_wrist_global = np.repeat(
    np.eye(3, dtype=np.float64)[None, :, :],
    num_frames,
    axis=0,
)

for jid in LEFT_WRIST_CHAIN:
    R_human_wrist_global = np.einsum(
        "nij,njk->nik",
        R_human_wrist_global,
        theta[:, jid],
    )

# Express wrist orientation in torso-local coordinates.
R_human_hand_torso = np.einsum(
    "nij,njk->nik",
    np.transpose(R_torso, (0, 2, 1)),
    R_human_wrist_global,
)

for i in range(num_frames):
    R_human_hand_torso[i] = project_to_so3(
        R_human_hand_torso[i]
    )

# Remove the geometric arm-frame motion.
R_human_hand_rel = np.einsum(
    "nij,njk->nik",
    np.transpose(R_human_arm, (0, 2, 1)),
    R_human_hand_torso,
)

for i in range(num_frames):
    R_human_hand_rel[i] = project_to_so3(
        R_human_hand_rel[i]
    )

# Residual hand/wrist motion relative to frame 0.
R_human_hand_rel_delta = np.zeros_like(R_human_hand_rel)

for i in range(num_frames):
    R_delta = R_human_hand_rel[0].T @ R_human_hand_rel[i]
    R_human_hand_rel_delta[i] = scale_rotation(
        R_delta,
        WRIST_ROTATION_SCALE,
    )


# ============================================================
# Load verified 4DoF G1 trajectory
# ============================================================

traj4 = np.load(G1_4DOF_PATH, allow_pickle=True)
q4_ref = np.asarray(traj4["q"], dtype=np.float64)
joint_names4 = [str(x) for x in traj4["joint_names"]]

expected4 = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
]

if q4_ref.ndim != 2 or q4_ref.shape[1] != 4:
    raise ValueError(
        f"Expected verified trajectory [N,4], got {q4_ref.shape}"
    )

if len(q4_ref) != num_frames:
    raise ValueError(
        f"Frame mismatch: Human={num_frames}, G1 4DoF={len(q4_ref)}"
    )

if joint_names4 != expected4:
    raise ValueError(
        "Unexpected verified 4DoF joint order:\n"
        f"{joint_names4}"
    )


# ============================================================
# MuJoCo model / joints
# ============================================================

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]

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
        raise RuntimeError(f"Joint not found: {name}")

    qpos_adrs.append(model.jnt_qposadr[jid])
    lower.append(model.jnt_range[jid, 0])
    upper.append(model.jnt_range[jid, 1])

lower = np.asarray(lower, dtype=np.float64)
upper = np.asarray(upper, dtype=np.float64)


# ============================================================
# MuJoCo bodies
# ============================================================

SHOULDER_BODY = "left_shoulder_yaw_link"
ELBOW_BODY = "left_elbow_link"
WRIST_BODY = "left_wrist_roll_link"
HAND_BODY = "left_wrist_yaw_link"


def body_id(name):
    bid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        name,
    )
    if bid < 0:
        raise RuntimeError(f"Body not found: {name}")
    return bid


shoulder_bid = body_id(SHOULDER_BODY)
elbow_bid = body_id(ELBOW_BODY)
wrist_bid = body_id(WRIST_BODY)
hand_bid = body_id(HAND_BODY)


# ============================================================
# Robot FK
# ============================================================

def set_pose(q7):
    data.qpos[:] = model.qpos0.copy()
    for adr, value in zip(qpos_adrs, q7):
        data.qpos[adr] = value
    mujoco.mj_forward(model, data)


def get_robot_geometry(q7):
    set_pose(q7)

    shoulder = data.xpos[shoulder_bid].copy()
    elbow = data.xpos[elbow_bid].copy()
    wrist = data.xpos[wrist_bid].copy()

    R_hand = data.xmat[hand_bid].reshape(3, 3).copy()
    R_hand = project_to_so3(R_hand)

    upper_vec = elbow - shoulder
    fore_vec = wrist - elbow
    upper_dir = normalize(upper_vec)
    fore_dir = normalize(fore_vec)

    R_arm = arm_frame_from_points(
        shoulder,
        elbow,
        wrist,
    )

    if R_arm is None:
        raise RuntimeError(
            "G1 arm frame cannot be constructed for current q7."
        )

    R_hand_rel = project_to_so3(R_arm.T @ R_hand)

    return (
        shoulder,
        elbow,
        wrist,
        upper_dir,
        fore_dir,
        R_arm,
        R_hand,
        R_hand_rel,
    )


# ============================================================
# G1 morphology / spatial targets
# ============================================================

q_zero7 = np.zeros(7, dtype=np.float64)
(
    shoulder0,
    elbow0,
    wrist0,
    _,
    _,
    _,
    _,
    _,
) = get_robot_geometry(q_zero7)

G1_UPPER_LEN = np.linalg.norm(elbow0 - shoulder0)
G1_FORE_LEN = np.linalg.norm(wrist0 - elbow0)

print()
print("==========================================")
print("Robot geometry")
print("==========================================")
print("G1 upper length  :", G1_UPPER_LEN)
print("G1 forearm length:", G1_FORE_LEN)

# All targets are shoulder-relative.
target_elbow_rel = mapped_upper_dir * G1_UPPER_LEN

target_wrist_rel = (
    target_elbow_rel
    + mapped_fore_dir * G1_FORE_LEN
)


# ============================================================
# Frame-0 arm calibration
# ============================================================

q_ref0 = np.concatenate(
    [q4_ref[0], np.zeros(3, dtype=np.float64)]
)

(
    _,
    _,
    _,
    _,
    _,
    R_g1_arm_ref0,
    _,
    R_g1_hand_rel_ref0,
) = get_robot_geometry(q_ref0)

R_arm_calib = project_to_so3(
    R_g1_arm_ref0 @ R_human_arm_mapped[0].T
)

R_arm_target = np.einsum(
    "ij,njk->nik",
    R_arm_calib,
    R_human_arm_mapped,
)

for i in range(num_frames):
    R_arm_target[i] = project_to_so3(R_arm_target[i])

print()
print("==========================================")
print("Frame-0 arm calibration")
print("==========================================")
print("R_arm_calib:")
print(R_arm_calib)
print("det:", np.linalg.det(R_arm_calib))
print(
    "angle:",
    rotation_angle_deg(R_arm_calib),
    "deg",
)


# ============================================================
# G1 hand orientation target relative to G1 arm frame
# ============================================================

R_hand_rel_target = np.zeros(
    (num_frames, 3, 3),
    dtype=np.float64,
)

for i in range(num_frames):
    R_hand_rel_target[i] = project_to_so3(
        R_g1_hand_rel_ref0
        @ R_human_hand_rel_delta[i]
    )


# ============================================================
# Diagnostic: residual Human hand/wrist signal
# ============================================================

human_hand_rel_step = temporal_rotation_steps_deg(
    R_human_hand_rel
)
human_hand_rel_from0 = relative_frame0_rotation_deg(
    R_human_hand_rel
)

target_hand_rel_step = temporal_rotation_steps_deg(
    R_hand_rel_target
)
target_hand_rel_from0 = relative_frame0_rotation_deg(
    R_hand_rel_target
)

print()
print("==========================================")
print("Human hand-relative motion")
print("==========================================")
print("mean step:", human_hand_rel_step.mean(), "deg")
print("max step :", human_hand_rel_step.max(), "deg")
print(">10 deg  :", int(np.sum(human_hand_rel_step > 10)))
print(">20 deg  :", int(np.sum(human_hand_rel_step > 20)))
print(">30 deg  :", int(np.sum(human_hand_rel_step > 30)))
print(
    "max motion relative frame0:",
    human_hand_rel_from0.max(),
    "deg",
)


# ============================================================
# Sequential unified 7DoF IK
# ============================================================

q_traj = np.zeros((num_frames, 7), dtype=np.float64)
success_flags = np.zeros(num_frames, dtype=bool)

upper_errors = np.zeros(num_frames, dtype=np.float64)
fore_errors = np.zeros(num_frames, dtype=np.float64)
elbow_pos_errors = np.zeros(num_frames, dtype=np.float64)
wrist_pos_errors = np.zeros(num_frames, dtype=np.float64)
arm_frame_errors = np.zeros(num_frames, dtype=np.float64)
hand_rel_errors = np.zeros(num_frames, dtype=np.float64)
arm_anchor_error = np.zeros((num_frames, 4), dtype=np.float64)

q_prev = np.clip(q_ref0.copy(), lower, upper)

for frame_idx in range(num_frames):
    target_upper = mapped_upper_dir[frame_idx]
    target_fore = mapped_fore_dir[frame_idx]
    target_elbow = target_elbow_rel[frame_idx]
    target_wrist = target_wrist_rel[frame_idx]
    target_arm_R = R_arm_target[frame_idx]
    target_hand_rel_R = R_hand_rel_target[frame_idx]
    q4_anchor = q4_ref[frame_idx]

    def residual(q7):
        (
            shoulder,
            elbow,
            wrist,
            robot_upper,
            robot_fore,
            robot_arm_R,
            _,
            robot_hand_rel_R,
        ) = get_robot_geometry(q7)

        robot_elbow_rel = elbow - shoulder
        robot_wrist_rel = wrist - shoulder

        r_elbow_pos = (
            np.sqrt(ELBOW_POS_WEIGHT)
            * (robot_elbow_rel - target_elbow)
        )

        r_wrist_pos = (
            np.sqrt(WRIST_POS_WEIGHT)
            * (robot_wrist_rel - target_wrist)
        )

        r_upper_dir = (
            np.sqrt(UPPER_DIR_WEIGHT)
            * (robot_upper - target_upper)
        )

        r_fore_dir = (
            np.sqrt(FORE_DIR_WEIGHT)
            * (robot_fore - target_fore)
        )

        r_arm_frame = (
            np.sqrt(ARM_FRAME_WEIGHT)
            * rotation_error_rotvec(
                target_arm_R,
                robot_arm_R,
            )
        )

        r_hand_rel = (
            np.sqrt(HAND_REL_ROT_WEIGHT)
            * rotation_error_rotvec(
                target_hand_rel_R,
                robot_hand_rel_R,
            )
        )

        r_arm_anchor = (
            np.sqrt(ARM_ANCHOR_WEIGHT)
            * (q7[:4] - q4_anchor)
        )

        residuals = [
            r_elbow_pos,
            r_wrist_pos,
            r_upper_dir,
            r_fore_dir,
            r_arm_frame,
            r_hand_rel,
            r_arm_anchor,
        ]

        if frame_idx > 0:
            residuals.append(
                np.sqrt(TEMPORAL_WEIGHT)
                * (q7 - q_prev)
            )

        return np.concatenate(residuals)

    q_init = q_prev.copy()
    q_init[:4] = 0.5 * q_prev[:4] + 0.5 * q4_anchor
    q_init = np.clip(q_init, lower + 1e-9, upper - 1e-9)

    result = least_squares(
        residual,
        q_init,
        bounds=(lower, upper),
        max_nfev=MAX_NFEV,
        xtol=1e-9,
        ftol=1e-9,
        gtol=1e-9,
    )

    q_opt = result.x
    q_traj[frame_idx] = q_opt
    success_flags[frame_idx] = result.success

    (
        shoulder,
        elbow,
        wrist,
        robot_upper,
        robot_fore,
        robot_arm_R,
        _,
        robot_hand_rel_R,
    ) = get_robot_geometry(q_opt)

    robot_elbow_rel = elbow - shoulder
    robot_wrist_rel = wrist - shoulder

    upper_errors[frame_idx] = angle_deg(
        robot_upper,
        target_upper,
    )
    fore_errors[frame_idx] = angle_deg(
        robot_fore,
        target_fore,
    )
    elbow_pos_errors[frame_idx] = np.linalg.norm(
        robot_elbow_rel - target_elbow
    )
    wrist_pos_errors[frame_idx] = np.linalg.norm(
        robot_wrist_rel - target_wrist
    )
    arm_frame_errors[frame_idx] = rotation_angle_deg(
        target_arm_R.T @ robot_arm_R
    )
    hand_rel_errors[frame_idx] = rotation_angle_deg(
        target_hand_rel_R.T @ robot_hand_rel_R
    )
    arm_anchor_error[frame_idx] = q_opt[:4] - q4_anchor

    q_prev = q_opt.copy()

    if (
        frame_idx % PRINT_FREQ == 0
        or frame_idx == num_frames - 1
    ):
        print(
            f"[{frame_idx:4d}/{num_frames - 1}] "
            f"upper={upper_errors[frame_idx]:6.2f} deg, "
            f"fore={fore_errors[frame_idx]:6.2f} deg, "
            f"armR={arm_frame_errors[frame_idx]:6.2f} deg, "
            f"handRel={hand_rel_errors[frame_idx]:6.2f} deg, "
            f"anchor={np.degrees(np.linalg.norm(arm_anchor_error[frame_idx])):6.2f} deg"
        )


# ============================================================
# Temporal / limit statistics
# ============================================================

dq = np.diff(q_traj, axis=0)

mean_step_deg = np.degrees(
    np.mean(np.abs(dq), axis=0)
)
max_step_deg = np.degrees(
    np.max(np.abs(dq), axis=0)
)

LIMIT_MARGIN = np.deg2rad(1.0)
near_limit = (
    (q_traj <= lower[None, :] + LIMIT_MARGIN)
    |
    (q_traj >= upper[None, :] - LIMIT_MARGIN)
)


# ============================================================
# Save
# ============================================================

output_dir = os.path.dirname(OUT_PATH)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

np.savez(
    OUT_PATH,
    q=q_traj,
    joint_names=np.asarray(JOINT_NAMES),
    q4_reference=q4_ref,
    target_elbow_rel=target_elbow_rel,
    target_wrist_rel=target_wrist_rel,
    mapped_upper_dir=mapped_upper_dir,
    mapped_fore_dir=mapped_fore_dir,
    human_arm_frame=R_human_arm,
    human_arm_frame_mapped=R_human_arm_mapped,
    arm_frame_calibration=R_arm_calib,
    target_arm_frame=R_arm_target,
    human_hand_torso=R_human_hand_torso,
    human_hand_relative_arm=R_human_hand_rel,
    human_hand_relative_delta=R_human_hand_rel_delta,
    target_hand_relative_arm=R_hand_rel_target,
    upper_error_deg=upper_errors,
    fore_error_deg=fore_errors,
    elbow_position_error=elbow_pos_errors,
    wrist_position_error=wrist_pos_errors,
    arm_frame_error_deg=arm_frame_errors,
    hand_relative_error_deg=hand_rel_errors,
    arm_anchor_error_rad=arm_anchor_error,
    success=success_flags,
    lower_limit=lower,
    upper_limit=upper,
    human_hand_relative_step_deg=human_hand_rel_step,
    human_hand_relative_frame0_deg=human_hand_rel_from0,
    target_hand_relative_step_deg=target_hand_rel_step,
    target_hand_relative_frame0_deg=target_hand_rel_from0,
    wrist_rotation_scale=np.asarray(WRIST_ROTATION_SCALE),
)


# ============================================================
# Summary
# ============================================================

print()
print("==========================================")
print("Done")
print("==========================================")
print("Saved:", OUT_PATH)
print("q shape:", q_traj.shape)
print(
    "IK success:",
    int(np.sum(success_flags)),
    "/",
    num_frames,
)

print()
print("========== Spatial errors ==========")
print(
    "Upper mean/max:",
    upper_errors.mean(),
    "/",
    upper_errors.max(),
    "deg",
)
print(
    "Fore mean/max :",
    fore_errors.mean(),
    "/",
    fore_errors.max(),
    "deg",
)
print(
    "Elbow pos mean/max:",
    elbow_pos_errors.mean(),
    "/",
    elbow_pos_errors.max(),
)
print(
    "Wrist pos mean/max:",
    wrist_pos_errors.mean(),
    "/",
    wrist_pos_errors.max(),
)

print()
print("========== Relative rotation errors ==========")
print(
    "Arm frame mean/max:",
    arm_frame_errors.mean(),
    "/",
    arm_frame_errors.max(),
    "deg",
)
print(
    "Hand relative mean/max:",
    hand_rel_errors.mean(),
    "/",
    hand_rel_errors.max(),
    "deg",
)

print()
print("========== 4DoF anchor deviation ==========")
for i, name in enumerate(JOINT_NAMES[:4]):
    abs_deg = np.degrees(np.abs(arm_anchor_error[:, i]))
    print(
        f"{name:30s}: "
        f"mean={abs_deg.mean():.3f} deg, "
        f"max={abs_deg.max():.3f} deg"
    )

print()
print("========== Joint ranges ==========")
for i, name in enumerate(JOINT_NAMES):
    print(
        f"{name:30s}: "
        f"{np.degrees(q_traj[:, i].min()):+8.2f} "
        f"~ "
        f"{np.degrees(q_traj[:, i].max()):+8.2f} deg"
    )

print()
print("========== Mean frame-to-frame step ==========")
for name, value in zip(JOINT_NAMES, mean_step_deg):
    print(
        f"{name:30s}: {value:.3f} deg/frame"
    )

print()
print("========== Max frame-to-frame step ==========")
for name, value in zip(JOINT_NAMES, max_step_deg):
    print(
        f"{name:30s}: {value:.3f} deg/frame"
    )

print()
print("========== Near joint limits ==========")
for i, name in enumerate(JOINT_NAMES):
    count = int(near_limit[:, i].sum())
    print(
        f"{name:30s}: {count}/{num_frames}"
    )

print()
print("========== Human residual wrist signal ==========")
print("mean step:", human_hand_rel_step.mean(), "deg")
print("max step :", human_hand_rel_step.max(), "deg")
print(">10 deg  :", int(np.sum(human_hand_rel_step > 10)))
print(">20 deg  :", int(np.sum(human_hand_rel_step > 20)))
print(">30 deg  :", int(np.sum(human_hand_rel_step > 30)))
print(
    "max relative frame0:",
    human_hand_rel_from0.max(),
    "deg",
)
