#!/usr/bin/env python3
import argparse
import os

import mujoco
import numpy as np
import torch


parser = argparse.ArgumentParser(
    description=(
        "Inspect RIGHT shoulder-elbow-wrist arm-frame stability and "
        "Human-to-G1 frame-0 calibration."
    )
)
parser.add_argument("--hybrik-pt-path", default="res/pick_place/hybrikx_output.pt")
parser.add_argument("--retarget-path", default="res/pick_place/retarget_input.npz")
parser.add_argument("--g1-traj-path", default="res/pick_place/g1_right_arm_4dof.npz")
parser.add_argument(
    "--xml-path",
    default="/data/wx/code-IK/unitree_ros/robots/g1_description/g1_29dof.xml",
)
parser.add_argument("--out-path", default="res/pick_place/right_arm_frame_calibration.npz")
parser.add_argument("--degenerate-threshold", type=float, default=1e-4)
args = parser.parse_args()

SPINE3 = 9
NECK = 12
RIGHT_SHOULDER = 17
RIGHT_ELBOW = 19
RIGHT_WRIST = 21

R_HUMAN_TO_G1 = np.array(
    [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)


def normalize(v, eps=1e-8):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < eps:
        return None
    return v / n


def rotation_angle_deg(R):
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(c))


def rotation_sequence_step_deg(R_seq):
    return np.asarray(
        [rotation_angle_deg(R_seq[i - 1].T @ R_seq[i]) for i in range(1, len(R_seq))],
        dtype=np.float64,
    )


def rotation_relative_frame0_deg(R_seq):
    R0 = R_seq[0]
    return np.asarray([rotation_angle_deg(R0.T @ R) for R in R_seq], dtype=np.float64)


def build_arm_frame(shoulder, elbow, wrist):
    z = normalize(wrist - shoulder)
    if z is None:
        return None, np.nan
    se = elbow - shoulder
    x_raw = se - np.dot(se, z) * z
    proj_norm = np.linalg.norm(x_raw)
    if proj_norm < args.degenerate_threshold:
        return None, proj_norm
    x = x_raw / proj_norm
    y = normalize(np.cross(z, x))
    if y is None:
        return None, proj_norm
    x = normalize(np.cross(y, z))
    if x is None:
        return None, proj_norm
    return np.stack([x, y, z], axis=-1), proj_norm


def build_sequence(S, E, W, name):
    frames, valid, norms = [], np.ones(len(S), dtype=bool), np.zeros(len(S))
    prev = None
    for i in range(len(S)):
        R, n = build_arm_frame(S[i], E[i], W[i])
        norms[i] = n
        if R is None:
            valid[i] = False
            if prev is None:
                raise RuntimeError(f"{name}: frame {i} is degenerate and no previous frame exists")
            R = prev.copy()
        frames.append(R)
        prev = R
    return np.asarray(frames), valid, norms


hybrik = torch.load(args.hybrik_pt_path, map_location="cpu")
xyz = np.asarray(hybrik["pred_xyz_hybrik"], dtype=np.float64)
if xyz.ndim != 3:
    raise ValueError(f"Expected [N,71,3], got {xyz.shape}")
N = len(xyz)

retarget = np.load(args.retarget_path, allow_pickle=True)
R_torso = np.asarray(retarget["torso_rotation"], dtype=np.float64)
if len(R_torso) != N:
    raise ValueError("HybrIK / retarget frame count mismatch")

torso_center = (xyz[:, SPINE3] + xyz[:, NECK]) / 2.0

def to_torso_local(points):
    rel = points - torso_center
    return np.einsum("nij,nj->ni", np.transpose(R_torso, (0, 2, 1)), rel)

S_h = to_torso_local(xyz[:, RIGHT_SHOULDER])
E_h = to_torso_local(xyz[:, RIGHT_ELBOW])
W_h = to_torso_local(xyz[:, RIGHT_WRIST])
R_human, human_valid, human_proj = build_sequence(S_h, E_h, W_h, "Human right arm")
R_human_mapped = np.einsum("ij,njk->nik", R_HUMAN_TO_G1, R_human)

traj = np.load(args.g1_traj_path, allow_pickle=True)
q = np.asarray(traj["q"], dtype=np.float64)
joint_names = [str(x) for x in traj["joint_names"]]
expected = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]
if len(q) != N:
    raise ValueError(f"Frame mismatch Human={N}, G1={len(q)}")
if joint_names != expected:
    raise ValueError(f"Unexpected G1 joint order: {joint_names}")

model = mujoco.MjModel.from_xml_path(args.xml_path)
data = mujoco.MjData(model)
qpos_adrs = []
for name in joint_names:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise RuntimeError(f"Joint not found: {name}")
    qpos_adrs.append(model.jnt_qposadr[jid])

body_names = ["right_shoulder_yaw_link", "right_elbow_link", "right_wrist_roll_link"]
bids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in body_names]
if min(bids) < 0:
    raise RuntimeError(f"Failed to resolve G1 right-arm bodies: {body_names}")

S_g = np.zeros((N, 3)); E_g = np.zeros((N, 3)); W_g = np.zeros((N, 3))
for i in range(N):
    data.qpos[:] = model.qpos0.copy()
    for adr, value in zip(qpos_adrs, q[i]):
        data.qpos[adr] = value
    mujoco.mj_forward(model, data)
    S_g[i] = data.xpos[bids[0]]
    E_g[i] = data.xpos[bids[1]]
    W_g[i] = data.xpos[bids[2]]

R_g1, g1_valid, g1_proj = build_sequence(S_g, E_g, W_g, "G1 right arm")
R_calib = R_g1[0] @ R_human_mapped[0].T
R_target = np.einsum("ij,njk->nik", R_calib, R_human_mapped)
err = np.asarray([rotation_angle_deg(R_target[i].T @ R_g1[i]) for i in range(N)])

human_step = rotation_sequence_step_deg(R_human)
target_step = rotation_sequence_step_deg(R_target)
g1_step = rotation_sequence_step_deg(R_g1)
human_rel0 = rotation_relative_frame0_deg(R_human)
target_rel0 = rotation_relative_frame0_deg(R_target)
g1_rel0 = rotation_relative_frame0_deg(R_g1)

print("==========================================")
print("RIGHT arm frame calibration")
print("==========================================")
print("frames:", N)
print("calibration angle:", rotation_angle_deg(R_calib), "deg")
for name, values in [
    ("Human arm frame temporal", human_step),
    ("Calibrated Human target temporal", target_step),
    ("G1 arm frame temporal", g1_step),
]:
    print(f"\n========== {name} ==========")
    print("mean step:", values.mean(), "deg")
    print("max step :", values.max(), "deg")
    print(">10 deg  :", int(np.sum(values > 10)))
    print(">20 deg  :", int(np.sum(values > 20)))
    print(">30 deg  :", int(np.sum(values > 30)))

print("\n========== Arm-frame motion relative to frame 0 ==========")
print("Human max:", human_rel0.max(), "deg")
print("Target max:", target_rel0.max(), "deg")
print("G1 max:", g1_rel0.max(), "deg")
print("\n========== Calibrated Human target vs G1 ==========")
print("mean error:", err.mean(), "deg")
print("max error :", err.max(), "deg")
print(">10 deg   :", int(np.sum(err > 10)))
print("max error frame:", int(np.argmax(err)))
print("\n========== Frame construction health ==========")
print("Human valid:", int(human_valid.sum()), "/", N)
print("Human projected elbow norm min/mean:", np.nanmin(human_proj), np.nanmean(human_proj))
print("G1 valid:", int(g1_valid.sum()), "/", N)
print("G1 projected elbow norm min/mean:", np.nanmin(g1_proj), np.nanmean(g1_proj))

out_dir = os.path.dirname(args.out_path)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)
np.savez(
    args.out_path,
    human_arm_frame=R_human,
    human_arm_frame_mapped=R_human_mapped,
    g1_arm_frame=R_g1,
    calibration_rotation=R_calib,
    calibrated_target_frame=R_target,
    target_vs_g1_error_deg=err,
    human_step_deg=human_step,
    g1_step_deg=g1_step,
    target_step_deg=target_step,
    human_relative_frame0_deg=human_rel0,
    g1_relative_frame0_deg=g1_rel0,
    target_relative_frame0_deg=target_rel0,
    human_valid=human_valid,
    g1_valid=g1_valid,
    human_projection_norm=human_proj,
    g1_projection_norm=g1_proj,
    human_to_g1_mapping=R_HUMAN_TO_G1,
)
print("\nSaved:", args.out_path)
