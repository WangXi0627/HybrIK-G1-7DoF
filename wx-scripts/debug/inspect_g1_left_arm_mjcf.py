import mujoco
import numpy as np


# 改成你本机这个 xml 的真实路径
XML_PATH = "/data/wx/code-IK/unitree_ros/robots/g1_description/g1_29dof.xml"


model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)


LEFT_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]


print("========== Joint info ==========")

for name in LEFT_JOINTS:
    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    qpos_adr = model.jnt_qposadr[jid]

    print()
    print(name)
    print("  joint id :", jid)
    print("  qpos adr :", qpos_adr)
    print("  axis     :", model.jnt_axis[jid])
    print("  range    :", model.jnt_range[jid])


# =========================
# Set arm joints to zero
# =========================

for name in LEFT_JOINTS:
    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    qpos_adr = model.jnt_qposadr[jid]
    data.qpos[qpos_adr] = 0.0


# Forward kinematics
mujoco.mj_forward(model, data)


# =========================
# Body positions
# =========================

BODY_NAMES = [
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
]


print()
print("========== Body positions ==========")

positions = {}

for name in BODY_NAMES:
    bid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        name,
    )

    pos = data.xpos[bid].copy()
    positions[name] = pos

    print(
        f"{name:28s}: "
        f"{pos}"
    )


# =========================
# Arm direction sanity check
# =========================

shoulder = positions["left_shoulder_yaw_link"]
elbow = positions["left_elbow_link"]
wrist = positions["left_wrist_roll_link"]


def normalize(v):
    return v / (np.linalg.norm(v) + 1e-8)


upper_arm_dir = normalize(
    elbow - shoulder
)

forearm_dir = normalize(
    wrist - elbow
)


print()
print("========== Directions ==========")

print(
    "upper arm dir:",
    upper_arm_dir
)

print(
    "forearm dir  :",
    forearm_dir
)

print(
    "upper norm   :",
    np.linalg.norm(upper_arm_dir)
)

print(
    "forearm norm :",
    np.linalg.norm(forearm_dir)
)