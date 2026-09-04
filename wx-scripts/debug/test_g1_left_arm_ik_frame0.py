import mujoco
import numpy as np
from scipy.optimize import least_squares


XML_PATH = "/data/wx/code-IK/unitree_ros/robots/g1_description/g1_29dof.xml"
RETARGET_PATH = "res/pick_place/retarget_input.npz"


# ============================================================
# Utilities
# ============================================================

def normalize(v):
    return v / (np.linalg.norm(v) + 1e-8)


# Human torso:
#   +x = right
#   +y = up
#   +z = forward
#
# G1 / MuJoCo:
#   +x = forward
#   +y = left
#   +z = up
#
# Therefore:
#
#   x_g1 =  z_human
#   y_g1 = -x_human
#   z_g1 =  y_human
#
R_HUMAN_TO_G1 = np.array([
    [0.0,  0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0,  1.0, 0.0],
])


# ============================================================
# Load human target
# ============================================================

human = np.load(RETARGET_PATH)

human_upper = human["left_upper_arm_dir"][0]
human_fore = human["left_forearm_dir"][0]

target_upper = normalize(
    R_HUMAN_TO_G1 @ human_upper
)

target_fore = normalize(
    R_HUMAN_TO_G1 @ human_fore
)


print("========== Human frame 0 ==========")

print("human upper :", human_upper)
print("human fore  :", human_fore)

print()
print("========== Target in G1 frame ==========")

print("target upper:", target_upper)
print("target fore :", target_fore)


# ============================================================
# Load G1
# ============================================================

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)


JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
]


joint_ids = []
qpos_adrs = []
lower = []
upper = []

for name in JOINT_NAMES:

    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    joint_ids.append(jid)

    qadr = model.jnt_qposadr[jid]
    qpos_adrs.append(qadr)

    lower.append(model.jnt_range[jid, 0])
    upper.append(model.jnt_range[jid, 1])


lower = np.asarray(lower)
upper = np.asarray(upper)


# Body IDs
shoulder_bid = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "left_shoulder_yaw_link",
)

elbow_bid = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "left_elbow_link",
)

wrist_bid = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "left_wrist_roll_link",
)


# ============================================================
# Forward kinematics
# ============================================================

def set_q(q):

    for adr, value in zip(qpos_adrs, q):
        data.qpos[adr] = value

    mujoco.mj_forward(model, data)


def get_arm_dirs(q):

    set_q(q)

    shoulder = data.xpos[shoulder_bid].copy()
    elbow = data.xpos[elbow_bid].copy()
    wrist = data.xpos[wrist_bid].copy()

    upper_dir = normalize(
        elbow - shoulder
    )

    fore_dir = normalize(
        wrist - elbow
    )

    return upper_dir, fore_dir


# ============================================================
# IK residual
# ============================================================

def residual(q):

    g1_upper, g1_fore = get_arm_dirs(q)

    r_upper = g1_upper - target_upper
    r_fore = g1_fore - target_fore

    return np.concatenate([
        r_upper,
        r_fore,
    ])


# ============================================================
# Solve
# ============================================================

q0 = np.zeros(4)

print()
print("========== G1 zero pose ==========")

zero_upper, zero_fore = get_arm_dirs(q0)

print("zero upper:", zero_upper)
print("zero fore :", zero_fore)


result = least_squares(
    residual,
    q0,
    bounds=(lower, upper),
    max_nfev=500,
    xtol=1e-10,
    ftol=1e-10,
    gtol=1e-10,
)


q_opt = result.x

g1_upper, g1_fore = get_arm_dirs(q_opt)


# ============================================================
# Results
# ============================================================

print()
print("========== IK result ==========")

for name, q in zip(JOINT_NAMES, q_opt):

    print(
        f"{name:30s}: "
        f"{q:+.6f} rad  "
        f"{np.degrees(q):+.2f} deg"
    )


print()
print("========== Direction comparison ==========")

print("target upper:", target_upper)
print("G1 upper    :", g1_upper)

print()

print("target fore :", target_fore)
print("G1 fore     :", g1_fore)


upper_error = np.degrees(
    np.arccos(
        np.clip(
            np.dot(target_upper, g1_upper),
            -1.0,
            1.0,
        )
    )
)

fore_error = np.degrees(
    np.arccos(
        np.clip(
            np.dot(target_fore, g1_fore),
            -1.0,
            1.0,
        )
    )
)


print()
print("========== Angular error ==========")

print(
    "upper arm error:",
    upper_error,
    "deg",
)

print(
    "forearm error  :",
    fore_error,
    "deg",
)

print()
print("optimizer success:", result.success)
print("optimizer message:", result.message)
print("cost:", result.cost)