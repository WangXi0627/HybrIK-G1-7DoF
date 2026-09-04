import mujoco
import numpy as np


# ============================================================
# Config
# ============================================================

RETARGET_PATH = (
    "res/pick_place/retarget_input.npz"
)

TRAJ_PATH = (
    "res/pick_place/g1_left_arm_trajectory.npz"
)

XML_PATH = (
    "/data/wx/code-IK/unitree_ros/"
    "robots/g1_description/g1_29dof.xml"
)


# 你刚刚截图的帧
CHECK_FRAMES = [
    0,
    83,
    147,
    199,
    250,
    361,
]


# ============================================================
# Current mapping
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
# Utils
# ============================================================

def normalize(v):
    return v / (
        np.linalg.norm(v)
        + 1e-8
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
# Load human retarget signal
# ============================================================

retarget = np.load(
    RETARGET_PATH
)

human_upper = np.asarray(
    retarget[
        "left_upper_arm_dir"
    ],
    dtype=np.float64,
)

human_fore = np.asarray(
    retarget[
        "left_forearm_dir"
    ],
    dtype=np.float64,
)


# ============================================================
# Load G1 trajectory
# ============================================================

traj = np.load(
    TRAJ_PATH,
    allow_pickle=True,
)

q = np.asarray(
    traj["q"],
    dtype=np.float64,
)


print(
    "human upper:",
    human_upper.shape
)

print(
    "human fore :",
    human_fore.shape
)

print(
    "G1 q       :",
    q.shape
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


JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
]


qpos_adrs = []

for name in JOINT_NAMES:

    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )

    if jid < 0:
        raise RuntimeError(
            f"joint not found: {name}"
        )

    qpos_adrs.append(
        model.jnt_qposadr[jid]
    )


# ============================================================
# Bodies used by original IK
# ============================================================

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


if min(
    shoulder_bid,
    elbow_bid,
    wrist_bid,
) < 0:

    raise RuntimeError(
        "Failed to resolve G1 arm bodies."
    )


# ============================================================
# Set G1 pose
# ============================================================

def set_pose(q4):

    data.qpos[:] = 0.0

    # floating base quaternion
    if model.nq >= 7:
        data.qpos[3] = 1.0

    for adr, value in zip(
        qpos_adrs,
        q4,
    ):
        data.qpos[adr] = value

    mujoco.mj_forward(
        model,
        data,
    )


def get_robot_directions(q4):

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

    upper = normalize(
        elbow - shoulder
    )

    fore = normalize(
        wrist - elbow
    )

    return (
        upper,
        fore,
        shoulder,
        elbow,
        wrist,
    )


# ============================================================
# Print coordinate convention
# ============================================================

print()
print(
    "=========================================="
)
print(
    "Coordinate convention"
)
print(
    "=========================================="
)

print(
    "Human torso local:"
)

print(
    "  +x = presumed anatomical RIGHT"
)

print(
    "  +y = presumed UP"
)

print(
    "  +z = presumed FORWARD <-- needs verification"
)

print()

print(
    "G1:"
)

print(
    "  +x = forward"
)

print(
    "  +y = left"
)

print(
    "  +z = up"
)


# ============================================================
# Frames
# ============================================================

for frame in CHECK_FRAMES:

    if frame >= len(q):
        continue


    h_upper = normalize(
        human_upper[frame]
    )

    h_fore = normalize(
        human_fore[frame]
    )


    target_upper = normalize(
        R_HUMAN_TO_G1
        @ h_upper
    )

    target_fore = normalize(
        R_HUMAN_TO_G1
        @ h_fore
    )


    (
        robot_upper,
        robot_fore,
        shoulder,
        elbow,
        wrist,
    ) = get_robot_directions(
        q[frame]
    )


    print()
    print(
        "=========================================="
    )

    print(
        f"FRAME {frame}"
    )

    print(
        "=========================================="
    )


    print(
        "Human torso-local:"
    )

    print(
        "  upper:",
        np.round(
            h_upper,
            4,
        )
    )

    print(
        "  fore :",
        np.round(
            h_fore,
            4,
        )
    )


    print()
    print(
        "Mapped G1 target:"
    )

    print(
        "  upper:",
        np.round(
            target_upper,
            4,
        )
    )

    print(
        "  fore :",
        np.round(
            target_fore,
            4,
        )
    )


    print()
    print(
        "Actual G1 FK:"
    )

    print(
        "  upper:",
        np.round(
            robot_upper,
            4,
        )
    )

    print(
        "  fore :",
        np.round(
            robot_fore,
            4,
        )
    )


    print()
    print(
        "Tracking error:"
    )

    print(
        "  upper:",
        angle_deg(
            target_upper,
            robot_upper,
        ),
        "deg"
    )

    print(
        "  fore :",
        angle_deg(
            target_fore,
            robot_fore,
        ),
        "deg"
    )


    print()
    print(
        "Robot positions:"
    )

    print(
        "  shoulder:",
        np.round(
            shoulder,
            4,
        )
    )

    print(
        "  elbow   :",
        np.round(
            elbow,
            4,
        )
    )

    print(
        "  wrist   :",
        np.round(
            wrist,
            4,
        )
    )


    print()
    print(
        "G1 q deg:"
    )

    print(
        np.round(
            np.degrees(
                q[frame]
            ),
            2,
        )
    )