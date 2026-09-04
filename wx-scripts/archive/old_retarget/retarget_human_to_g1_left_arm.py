import mujoco
import numpy as np
from scipy.optimize import least_squares


XML_PATH = "/data/wx/code-IK/unitree_ros/robots/g1_description/g1_29dof.xml"
RETARGET_PATH = "res/pick_place/retarget_input.npz"

OUT_PATH = "res/pick_place/g1_left_arm_trajectory.npz"


# ============================================================
# Configuration
# ============================================================

TEMPORAL_WEIGHT = 0.03
MAX_NFEV = 200


# ============================================================
# Utilities
# ============================================================

def normalize(v):
    return v / (np.linalg.norm(v) + 1e-8)


def angular_error_deg(a, b):

    value = np.clip(
        np.dot(normalize(a), normalize(b)),
        -1.0,
        1.0,
    )

    return np.degrees(np.arccos(value))


# Human torso:
#
# +x = right
# +y = up
# +z = forward
#
# G1:
#
# +x = forward
# +y = left
# +z = up

R_HUMAN_TO_G1 = np.array([
    [0.0,  0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0,  1.0, 0.0],
])


# ============================================================
# Human data
# ============================================================

human = np.load(RETARGET_PATH)

human_upper_all = human["left_upper_arm_dir"]
human_fore_all = human["left_forearm_dir"]

num_frames = len(human_upper_all)

print("Frames:", num_frames)


target_upper_all = np.asarray([
    normalize(R_HUMAN_TO_G1 @ v)
    for v in human_upper_all
])

target_fore_all = np.asarray([
    normalize(R_HUMAN_TO_G1 @ v)
    for v in human_fore_all
])


# ============================================================
# G1 model
# ============================================================

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)


JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
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

    qpos_adrs.append(
        model.jnt_qposadr[jid]
    )

    lower.append(
        model.jnt_range[jid, 0]
    )

    upper.append(
        model.jnt_range[jid, 1]
    )


lower = np.asarray(lower)
upper = np.asarray(upper)


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
# FK
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
# Storage
# ============================================================

trajectory = np.zeros(
    (num_frames, 4),
    dtype=np.float64,
)

upper_errors = np.zeros(num_frames)
fore_errors = np.zeros(num_frames)

costs = np.zeros(num_frames)
success_flags = np.zeros(
    num_frames,
    dtype=bool,
)


# ============================================================
# Sequential IK
# ============================================================

q_prev = np.zeros(4)


for frame_idx in range(num_frames):

    target_upper = target_upper_all[frame_idx]
    target_fore = target_fore_all[frame_idx]


    def residual(q):

        g1_upper, g1_fore = get_arm_dirs(q)

        r_upper = (
            g1_upper
            -
            target_upper
        )

        r_fore = (
            g1_fore
            -
            target_fore
        )

        # Encourage current solution to remain
        # close to previous frame.
        r_temporal = (
            np.sqrt(TEMPORAL_WEIGHT)
            *
            (q - q_prev)
        )

        return np.concatenate([
            r_upper,
            r_fore,
            r_temporal,
        ])


    result = least_squares(
        residual,
        q_prev,
        bounds=(lower, upper),
        max_nfev=MAX_NFEV,
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8,
    )


    q_opt = result.x

    trajectory[frame_idx] = q_opt

    success_flags[frame_idx] = result.success
    costs[frame_idx] = result.cost


    g1_upper, g1_fore = get_arm_dirs(q_opt)

    upper_errors[frame_idx] = angular_error_deg(
        target_upper,
        g1_upper,
    )

    fore_errors[frame_idx] = angular_error_deg(
        target_fore,
        g1_fore,
    )


    q_prev = q_opt.copy()


    if (
        frame_idx % 50 == 0
        or frame_idx == num_frames - 1
    ):

        print(
            f"[{frame_idx:4d}/{num_frames}] "
            f"upper={upper_errors[frame_idx]:6.2f} deg, "
            f"fore={fore_errors[frame_idx]:6.2f} deg"
        )


# ============================================================
# Joint velocity / jump statistics
# ============================================================

dq = np.diff(
    trajectory,
    axis=0,
)

max_step_rad = np.max(
    np.abs(dq),
    axis=0,
)

max_step_deg = np.degrees(
    max_step_rad
)


# ============================================================
# Limit statistics
# ============================================================

margin = np.deg2rad(1.0)

near_lower = (
    trajectory
    <=
    lower[None, :] + margin
)

near_upper = (
    trajectory
    >=
    upper[None, :] - margin
)

near_limit = (
    near_lower
    |
    near_upper
)


# ============================================================
# Save
# ============================================================

np.savez(
    OUT_PATH,

    q=trajectory,

    joint_names=np.asarray(
        JOINT_NAMES
    ),

    target_upper=target_upper_all,
    target_fore=target_fore_all,

    upper_error_deg=upper_errors,
    fore_error_deg=fore_errors,

    success=success_flags,
    cost=costs,

    lower_limit=lower,
    upper_limit=upper,
)


# ============================================================
# Summary
# ============================================================

print()
print("========== Done ==========")

print("Saved:", OUT_PATH)

print()
print("IK success:")
print(
    np.sum(success_flags),
    "/",
    num_frames,
)


print()
print("Upper arm angular error:")

print(
    "  mean:",
    np.mean(upper_errors)
)

print(
    "  max :",
    np.max(upper_errors)
)


print()
print("Forearm angular error:")

print(
    "  mean:",
    np.mean(fore_errors)
)

print(
    "  max :",
    np.max(fore_errors)
)


print()
print("========== Joint ranges ==========")

for i, name in enumerate(JOINT_NAMES):

    print(name)

    print(
        "  min:",
        np.degrees(
            trajectory[:, i].min()
        ),
        "deg"
    )

    print(
        "  max:",
        np.degrees(
            trajectory[:, i].max()
        ),
        "deg"
    )

    print(
        "  model limit:",
        np.degrees(lower[i]),
        "~",
        np.degrees(upper[i]),
        "deg"
    )


print()
print("========== Max frame-to-frame step ==========")

for name, value in zip(
    JOINT_NAMES,
    max_step_deg,
):

    print(
        f"{name:30s}: "
        f"{value:.3f} deg/frame"
    )


print()
print("========== Near joint limits ==========")

for i, name in enumerate(JOINT_NAMES):

    count = int(
        near_limit[:, i].sum()
    )

    print(
        f"{name:30s}: "
        f"{count}/{num_frames}"
    )