import os
import cv2
import mujoco
import numpy as np


# ============================================================
# User config
# ============================================================

# G1 MJCF
XML_PATH = (
    "/data/wx/code-IK/unitree_ros/"
    "robots/g1_description/g1_29dof.xml"
)

# Retargeted trajectory npz
NPZ_PATH = (
    "res/pick_place/g1_left_arm_4dof.npz"
)

# Output video path
OUTPUT_MP4_PATH = "res/pick_place/g1_left_arm_4dof.mp4"

# Render size
WIDTH = 1280
HEIGHT = 720

# Video FPS
FPS = 30

# Frame range
START_FRAME = 0
END_FRAME = None   # None means use all frames

# Playback speed
# 1.0 = normal
# 0.5 = half speed
# 2.0 = double speed
SPEED = 1.0

# Whether to draw frame index text
DRAW_TEXT = True

# Wrist joints are not yet retargeted, keep them fixed at zero
FIX_WRIST_TO_ZERO = False


# ============================================================
# Camera config
# ============================================================

CAM_LOOKAT = np.array([0.0, 0.0, 1.0], dtype=np.float64)
CAM_DISTANCE = 2.5
CAM_AZIMUTH = 150.0
CAM_ELEVATION = -15.0


# ============================================================
# Load trajectory
# ============================================================

traj_data = np.load(
    NPZ_PATH,
    allow_pickle=True,
)

print("========== NPZ keys ==========")
for key in traj_data.files:
    value = traj_data[key]
    if hasattr(value, "shape"):
        print(f"{key}: {value.shape}")
    else:
        print(f"{key}: {type(value)}")

q_traj = traj_data["q"]
joint_names = [str(x) for x in traj_data["joint_names"]]

num_frames = q_traj.shape[0]

if END_FRAME is None:
    END_FRAME = num_frames

START_FRAME = max(0, START_FRAME)
END_FRAME = min(num_frames, END_FRAME)

if START_FRAME >= END_FRAME:
    raise ValueError(
        f"Invalid frame range: START_FRAME={START_FRAME}, END_FRAME={END_FRAME}"
    )

print()
print("========== Trajectory ==========")
print("NPZ path   :", NPZ_PATH)
print("q shape    :", q_traj.shape)
print("joint names:", joint_names)
print("frame range:", START_FRAME, "->", END_FRAME - 1)


# ============================================================
# Load MuJoCo model
# ============================================================

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)
# enlarge offscreen framebuffer
model.vis.global_.offwidth = WIDTH
model.vis.global_.offheight = HEIGHT
print()
print("========== MuJoCo model ==========")
print("XML path:", XML_PATH)
print("nq      :", model.nq)
print("nv      :", model.nv)


# ============================================================
# Resolve joint qpos addresses
# ============================================================

qpos_adrs = []

print()
print("========== Joint mapping ==========")
for name in joint_names:
    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )
    if jid < 0:
        raise RuntimeError(f"Joint not found in MJCF: {name}")

    qadr = model.jnt_qposadr[jid]
    qpos_adrs.append(qadr)

    print(f"{name:30s} -> joint_id={jid:2d}, qpos_adr={qadr:2d}")


# ============================================================
# Wrist joints
# ============================================================

WRIST_JOINT_NAMES = [
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]

wrist_qpos_adrs = []

for name in WRIST_JOINT_NAMES:
    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )
    if jid >= 0:
        wrist_qpos_adrs.append(model.jnt_qposadr[jid])


# ============================================================
# Init qpos
# ============================================================

data.qpos[:] = model.qpos0.copy()

# If floating base exists, set quaternion to identity
# qpos[0:3] = xyz
# qpos[3:7] = wxyz quaternion
if model.nq >= 7:
    data.qpos[3] = 1.0
    data.qpos[4] = 0.0
    data.qpos[5] = 0.0
    data.qpos[6] = 0.0

if FIX_WRIST_TO_ZERO:
    for adr in wrist_qpos_adrs:
        data.qpos[adr] = 0.0

mujoco.mj_forward(model, data)


# ============================================================
# Helper
# ============================================================

def set_arm_frame(frame_idx):
    q = q_traj[frame_idx]

    for adr, value in zip(qpos_adrs, q):
        data.qpos[adr] = value

    if FIX_WRIST_TO_ZERO:
        for adr in wrist_qpos_adrs:
            data.qpos[adr] = 0.0

    mujoco.mj_forward(model, data)


# ============================================================
# Camera
# ============================================================

cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat[:] = np.array([
    0.0,
    0.15,
    0.85,
])
cam.distance = 2.0
cam.azimuth = 230
cam.elevation = -8


# ============================================================
# Renderer
# ============================================================

renderer = mujoco.Renderer(
    model,
    height=HEIGHT,
    width=WIDTH,
)

# Optional: show contact points / etc if you want
renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1


# ============================================================
# Video writer
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_MP4_PATH),
    exist_ok=True,
)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    OUTPUT_MP4_PATH,
    fourcc,
    FPS * SPEED,
    (WIDTH, HEIGHT),
)

if not writer.isOpened():
    raise RuntimeError(
        f"Failed to open video writer: {OUTPUT_MP4_PATH}"
    )


# ============================================================
# Print first frame q
# ============================================================

print()
print("========== First frame ==========")
for name, value in zip(joint_names, q_traj[START_FRAME]):
    print(
        f"{name:30s}: "
        f"{value:+.6f} rad  "
        f"{np.degrees(value):+.2f} deg"
    )


# ============================================================
# Render loop
# ============================================================

print()
print("========== Rendering ==========")
print("Output:", OUTPUT_MP4_PATH)
print("Resolution:", WIDTH, "x", HEIGHT)
print("FPS:", FPS)
print("Speed:", SPEED)

total_render_frames = END_FRAME - START_FRAME

for out_idx, frame_idx in enumerate(range(START_FRAME, END_FRAME)):
    set_arm_frame(frame_idx)

    renderer.update_scene(
        data,
        camera=cam,
    )

    rgb = renderer.render()   # shape [H, W, 3], RGB uint8

    frame_bgr = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR,
    )

    if DRAW_TEXT:
        cv2.putText(
            frame_bgr,
            f"Frame {frame_idx}/{num_frames - 1}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    writer.write(frame_bgr)

    if out_idx % 20 == 0 or out_idx == total_render_frames - 1:
        print(
            f"[{out_idx + 1:4d}/{total_render_frames}] "
            f"source frame = {frame_idx}"
        )


# ============================================================
# Finish
# ============================================================

writer.release()
renderer.close()

print()
print("========== Done ==========")
print("Saved to:", OUTPUT_MP4_PATH)