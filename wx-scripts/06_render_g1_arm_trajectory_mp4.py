import argparse
import os
import cv2
import mujoco
import numpy as np


# ============================================================
# Command-line arguments
# ============================================================

parser = argparse.ArgumentParser(
    description="Render a G1 arm trajectory NPZ to MP4 with MuJoCo."
)

# Paths
parser.add_argument(
    "--xml-path",
    default=(
        "/data/wx/code-IK/unitree_ros/"
        "robots/g1_description/g1_29dof.xml"
    ),
    help="Path to G1 MuJoCo MJCF/XML",
)
parser.add_argument(
    "--npz-path",
    default="res/pick_place/g1_left_arm_4dof.npz",
    help="Input retargeted trajectory NPZ",
)
parser.add_argument(
    "--output-mp4-path",
    default="res/pick_place/g1_left_arm_4dof.mp4",
    help="Output MP4 path",
)

# Render/video
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--fps", type=float, default=30.0)
parser.add_argument("--start-frame", type=int, default=0)
parser.add_argument(
    "--end-frame",
    type=int,
    default=None,
    help="Exclusive end frame; default uses all frames",
)
parser.add_argument(
    "--speed",
    type=float,
    default=1.0,
    help="Playback speed multiplier written into output FPS",
)
# Draw frame index text
parser.add_argument(
    "--draw-text",
    dest="draw_text",
    action="store_true",
    help="Draw frame index text",
)
parser.add_argument(
    "--no-draw-text",
    dest="draw_text",
    action="store_false",
    help="Do not draw frame index text",
)
parser.set_defaults(
    draw_text=True
)
# Whether to force wrist joints to zero
parser.add_argument(
    "--fix-wrist-to-zero",
    dest="fix_wrist_to_zero",
    action="store_true",
    help="Force the three left wrist joints to zero",
)
parser.add_argument(
    "--no-fix-wrist-to-zero",
    dest="fix_wrist_to_zero",
    action="store_false",
    help="Keep wrist trajectory from input NPZ",
)

parser.set_defaults(
    fix_wrist_to_zero=False
)

# Camera
parser.add_argument(
    "--cam-lookat",
    type=float,
    nargs=3,
    metavar=("X", "Y", "Z"),
    default=[0.0, 0.15, 0.85],
    help="Camera look-at point",
)
parser.add_argument("--cam-distance", type=float, default=2.0)
parser.add_argument("--cam-azimuth", type=float, default=230.0)
parser.add_argument("--cam-elevation", type=float, default=-8.0)

args = parser.parse_args()

XML_PATH = args.xml_path
NPZ_PATH = args.npz_path
OUTPUT_MP4_PATH = args.output_mp4_path

WIDTH = args.width
HEIGHT = args.height
FPS = args.fps
START_FRAME = args.start_frame
END_FRAME = args.end_frame
SPEED = args.speed
DRAW_TEXT = args.draw_text
FIX_WRIST_TO_ZERO = args.fix_wrist_to_zero

CAM_LOOKAT = np.asarray(args.cam_lookat, dtype=np.float64)
CAM_DISTANCE = args.cam_distance
CAM_AZIMUTH = args.cam_azimuth
CAM_ELEVATION = args.cam_elevation


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
cam.lookat[:] = CAM_LOOKAT
cam.distance = CAM_DISTANCE
cam.azimuth = CAM_AZIMUTH
cam.elevation = CAM_ELEVATION


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

output_dir = os.path.dirname(OUTPUT_MP4_PATH)
if output_dir:
    os.makedirs(
        output_dir,
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