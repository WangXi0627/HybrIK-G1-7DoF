import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import torch


# =========================
# Config
# =========================
PT_PATH = "res_dance_x/hybrikx_output.pt"
OUTPUT_PATH = "res_dance_x/arm_joints_3d.mp4"

FPS = 30

# 固定坐标范围，避免动画每帧自动缩放造成抖动
X_LIM = (-0.4, 0.4)
Y_LIM = (-0.4, 0.4)
Z_LIM = (-0.4, 0.4)


# =========================
# Load HybrIK-X output
# =========================
data = torch.load(
    PT_PATH,
    map_location="cpu",
    # weights_only=False,
)

xyz = np.asarray(data["pred_xyz_hybrik"])

if xyz.ndim != 3 or xyz.shape[1:] != (71, 3):
    raise ValueError(
        f"Expected pred_xyz_hybrik shape [N, 71, 3], "
        f"but got {xyz.shape}"
    )

num_frames = xyz.shape[0]

print("Loaded:", PT_PATH)
print("pred_xyz_hybrik shape:", xyz.shape)
print("num_frames:", num_frames)


# =========================
# Joint indices
# =========================

# Torso
PELVIS = 0
SPINE3 = 9
NECK = 12

# Arms
LEFT_SHOULDER = 16
RIGHT_SHOULDER = 17

LEFT_ELBOW = 18
RIGHT_ELBOW = 19

LEFT_WRIST = 20
RIGHT_WRIST = 21

LEFT_ARM = [
    LEFT_SHOULDER,
    LEFT_ELBOW,
    LEFT_WRIST,
]

RIGHT_ARM = [
    RIGHT_SHOULDER,
    RIGHT_ELBOW,
    RIGHT_WRIST,
]


# Left hand
LEFT_FINGERS = [
    [LEFT_WRIST, 25, 26, 27],   # index
    [LEFT_WRIST, 28, 29, 30],   # middle
    [LEFT_WRIST, 31, 32, 33],   # pinky
    [LEFT_WRIST, 34, 35, 36],   # ring
    [LEFT_WRIST, 37, 38, 39],   # thumb
]


# Right hand
RIGHT_FINGERS = [
    [RIGHT_WRIST, 40, 41, 42],  # index
    [RIGHT_WRIST, 43, 44, 45],  # middle
    [RIGHT_WRIST, 46, 47, 48],  # pinky
    [RIGHT_WRIST, 49, 50, 51],  # ring
    [RIGHT_WRIST, 52, 53, 54],  # thumb
]


# =========================
# Visualization
# =========================
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")


def draw_chain(points, indices):
    """
    Draw one skeleton chain.
    """
    p = points[indices]

    ax.plot(
        p[:, 0],
        p[:, 1],
        p[:, 2],
        marker="o",
        linewidth=2,
        markersize=4,
    )


def update(frame):
    ax.cla()

    points = xyz[frame]

    # ---------------------
    # Torso reference
    # ---------------------
    draw_chain(
        points,
        [PELVIS, SPINE3, NECK]
    )

    # Shoulder line
    draw_chain(
        points,
        [LEFT_SHOULDER, RIGHT_SHOULDER]
    )

    # ---------------------
    # Arms
    # ---------------------
    draw_chain(points, LEFT_ARM)
    draw_chain(points, RIGHT_ARM)

    # ---------------------
    # Hands
    # ---------------------
    for finger in LEFT_FINGERS:
        draw_chain(points, finger)

    for finger in RIGHT_FINGERS:
        draw_chain(points, finger)

    # ---------------------
    # Joint labels
    # ---------------------
    important_joints = {
        "LS": LEFT_SHOULDER,
        "LE": LEFT_ELBOW,
        "LW": LEFT_WRIST,
        "RS": RIGHT_SHOULDER,
        "RE": RIGHT_ELBOW,
        "RW": RIGHT_WRIST,
    }

    for name, idx in important_joints.items():
        p = points[idx]

        ax.text(
            p[0],
            p[1],
            p[2],
            name,
            fontsize=8,
        )

    # ---------------------
    # Axis
    # ---------------------
    ax.set_xlim(*X_LIM)
    ax.set_ylim(*Y_LIM)
    ax.set_zlim(*Z_LIM)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_box_aspect([1, 1, 1])

    ax.set_title(
        f"HybrIK-X Upper Body + Hands | "
        f"Frame {frame}/{num_frames - 1}"
    )

    # 固定观察角度
    ax.view_init(
        elev=20,
        azim=-70,
    )


# =========================
# Generate animation
# =========================
ani = FuncAnimation(
    fig,
    update,
    frames=num_frames,
    interval=1000 / FPS,
    repeat=False,
)


# =========================
# Save
# =========================
output_dir = os.path.dirname(OUTPUT_PATH)

if output_dir:
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

print("Saving video to:", OUTPUT_PATH)

ani.save(
    OUTPUT_PATH,
    writer="ffmpeg",
    fps=FPS,
)

plt.close(fig)

print("Done.")
print("Saved to:", OUTPUT_PATH)