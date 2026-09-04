#!/usr/bin/env python3
"""
Render the FULL G1 body + RH56DFTP RIGHT hand calibration trajectory.

Optionally, also load a G1 right-arm trajectory NPZ and replay it together
with the right-hand calibration for coordinated arm-hand visualization.

Hand NPZ expected:
    timestamps      [Nh]
    qpos12_rad      [Nh, 12]
    joint_names12   [12]

Optional arm NPZ expected:
    q               [Na, D]     (normally D=7 for right arm)
    joint_names     [D]

The combined MuJoCo model is always the FULL G1 robot with the RH56DFTP
right hand attached.  Joints not supplied by the hand/arm trajectories
remain at model.qpos0.
"""

import argparse
import json
import os

import cv2
import mujoco
import numpy as np


# ============================================================
# CLI
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Render full G1 body + RH56DFTP right hand calibration, "
        "optionally with a right-arm trajectory."
    )
)

parser.add_argument(
    "--xml-path",
    default=(
        "/data/wx/code-IK/unitree_ros/robots/"
        "g1_description/g1_29dof_rh56dftp_right.xml"
    ),
    help="Combined FULL G1 + RH56DFTP right-hand MJCF.",
)

parser.add_argument(
    "--hand-npz",
    required=True,
    help="RH56DFTP right-hand calibration trajectory NPZ.",
)

parser.add_argument(
    "--arm-npz",
    default=None,
    help=(
        "Optional right-arm trajectory NPZ, e.g. "
        "g1_right_arm_7dof_relative.npz."
    ),
)

parser.add_argument(
    "--arm-fps",
    type=float,
    default=30.0,
    help=(
        "Sampling FPS of --arm-npz when that NPZ has no timestamps. "
        "HybrIK video trajectories are normally 30 FPS."
    ),
)

parser.add_argument(
    "--output-mp4",
    default=(
        "res/pick_place/"
        "g1_fullbody_rh56dftp_right_calibration.mp4"
    ),
)

parser.add_argument(
    "--fps",
    type=float,
    default=30.0,
    help="Output video FPS.",
)

parser.add_argument(
    "--speed",
    type=float,
    default=1.0,
    help="Playback speed. 1.0=real time, 0.5=half speed.",
)

parser.add_argument(
    "--width",
    type=int,
    default=1280,
)

parser.add_argument(
    "--height",
    type=int,
    default=720,
)

parser.add_argument(
    "--start-time",
    type=float,
    default=0.0,
)

parser.add_argument(
    "--end-time",
    type=float,
    default=None,
)

# Whole-body camera defaults.
parser.add_argument(
    "--cam-lookat",
    type=float,
    nargs=3,
    default=[0.0, 0.0, 0.85],
    metavar=("X", "Y", "Z"),
)

parser.add_argument(
    "--cam-distance",
    type=float,
    default=2.4,
)

parser.add_argument(
    "--cam-azimuth",
    type=float,
    default=140.0,
    help="Default view favors the robot's right side while keeping full body.",
)

parser.add_argument(
    "--cam-elevation",
    type=float,
    default=-10.0,
)

parser.add_argument(
    "--draw-text",
    dest="draw_text",
    action="store_true",
)

parser.add_argument(
    "--no-draw-text",
    dest="draw_text",
    action="store_false",
)

parser.set_defaults(
    draw_text=True
)

args = parser.parse_args()


# ============================================================
# Basic validation
# ============================================================

if args.fps <= 0:
    raise ValueError("--fps must be > 0")

if args.speed <= 0:
    raise ValueError("--speed must be > 0")

if args.arm_fps <= 0:
    raise ValueError("--arm-fps must be > 0")


# ============================================================
# Load hand trajectory
# ============================================================

hand = np.load(
    args.hand_npz,
    allow_pickle=True,
)

for key in [
    "timestamps",
    "qpos12_rad",
    "joint_names12",
]:
    if key not in hand:
        raise KeyError(
            "Missing key in hand NPZ: {}".format(key)
        )


hand_timestamps = np.asarray(
    hand["timestamps"],
    dtype=np.float64,
)

hand_q = np.asarray(
    hand["qpos12_rad"],
    dtype=np.float64,
)

hand_joint_names_raw = [
    str(x)
    for x in hand["joint_names12"]
]


if hand_q.ndim != 2 or hand_q.shape[1] != 12:
    raise ValueError(
        "qpos12_rad must be [N,12], got {}".format(
            hand_q.shape
        )
    )

if len(hand_timestamps) != len(hand_q):
    raise ValueError(
        "Hand timestamp/frame mismatch."
    )


hand_time = (
    hand_timestamps
    -
    hand_timestamps[0]
)

hand_duration = float(
    hand_time[-1]
)


# Calibration files often omit the "right_" prefix.
hand_joint_names = []

for name in hand_joint_names_raw:
    if name.startswith("right_"):
        hand_joint_names.append(
            name
        )
    else:
        hand_joint_names.append(
            "right_" + name
        )


# ============================================================
# Optional arm trajectory
# ============================================================

arm_q = None
arm_joint_names = None
arm_time = None
arm_duration = None

if args.arm_npz is not None:
    arm = np.load(
        args.arm_npz,
        allow_pickle=True,
    )

    if "q" not in arm:
        raise KeyError(
            "Arm NPZ is missing key: q"
        )

    if "joint_names" not in arm:
        raise KeyError(
            "Arm NPZ is missing key: joint_names"
        )

    arm_q = np.asarray(
        arm["q"],
        dtype=np.float64,
    )

    arm_joint_names = [
        str(x)
        for x in arm["joint_names"]
    ]

    if arm_q.ndim != 2:
        raise ValueError(
            "Arm q must be [N,D], got {}".format(
                arm_q.shape
            )
        )

    if len(arm_joint_names) != arm_q.shape[1]:
        raise ValueError(
            "arm joint_names length does not match q dimension."
        )

    if "timestamps" in arm:
        raw_arm_time = np.asarray(
            arm["timestamps"],
            dtype=np.float64,
        )

        if len(raw_arm_time) != len(arm_q):
            raise ValueError(
                "Arm timestamps/frame mismatch."
            )

        arm_time = (
            raw_arm_time
            -
            raw_arm_time[0]
        )

    else:
        arm_time = (
            np.arange(
                len(arm_q),
                dtype=np.float64,
            )
            /
            args.arm_fps
        )

    arm_duration = float(
        arm_time[-1]
    )


# ============================================================
# Load FULL combined G1 model
# ============================================================

model = mujoco.MjModel.from_xml_path(
    args.xml_path
)

data = mujoco.MjData(
    model
)


def resolve_qpos_addresses(joint_names):
    addresses = []
    ranges = []

    for name in joint_names:
        jid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        if jid < 0:
            raise RuntimeError(
                "Joint not found in FULL G1 model: {}".format(
                    name
                )
            )

        addresses.append(
            int(
                model.jnt_qposadr[jid]
            )
        )

        ranges.append(
            model.jnt_range[jid].copy()
        )

    return (
        addresses,
        np.asarray(
            ranges,
            dtype=np.float64,
        ),
    )


hand_qpos_adrs, hand_ranges = (
    resolve_qpos_addresses(
        hand_joint_names
    )
)


arm_qpos_adrs = None
arm_ranges = None

if arm_joint_names is not None:
    arm_qpos_adrs, arm_ranges = (
        resolve_qpos_addresses(
            arm_joint_names
        )
    )


# ============================================================
# Limit checks
# ============================================================

def check_limits(
    q,
    ranges,
    label,
):
    violations = (
        (q < ranges[None, :, 0] - 1e-6)
        |
        (q > ranges[None, :, 1] + 1e-6)
    )

    count = int(
        np.sum(
            violations
        )
    )

    print(
        "{} joint-limit violations: {}".format(
            label,
            count,
        )
    )

    if count > 0:
        indices = np.argwhere(
            violations
        )

        print(
            "First violations:",
            indices[:10].tolist(),
        )

        raise RuntimeError(
            "{} trajectory exceeds model joint limits.".format(
                label
            )
        )


check_limits(
    hand_q,
    hand_ranges,
    "Hand",
)

if arm_q is not None:
    check_limits(
        arm_q,
        arm_ranges,
        "Arm",
    )


# ============================================================
# Determine render duration
# ============================================================

available_duration = (
    hand_duration
)

if arm_duration is not None:
    # Coordinated visualization normally uses the common overlap.
    available_duration = min(
        hand_duration,
        arm_duration,
    )


start_time = max(
    0.0,
    float(
        args.start_time
    ),
)

if args.end_time is None:
    end_time = available_duration
else:
    end_time = min(
        available_duration,
        float(
            args.end_time
        ),
    )


if end_time <= start_time:
    raise ValueError(
        "No valid playback interval."
    )


source_duration = (
    end_time
    -
    start_time
)

output_duration = (
    source_duration
    /
    args.speed
)

num_output_frames = max(
    1,
    int(
        np.ceil(
            output_duration
            *
            args.fps
        )
    )
)

output_time = (
    np.arange(
        num_output_frames,
        dtype=np.float64,
    )
    /
    args.fps
)

source_time = (
    start_time
    +
    output_time
    *
    args.speed
)

source_time = np.minimum(
    source_time,
    end_time,
)


# ============================================================
# Interpolation
# ============================================================

def interpolate_trajectory(
    t,
    times,
    q,
):
    result = np.empty(
        q.shape[1],
        dtype=np.float64,
    )

    for j in range(
        q.shape[1]
    ):
        result[j] = np.interp(
            t,
            times,
            q[:, j],
        )

    return result


# ============================================================
# Report
# ============================================================

print()
print(
    "=========================================="
)
print(
    "FULL G1 + RH56DFTP RIGHT replay"
)
print(
    "=========================================="
)

print(
    "XML:",
    args.xml_path,
)

print(
    "Model nq/nv/njnt/nbody:",
    model.nq,
    model.nv,
    model.njnt,
    model.nbody,
)

print(
    "Hand NPZ:",
    args.hand_npz,
)

print(
    "Hand frames/duration:",
    len(hand_q),
    hand_duration,
)

if arm_q is None:
    print(
        "Arm NPZ: none -> right arm remains at model.qpos0"
    )
else:
    print(
        "Arm NPZ:",
        args.arm_npz,
    )
    print(
        "Arm frames/duration:",
        len(arm_q),
        arm_duration,
    )
    print(
        "Arm joints:",
        arm_joint_names,
    )

print(
    "All other G1 joints remain at model.qpos0."
)


# ============================================================
# Renderer
# ============================================================

model.vis.global_.offwidth = max(
    model.vis.global_.offwidth,
    args.width,
)

model.vis.global_.offheight = max(
    model.vis.global_.offheight,
    args.height,
)


renderer = mujoco.Renderer(
    model,
    height=args.height,
    width=args.width,
)


camera = mujoco.MjvCamera()

camera.type = (
    mujoco.mjtCamera.mjCAMERA_FREE
)

camera.lookat[:] = np.asarray(
    args.cam_lookat,
    dtype=np.float64,
)

camera.distance = (
    args.cam_distance
)

camera.azimuth = (
    args.cam_azimuth
)

camera.elevation = (
    args.cam_elevation
)


output_dir = os.path.dirname(
    args.output_mp4
)

if output_dir:
    os.makedirs(
        output_dir,
        exist_ok=True,
    )


writer = cv2.VideoWriter(
    args.output_mp4,
    cv2.VideoWriter_fourcc(
        *"mp4v"
    ),
    args.fps,
    (
        args.width,
        args.height,
    ),
)


if not writer.isOpened():
    raise RuntimeError(
        "Failed to open output video: {}".format(
            args.output_mp4
        )
    )


# ============================================================
# Full-body render loop
# ============================================================

try:
    for frame_idx, t in enumerate(
        source_time
    ):
        # CRITICAL:
        # Reset the ENTIRE G1 to its nominal full-body pose first.
        data.qpos[:] = (
            model.qpos0.copy()
        )

        # Optional right arm.
        if arm_q is not None:
            q_arm_now = (
                interpolate_trajectory(
                    float(t),
                    arm_time,
                    arm_q,
                )
            )

            for adr, value in zip(
                arm_qpos_adrs,
                q_arm_now,
            ):
                data.qpos[
                    adr
                ] = value

        # Right RH56DFTP recorded calibration.
        q_hand_now = (
            interpolate_trajectory(
                float(t),
                hand_time,
                hand_q,
            )
        )

        for adr, value in zip(
            hand_qpos_adrs,
            q_hand_now,
        ):
            data.qpos[
                adr
            ] = value

        mujoco.mj_forward(
            model,
            data,
        )

        renderer.update_scene(
            data,
            camera=camera,
        )

        rgb = renderer.render()

        bgr = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2BGR,
        )

        if args.draw_text:
            line1 = (
                "FULL G1 + RH56DFTP RIGHT"
            )

            if arm_q is None:
                line2 = (
                    "hand calibration only | "
                    "arm/body = qpos0"
                )
            else:
                line2 = (
                    "right arm + right hand coordinated replay"
                )

            line3 = (
                "t={:.2f}s speed={:.2f}x".format(
                    float(t),
                    args.speed,
                )
            )

            cv2.putText(
                bgr,
                line1,
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                bgr,
                line2,
                (30, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                bgr,
                line3,
                (30, 108),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        writer.write(
            bgr
        )

        if (
            frame_idx % 30 == 0
            or
            frame_idx
            ==
            num_output_frames - 1
        ):
            print(
                "[{:4d}/{:4d}] t={:.3f}s".format(
                    frame_idx + 1,
                    num_output_frames,
                    float(t),
                )
            )

finally:
    writer.release()
    renderer.close()


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
    args.output_mp4,
)
