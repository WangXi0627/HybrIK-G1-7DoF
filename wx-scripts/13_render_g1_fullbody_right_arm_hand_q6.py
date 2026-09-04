#!/usr/bin/env python3
"""
Render FULL G1 body + RH56DFTP RIGHT hand using 6-DoF hand commands.

This is the recommended version for consistency with the real robot:
- real hardware control uses 6 hand DoFs
- MuJoCo hand state needs 12 joint qpos
- this script expands q6 -> q12 internally via the RH56DFTP mimic relations

Optional:
- also load a right-arm trajectory NPZ and replay it together with the hand

Hand NPZ expected:
    timestamps      [N]
    q6_rad          [N, 6]
    command_names6  [6]

Optional diagnostic keys:
    qpos12_rad      [N, 12]
    joint_names12   [12]

Optional arm NPZ expected:
    q               [N, D]
    joint_names     [D]
    timestamps      [N]          (optional)

The combined model is:
    FULL G1 + RH56DFTP right hand
"""

import argparse
import os
import json

import cv2
import mujoco
import numpy as np


# ============================================================
# CLI
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Render full G1 body + RH56DFTP right hand from 6-DoF hand commands, "
        "optionally with a right-arm trajectory."
    )
)

parser.add_argument(
    "--xml-path",
    default=(
        "/data/wx/code-IK/unitree_ros/robots/"
        "g1_description/g1_29dof_rh56dftp_right.xml"
    ),
    help="Combined FULL G1 + RH56DFTP-right MJCF.",
)

parser.add_argument(
    "--hand-npz",
    required=True,
    help="RH56DFTP right-hand trajectory NPZ containing q6_rad + command_names6.",
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
        "FPS used when arm NPZ has no timestamps. "
        "Usually 30 for HybrIK-derived trajectories."
    ),
)

parser.add_argument(
    "--output-mp4",
    default=(
        "res/pick_place/"
        "g1_fullbody_rh56dftp_right_q6.mp4"
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

# Whole-body camera defaults
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

parser.set_defaults(draw_text=True)

args = parser.parse_args()


# ============================================================
# Basic checks
# ============================================================

if args.fps <= 0:
    raise ValueError("--fps must be > 0")

if args.speed <= 0:
    raise ValueError("--speed must be > 0")

if args.arm_fps <= 0:
    raise ValueError("--arm-fps must be > 0")


# ============================================================
# Utilities
# ============================================================

def interp_traj(t, times, q):
    """
    q: [N, D]
    times: [N]
    return: [D]
    """
    out = np.empty(q.shape[1], dtype=np.float64)
    for j in range(q.shape[1]):
        out[j] = np.interp(t, times, q[:, j])
    return out


def resolve_qpos_addresses(model, joint_names):
    qpos_adrs = []
    joint_ranges = []

    for name in joint_names:
        jid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        if jid < 0:
            raise RuntimeError(
                "Joint not found in model: {}".format(name)
            )

        qpos_adrs.append(int(model.jnt_qposadr[jid]))
        joint_ranges.append(model.jnt_range[jid].copy())

    return qpos_adrs, np.asarray(joint_ranges, dtype=np.float64)


def check_limits(q, ranges, label):
    violations = (
        (q < ranges[None, :, 0] - 1e-6)
        |
        (q > ranges[None, :, 1] + 1e-6)
    )

    count = int(np.sum(violations))

    print("{} joint-limit violations: {}".format(label, count))

    if count > 0:
        bad = np.argwhere(violations)
        print("First violations:", bad[:10].tolist())
        raise RuntimeError(
            "{} trajectory exceeds model joint limits.".format(label)
        )


# ============================================================
# RH56DFTP q6 -> q12 expansion
# ============================================================

EXPECTED_Q6_NAMES = [
    "pinky",
    "ring",
    "middle",
    "index",
    "thumb_bend",
    "thumb_rotation",
]

Q12_JOINT_NAMES_MODEL = [
    "right_thumb_proximal_yaw_joint",
    "right_thumb_proximal_pitch_joint",
    "right_thumb_intermediate_joint",
    "right_thumb_distal_joint",
    "right_index_proximal_joint",
    "right_index_intermediate_joint",
    "right_middle_proximal_joint",
    "right_middle_intermediate_joint",
    "right_ring_proximal_joint",
    "right_ring_intermediate_joint",
    "right_pinky_proximal_joint",
    "right_pinky_intermediate_joint",
]


def reorder_q6(q6, names):
    """
    Reorder incoming q6 columns to:
        [pinky, ring, middle, index, thumb_bend, thumb_rotation]
    """
    names = [str(x) for x in names]
    index_of = {name: i for i, name in enumerate(names)}

    missing = [name for name in EXPECTED_Q6_NAMES if name not in index_of]
    if missing:
        raise KeyError(
            "Missing q6 commands: {}".format(missing)
        )

    cols = [index_of[name] for name in EXPECTED_Q6_NAMES]
    return q6[:, cols]


def q6_to_q12(q6_ordered):
    """
    Ordered q6:
        0 pinky
        1 ring
        2 middle
        3 index
        4 thumb_bend
        5 thumb_rotation

    Output q12 in the model joint order:
        [
            right_thumb_proximal_yaw_joint,
            right_thumb_proximal_pitch_joint,
            right_thumb_intermediate_joint,
            right_thumb_distal_joint,
            right_index_proximal_joint,
            right_index_intermediate_joint,
            right_middle_proximal_joint,
            right_middle_intermediate_joint,
            right_ring_proximal_joint,
            right_ring_intermediate_joint,
            right_pinky_proximal_joint,
            right_pinky_intermediate_joint,
        ]
    """
    q6_ordered = np.asarray(q6_ordered, dtype=np.float64)

    pinky = q6_ordered[:, 0]
    ring = q6_ordered[:, 1]
    middle = q6_ordered[:, 2]
    index = q6_ordered[:, 3]
    thumb_bend = q6_ordered[:, 4]
    thumb_rotation = q6_ordered[:, 5]

    thumb_proximal_yaw = thumb_rotation
    thumb_proximal_pitch = thumb_bend
    thumb_intermediate = 1.334 * thumb_bend
    thumb_distal = 0.667 * thumb_bend

    index_proximal = index
    index_intermediate = -0.04545 + 1.06399 * index

    middle_proximal = middle
    middle_intermediate = -0.04545 + 1.06399 * middle

    ring_proximal = ring
    ring_intermediate = -0.04545 + 1.06399 * ring

    pinky_proximal = pinky
    pinky_intermediate = -0.04545 + 1.06399 * pinky

    q12 = np.stack(
        [
            thumb_proximal_yaw,
            thumb_proximal_pitch,
            thumb_intermediate,
            thumb_distal,
            index_proximal,
            index_intermediate,
            middle_proximal,
            middle_intermediate,
            ring_proximal,
            ring_intermediate,
            pinky_proximal,
            pinky_intermediate,
        ],
        axis=1,
    )

    return q12


# ============================================================
# Load hand NPZ
# ============================================================

hand = np.load(args.hand_npz, allow_pickle=True)

for key in ["timestamps", "q6_rad", "command_names6"]:
    if key not in hand:
        raise KeyError("Missing key in hand NPZ: {}".format(key))

hand_timestamps = np.asarray(hand["timestamps"], dtype=np.float64)
hand_q6_raw = np.asarray(hand["q6_rad"], dtype=np.float64)
hand_command_names6 = [str(x) for x in hand["command_names6"]]

if hand_q6_raw.ndim != 2 or hand_q6_raw.shape[1] != 6:
    raise ValueError(
        "q6_rad must be [N,6], got {}".format(hand_q6_raw.shape)
    )

if len(hand_timestamps) != len(hand_q6_raw):
    raise ValueError("Hand timestamp/frame mismatch.")

hand_time = hand_timestamps - hand_timestamps[0]
hand_duration = float(hand_time[-1])

hand_q6 = reorder_q6(
    hand_q6_raw,
    hand_command_names6,
)

hand_q12 = q6_to_q12(hand_q6)


# Optional diagnostic comparison against recorded qpos12_rad
if "qpos12_rad" in hand:
    hand_q12_gt = np.asarray(hand["qpos12_rad"], dtype=np.float64)

    if hand_q12_gt.shape == hand_q12.shape:
        q12_diff = np.abs(hand_q12 - hand_q12_gt)
        print("q6->q12 vs recorded qpos12:")
        print("  mean abs diff:", float(q12_diff.mean()))
        print("  max  abs diff:", float(q12_diff.max()))
    else:
        print(
            "Warning: NPZ has qpos12_rad but shape mismatch:",
            hand_q12_gt.shape,
            "vs",
            hand_q12.shape,
        )


# ============================================================
# Optional arm NPZ
# ============================================================

arm_q = None
arm_joint_names = None
arm_time = None
arm_duration = None

if args.arm_npz is not None:
    arm = np.load(args.arm_npz, allow_pickle=True)

    if "q" not in arm:
        raise KeyError("Arm NPZ missing key: q")

    if "joint_names" not in arm:
        raise KeyError("Arm NPZ missing key: joint_names")

    arm_q = np.asarray(arm["q"], dtype=np.float64)
    arm_joint_names = [str(x) for x in arm["joint_names"]]

    if arm_q.ndim != 2:
        raise ValueError("Arm q must be [N,D], got {}".format(arm_q.shape))

    if len(arm_joint_names) != arm_q.shape[1]:
        raise ValueError(
            "Arm joint_names length does not match q dimension."
        )

    if "timestamps" in arm:
        raw_arm_time = np.asarray(arm["timestamps"], dtype=np.float64)

        if len(raw_arm_time) != len(arm_q):
            raise ValueError("Arm timestamps/frame mismatch.")

        arm_time = raw_arm_time - raw_arm_time[0]
    else:
        arm_time = np.arange(len(arm_q), dtype=np.float64) / args.arm_fps

    arm_duration = float(arm_time[-1])


# ============================================================
# Load FULL model
# ============================================================

model = mujoco.MjModel.from_xml_path(args.xml_path)
data = mujoco.MjData(model)

hand_qpos_adrs, hand_ranges = resolve_qpos_addresses(
    model,
    Q12_JOINT_NAMES_MODEL,
)

arm_qpos_adrs = None
arm_ranges = None

if arm_joint_names is not None:
    arm_qpos_adrs, arm_ranges = resolve_qpos_addresses(
        model,
        arm_joint_names,
    )


# ============================================================
# Joint limit checks
# ============================================================

check_limits(hand_q12, hand_ranges, "Hand(q6->q12)")

if arm_q is not None:
    check_limits(arm_q, arm_ranges, "Arm")


# ============================================================
# Duration and playback timeline
# ============================================================

available_duration = hand_duration

if arm_duration is not None:
    available_duration = min(available_duration, arm_duration)

start_time = max(0.0, float(args.start_time))

if args.end_time is None:
    end_time = available_duration
else:
    end_time = min(available_duration, float(args.end_time))

if end_time <= start_time:
    raise ValueError("No valid playback interval.")

source_duration = end_time - start_time
output_duration = source_duration / args.speed

num_output_frames = max(
    1,
    int(np.ceil(output_duration * args.fps))
)

output_time = np.arange(num_output_frames, dtype=np.float64) / args.fps
source_time = start_time + output_time * args.speed
source_time = np.minimum(source_time, end_time)


# ============================================================
# Report
# ============================================================

print()
print("==========================================")
print("FULL G1 + RH56DFTP RIGHT replay (6DoF)")
print("==========================================")
print("XML:", args.xml_path)
print("Model nq/nv/njnt/nbody:", model.nq, model.nv, model.njnt, model.nbody)

print()
print("Hand NPZ:", args.hand_npz)
print("Hand frames:", len(hand_q6))
print("Hand duration:", hand_duration)
print("Input q6 command names:", hand_command_names6)
print("Normalized q6 order:", EXPECTED_Q6_NAMES)
print("Expanded q12 joints:", Q12_JOINT_NAMES_MODEL)

if "metadata_json" in hand:
    try:
        metadata = json.loads(str(hand["metadata_json"].item()))
        print("Hand metadata keys:", sorted(list(metadata.keys())))
    except Exception:
        pass

print()
if arm_q is None:
    print("Arm NPZ: none -> right arm remains at model.qpos0")
else:
    print("Arm NPZ:", args.arm_npz)
    print("Arm frames:", len(arm_q))
    print("Arm duration:", arm_duration)
    print("Arm joints:", arm_joint_names)

print("All other G1 joints remain at model.qpos0.")


# ============================================================
# Renderer
# ============================================================

model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)

renderer = mujoco.Renderer(
    model,
    height=args.height,
    width=args.width,
)

camera = mujoco.MjvCamera()
camera.type = mujoco.mjtCamera.mjCAMERA_FREE
camera.lookat[:] = np.asarray(args.cam_lookat, dtype=np.float64)
camera.distance = args.cam_distance
camera.azimuth = args.cam_azimuth
camera.elevation = args.cam_elevation


# ============================================================
# Video writer
# ============================================================

output_dir = os.path.dirname(args.output_mp4)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

writer = cv2.VideoWriter(
    args.output_mp4,
    cv2.VideoWriter_fourcc(*"mp4v"),
    args.fps,
    (args.width, args.height),
)

if not writer.isOpened():
    raise RuntimeError("Failed to open output video: {}".format(args.output_mp4))


# ============================================================
# Render loop
# ============================================================

try:
    for frame_idx, t in enumerate(source_time):
        # Reset entire full-body robot first.
        data.qpos[:] = model.qpos0.copy()

        # Optional right arm replay.
        if arm_q is not None:
            q_arm_now = interp_traj(float(t), arm_time, arm_q)
            for adr, value in zip(arm_qpos_adrs, q_arm_now):
                data.qpos[adr] = value

        # Right hand replay from q6 -> q12.
        q_hand_now = interp_traj(float(t), hand_time, hand_q12)
        for adr, value in zip(hand_qpos_adrs, q_hand_now):
            data.qpos[adr] = value

        mujoco.mj_forward(model, data)

        renderer.update_scene(data, camera=camera)
        rgb = renderer.render()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if args.draw_text:
            line1 = "FULL G1 + RH56DFTP RIGHT (6DoF hand control)"
            if arm_q is None:
                line2 = "hand=q6 replay -> q12 mimic expansion | arm/body=qpos0"
            else:
                line2 = "right arm + right hand coordinated replay (arm7 + hand6)"
            line3 = "t={:.2f}s speed={:.2f}x".format(float(t), args.speed)

            cv2.putText(
                bgr,
                line1,
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.78,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                bgr,
                line2,
                (30, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                bgr,
                line3,
                (30, 108),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        writer.write(bgr)

        if frame_idx % 30 == 0 or frame_idx == num_output_frames - 1:
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
print("==========================================")
print("Done")
print("==========================================")
print("Saved:", args.output_mp4)
