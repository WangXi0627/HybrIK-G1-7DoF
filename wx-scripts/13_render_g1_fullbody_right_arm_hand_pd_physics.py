#!/usr/bin/env python3
import argparse
import os

import cv2
import mujoco
import numpy as np


HAND_COMMAND_ORDER = [
    "pinky",
    "ring",
    "middle",
    "index",
    "thumb_bend",
    "thumb_rotation",
]

HAND_JOINT_NAMES = [
    "right_pinky_proximal_joint",
    "right_ring_proximal_joint",
    "right_middle_proximal_joint",
    "right_index_proximal_joint",
    "right_thumb_proximal_pitch_joint",
    "right_thumb_proximal_yaw_joint",
]

HAND_ACTUATOR_NAMES = [
    "right_pinky_act",
    "right_ring_act",
    "right_middle_act",
    "right_index_act",
    "right_thumb_pitch_act",
    "right_thumb_yaw_act",
]


HAND_DEPENDENT_JOINT_NAMES = [
    "right_pinky_intermediate_joint",
    "right_ring_intermediate_joint",
    "right_middle_intermediate_joint",
    "right_index_intermediate_joint",
    "right_thumb_intermediate_joint",
    "right_thumb_distal_joint",
]


def reorder_columns(q, names, expected):
    names = [str(x) for x in names]
    idx = {name: i for i, name in enumerate(names)}
    missing = [x for x in expected if x not in idx]
    if missing:
        raise KeyError(f"Missing trajectory channels: {missing}")
    return q[:, [idx[x] for x in expected]]


def interp_vec(t, ts, q):
    return np.asarray(
        [np.interp(t, ts, q[:, i]) for i in range(q.shape[1])],
        dtype=np.float64,
    )


def resolve_joint(model, name):
    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )
    if jid < 0:
        raise RuntimeError(f"Joint not found: {name}")
    return (
        jid,
        int(model.jnt_qposadr[jid]),
        int(model.jnt_dofadr[jid]),
    )


def resolve_actuator(model, name):
    aid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        name,
    )
    if aid < 0:
        raise RuntimeError(f"Actuator not found: {name}")
    return aid


def infer_arm_actuator(model, joint_name):
    candidates = [joint_name]
    if joint_name.endswith("_joint"):
        candidates.insert(0, joint_name[:-6])

    for name in candidates:
        aid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            name,
        )
        if aid >= 0:
            return aid

    raise RuntimeError(
        f"Cannot infer actuator for arm joint: {joint_name}"
    )


parser = argparse.ArgumentParser(
    description=(
        "Full G1 + RH56DFTP right-hand physics replay using "
        "6-DoF position-PD and MuJoCo equality constraints."
    )
)

parser.add_argument(
    "--xml-path",
    default=(
        "/data/wx/code-IK/unitree_ros/robots/"
        "g1_description/g1_29dof_rh56dftp_right.xml"
    ),
)
parser.add_argument("--hand-npz", required=True)
parser.add_argument("--arm-npz", default=None)
parser.add_argument("--arm-fps", type=float, default=30.0)

parser.add_argument(
    "--output-mp4",
    default=(
        "res/pick_place/"
        "g1_fullbody_right_hand_pd_physics.mp4"
    ),
)

parser.add_argument("--render-fps", type=float, default=30.0)
parser.add_argument("--speed", type=float, default=1.0)

parser.add_argument(
    "--hand-kp",
    type=float,
    nargs=6,
    default=[0.5] * 6,
)
parser.add_argument(
    "--hand-kd",
    type=float,
    nargs=6,
    default=[0.01] * 6,
)

parser.add_argument("--arm-kp", type=float, default=40.0)
parser.add_argument("--arm-kd", type=float, default=2.0)

parser.add_argument(
    "--arm-gravity-comp",
    dest="arm_gravity_comp",
    action="store_true",
    help="Add MuJoCo qfrc_bias as arm feed-forward torque.",
)
parser.add_argument(
    "--no-arm-gravity-comp",
    dest="arm_gravity_comp",
    action="store_false",
)
parser.set_defaults(arm_gravity_comp=True)

parser.add_argument(
    "--support-mode",
    choices=["freeze-uncontrolled", "free"],
    default="freeze-uncontrolled",
    help=(
        "freeze-uncontrolled: keep floating base and all non-controlled G1 "
        "DoFs at model.qpos0 while hand/optional arm use physics PD. "
        "Recommended for arm-hand retargeting tests. "
        "free: fully dynamic humanoid; without a balance controller it will fall."
    ),
)

parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument(
    "--cam-lookat",
    type=float,
    nargs=3,
    default=[0.0, 0.0, 0.85],
)
parser.add_argument("--cam-distance", type=float, default=2.4)
parser.add_argument("--cam-azimuth", type=float, default=140.0)
parser.add_argument("--cam-elevation", type=float, default=-10.0)
parser.add_argument("--no-draw-text", action="store_true")

args = parser.parse_args()

if args.render_fps <= 0:
    raise ValueError("--render-fps must be > 0")
if args.speed <= 0:
    raise ValueError("--speed must be > 0")
if args.arm_fps <= 0:
    raise ValueError("--arm-fps must be > 0")

hand_kp = np.asarray(args.hand_kp, dtype=np.float64)
hand_kd = np.asarray(args.hand_kd, dtype=np.float64)


# ============================================================
# Hand trajectory
# ============================================================

record = np.load(args.hand_npz, allow_pickle=True)

if "timestamps" not in record:
    raise KeyError("Hand NPZ missing timestamps")

timestamps = np.asarray(
    record["timestamps"],
    dtype=np.float64,
)

if "q_des6_rad" in record:
    q6_raw = np.asarray(record["q_des6_rad"], dtype=np.float64)
    q_key = "q_des6_rad"
elif "q6_rad" in record:
    q6_raw = np.asarray(record["q6_rad"], dtype=np.float64)
    q_key = "q6_rad"
else:
    raise KeyError("Need q_des6_rad or q6_rad")

if "joint_names6" in record:
    names6 = [str(x) for x in record["joint_names6"]]
    names_key = "joint_names6"
elif "command_names6" in record:
    names6 = [str(x) for x in record["command_names6"]]
    names_key = "command_names6"
else:
    raise KeyError("Need joint_names6 or command_names6")

if timestamps.ndim != 1:
    raise ValueError(f"timestamps must be [N], got {timestamps.shape}")

if q6_raw.ndim != 2 or q6_raw.shape[1] != 6:
    raise ValueError(f"{q_key} must be [N,6], got {q6_raw.shape}")

if len(timestamps) != len(q6_raw):
    raise ValueError("Hand timestamp/frame mismatch")

# Important: preserve original timestamps. Do not subtract timestamps[0].
q_des6 = reorder_columns(
    q6_raw,
    names6,
    HAND_COMMAND_ORDER,
)

hand_end_time = float(timestamps[-1])


# ============================================================
# Optional arm trajectory
# ============================================================

arm_q = None
arm_names = None
arm_times = None
arm_end_time = None

if args.arm_npz is not None:
    arm_record = np.load(args.arm_npz, allow_pickle=True)

    if "q" not in arm_record or "joint_names" not in arm_record:
        raise KeyError("Arm NPZ requires q and joint_names")

    arm_q = np.asarray(arm_record["q"], dtype=np.float64)
    arm_names = [str(x) for x in arm_record["joint_names"]]

    if arm_q.ndim != 2:
        raise ValueError(f"Arm q must be [N,D], got {arm_q.shape}")
    if len(arm_names) != arm_q.shape[1]:
        raise ValueError("Arm joint_names/q dimension mismatch")

    if "timestamps" in arm_record:
        arm_times = np.asarray(
            arm_record["timestamps"],
            dtype=np.float64,
        )
        if len(arm_times) != len(arm_q):
            raise ValueError("Arm timestamp/frame mismatch")
    else:
        arm_times = (
            np.arange(len(arm_q), dtype=np.float64)
            /
            args.arm_fps
        )

    arm_end_time = float(arm_times[-1])


# ============================================================
# MuJoCo model
# ============================================================

model = mujoco.MjModel.from_xml_path(args.xml_path)
data = mujoco.MjData(model)
dt = float(model.opt.timestep)

hand_qpos = []
hand_qvel = []
hand_act = []

for joint_name, actuator_name in zip(
    HAND_JOINT_NAMES,
    HAND_ACTUATOR_NAMES,
):
    _, qadr, vadr = resolve_joint(model, joint_name)
    aid = resolve_actuator(model, actuator_name)

    hand_qpos.append(qadr)
    hand_qvel.append(vadr)
    hand_act.append(aid)

arm_joint_ids = []
arm_qpos = []
arm_qvel = []
arm_act = []

if arm_q is not None:
    for joint_name in arm_names:
        jid, qadr, vadr = resolve_joint(model, joint_name)
        aid = infer_arm_actuator(model, joint_name)

        arm_joint_ids.append(jid)
        arm_qpos.append(qadr)
        arm_qvel.append(vadr)
        arm_act.append(aid)


# ============================================================
# Support mask for upper-body/hand physics tests
# ============================================================
#
# The stock G1 model contains a floating base.  If the full humanoid is
# simulated with gravity but without a locomotion/balance controller,
# it will naturally fall.  For arm-hand retargeting we therefore freeze:
#   - floating-base DoFs
#   - all non-controlled G1 joints
#
# while leaving these DoFs dynamic:
#   - 6 active RH56DFTP joints
#   - 6 RH56DFTP dependent joints (moved by equality constraints)
#   - optional arm trajectory joints
#
# This keeps the robot standing while preserving the hand's motor-PD +
# equality-constraint physics.

dynamic_qpos_adrs = set(hand_qpos)
dynamic_qvel_adrs = set(hand_qvel)

for dep_name in HAND_DEPENDENT_JOINT_NAMES:
    _, qadr, vadr = resolve_joint(model, dep_name)
    dynamic_qpos_adrs.add(qadr)
    dynamic_qvel_adrs.add(vadr)

if arm_q is not None:
    dynamic_qpos_adrs.update(arm_qpos)
    dynamic_qvel_adrs.update(arm_qvel)

freeze_qpos_adrs = np.asarray(
    [i for i in range(model.nq) if i not in dynamic_qpos_adrs],
    dtype=np.int32,
)

freeze_qvel_adrs = np.asarray(
    [i for i in range(model.nv) if i not in dynamic_qvel_adrs],
    dtype=np.int32,
)

qpos0_reference = model.qpos0.copy()


def enforce_support():
    """
    Hold the floating base and all non-controlled DoFs at the nominal pose.

    This is intentionally a support constraint for arm-hand retargeting,
    not a whole-body balance controller.
    """
    if args.support_mode != "freeze-uncontrolled":
        return

    data.qpos[freeze_qpos_adrs] = qpos0_reference[freeze_qpos_adrs]
    data.qvel[freeze_qvel_adrs] = 0.0


def write_motor_ctrl(aid, value):
    """
    Write actuator control correctly.

    IMPORTANT:
    Unitree G1 <motor> actuators in g1_29dof.xml do not define ctrlrange.
    For such actuators actuator_ctrllimited is false, and the stored
    actuator_ctrlrange may be [0, 0].  Clipping unconditionally to that
    array would silently turn every arm torque into zero.

    RH56DFTP motors do explicitly define ctrlrange="-1 1", so those are
    clipped here.
    """
    if bool(model.actuator_ctrllimited[aid]):
        lo, hi = model.actuator_ctrlrange[aid]
        value = np.clip(value, lo, hi)

    data.ctrl[aid] = value
    return float(value)


print("==============================================")
print("FULL G1 + RH56DFTP RIGHT physics replay")
print("==============================================")
print("XML:", args.xml_path)
print("MuJoCo timestep:", dt)
print("Support mode:", args.support_mode)
print("Hand NPZ:", args.hand_npz)
print("Hand value key:", q_key)
print("Hand name key:", names_key)
print("timestamps[0]:", timestamps[0])
print("timestamps[-1]:", timestamps[-1])

print()
print("Hand joint/actuator mapping:")
for i in range(6):
    aid = hand_act[i]
    print(
        f"  {HAND_COMMAND_ORDER[i]:14s}"
        f" -> {HAND_JOINT_NAMES[i]:38s}"
        f" qpos={hand_qpos[i]:3d}"
        f" qvel={hand_qvel[i]:3d}"
        f" actuator={HAND_ACTUATOR_NAMES[i]:24s}"
        f" ctrlrange={model.actuator_ctrlrange[aid].tolist()}"
    )

if arm_q is not None:
    print()
    print("Arm NPZ:", args.arm_npz)
    print("Arm joints:")
    for i, name in enumerate(arm_names):
        aname = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            arm_act[i],
        )
        jid = arm_joint_ids[i]
        ctrl_limited = bool(model.actuator_ctrllimited[arm_act[i]])
        joint_force_limited = bool(model.jnt_actfrclimited[jid])

        print(
            f"  {name:36s} "
            f"qpos={arm_qpos[i]} "
            f"qvel={arm_qvel[i]} "
            f"actuator={aname} "
            f"ctrl_limited={ctrl_limited} "
            f"joint_actfrcrange={model.jnt_actfrcrange[jid].tolist() if joint_force_limited else 'unlimited'}"
        )


# ============================================================
# Initial state
# ============================================================

data.qpos[:] = model.qpos0.copy()
data.qvel[:] = 0.0

# Active hand joints only.
# np.interp(0, timestamps, ...) keeps the first valid target if timestamps[0] > 0.
q_hand0 = interp_vec(0.0, timestamps, q_des6)

for adr, value in zip(hand_qpos, q_hand0):
    data.qpos[adr] = value

# Do not initialize dependent joints manually.
# Equality constraints remain responsible for them.

if arm_q is not None:
    q_arm0 = interp_vec(0.0, arm_times, arm_q)
    for adr, value in zip(arm_qpos, q_arm0):
        data.qpos[adr] = value

mujoco.mj_forward(model, data)
enforce_support()
mujoco.mj_forward(model, data)


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
camera.type = mujoco.mjtCamera.mjCAMERA_FREE
camera.lookat[:] = np.asarray(
    args.cam_lookat,
    dtype=np.float64,
)
camera.distance = args.cam_distance
camera.azimuth = args.cam_azimuth
camera.elevation = args.cam_elevation

out_dir = os.path.dirname(args.output_mp4)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)

writer = cv2.VideoWriter(
    args.output_mp4,
    cv2.VideoWriter_fourcc(*"mp4v"),
    args.render_fps,
    (args.width, args.height),
)

if not writer.isOpened():
    raise RuntimeError(
        f"Failed to open video writer: {args.output_mp4}"
    )


# ============================================================
# Physics replay
# ============================================================

end_time = hand_end_time
if arm_end_time is not None:
    end_time = min(end_time, arm_end_time)

render_period = 1.0 / args.render_fps
next_render_time = 0.0

sim_time = 0.0
physics_steps = 0
render_frames = 0

try:
    while sim_time <= end_time:
        traj_time = sim_time * args.speed

        # ----------------------------------------------------
        # Hand: 6 active joints only
        # ----------------------------------------------------
        q_target6 = interp_vec(
            traj_time,
            timestamps,
            q_des6,
        )

        for i in range(6):
            q_now = data.qpos[hand_qpos[i]]
            dq_now = data.qvel[hand_qvel[i]]

            tau = (
                hand_kp[i] * (q_target6[i] - q_now)
                - hand_kd[i] * dq_now
            )

            aid = hand_act[i]
            write_motor_ctrl(
                aid,
                tau,
            )

        # ----------------------------------------------------
        # Optional arm: position PD
        # ----------------------------------------------------
        if arm_q is not None:
            q_arm_target = interp_vec(
                traj_time,
                arm_times,
                arm_q,
            )

            for i in range(arm_q.shape[1]):
                q_now = data.qpos[arm_qpos[i]]
                dq_now = data.qvel[arm_qvel[i]]

                tau_pd = (
                    args.arm_kp * (q_arm_target[i] - q_now)
                    - args.arm_kd * dq_now
                )

                # qfrc_bias contains gravity + Coriolis/centrifugal
                # generalized forces.  With a fixed/support body this
                # provides useful feed-forward compensation for the arm,
                # preventing the arm from drooping just to generate
                # gravity-support torque through position error.
                tau_ff = 0.0

                if args.arm_gravity_comp:
                    tau_ff = float(
                        data.qfrc_bias[
                            arm_qvel[i]
                        ]
                    )

                tau = (
                    tau_pd
                    +
                    tau_ff
                )

                aid = arm_act[i]

                # Do NOT unconditionally clip with actuator_ctrlrange.
                # G1 motors do not specify ctrlrange in the stock MJCF.
                # MuJoCo enforces the joint actuatorfrcrange itself.
                write_motor_ctrl(
                    aid,
                    tau,
                )

        # Keep the robot supported before advancing physics.
        enforce_support()

        # Real MuJoCo physics step for the controlled arm/hand DoFs.
        mujoco.mj_step(model, data)

        # Remove any numerical drift of the floating base / uncontrolled
        # body joints, while leaving hand dependent joints free for equality.
        enforce_support()
        mujoco.mj_forward(model, data)

        physics_steps += 1
        sim_time += dt

        # ----------------------------------------------------
        # Independent video rendering
        # ----------------------------------------------------
        if sim_time + 1e-12 >= next_render_time:
            renderer.update_scene(
                data,
                camera=camera,
            )

            rgb = renderer.render()
            frame = cv2.cvtColor(
                rgb,
                cv2.COLOR_RGB2BGR,
            )

            if not args.no_draw_text:
                cv2.putText(
                    frame,
                    "FULL G1 + RH56DFTP RIGHT | PD physics",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.78,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                mode = (
                    "hand6 PD"
                    if arm_q is None
                    else
                    "right arm + hand6 PD"
                )

                cv2.putText(
                    frame,
                    mode,
                    (30, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.68,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    (
                        f"sim={sim_time:.3f}s "
                        f"traj={traj_time:.3f}s "
                        f"dt={dt:.4f}s"
                    ),
                    (30, 108),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.66,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            writer.write(frame)
            render_frames += 1
            next_render_time += render_period

        # Once per simulated second, report tracking error.
        step_per_sec = max(
            1,
            int(round(1.0 / dt)),
        )

        if physics_steps % step_per_sec == 0:
            q_now6 = np.asarray(
                [data.qpos[x] for x in hand_qpos],
                dtype=np.float64,
            )

            err = q_target6 - q_now6

            msg = (
                f"sim={sim_time:7.3f}s "
                f"traj={traj_time:7.3f}s "
                f"mean|hand err|={np.mean(np.abs(err)):.5f} rad "
                f"max|hand err|={np.max(np.abs(err)):.5f} rad"
            )

            if arm_q is not None:
                q_arm_now = np.asarray(
                    [data.qpos[x] for x in arm_qpos],
                    dtype=np.float64,
                )

                arm_err = (
                    q_arm_target
                    -
                    q_arm_now
                )

                msg += (
                    f" mean|arm err|={np.mean(np.abs(arm_err)):.5f} rad"
                    f" max|arm err|={np.max(np.abs(arm_err)):.5f} rad"
                )

            print(msg)

finally:
    writer.release()
    renderer.close()


print()
print("==============================================")
print("DONE")
print("==============================================")
print("Physics steps:", physics_steps)
print("Rendered frames:", render_frames)
print("Saved:", args.output_mp4)
