#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import mujoco
import numpy as np


ACTIVE = {
    "thumb_yaw": ("right_thumb_proximal_yaw_joint", 1.308),
    "thumb_pitch": ("right_thumb_proximal_pitch_joint", 0.6),
    "index": ("right_index_proximal_joint", 1.47),
    "middle": ("right_middle_proximal_joint", 1.47),
    "ring": ("right_ring_proximal_joint", 1.47),
    "pinky": ("right_pinky_proximal_joint", 1.47),
}

MIMIC = [
    ("right_thumb_intermediate_joint", "right_thumb_proximal_pitch_joint", 1.334, 0.0),
    ("right_thumb_distal_joint", "right_thumb_proximal_pitch_joint", 0.667, 0.0),
    ("right_index_intermediate_joint", "right_index_proximal_joint", 1.06399, -0.04545),
    ("right_middle_intermediate_joint", "right_middle_proximal_joint", 1.06399, -0.04545),
    ("right_ring_intermediate_joint", "right_ring_proximal_joint", 1.06399, -0.04545),
    ("right_pinky_intermediate_joint", "right_pinky_proximal_joint", 1.06399, -0.04545),
]


def joint_info(model, name):
    jid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name,
    )
    if jid < 0:
        raise RuntimeError(
            "Joint not found: " + name
        )

    qadr = model.jnt_qposadr[jid]
    lo = model.jnt_range[jid, 0]
    hi = model.jnt_range[jid, 1]
    return jid, qadr, lo, hi


def set_joint_qpos(model, data, name, value):
    _, qadr, lo, hi = joint_info(
        model,
        name,
    )
    data.qpos[qadr] = np.clip(
        value,
        lo,
        hi,
    )


def get_joint_qpos(model, data, name):
    _, qadr, _, _ = joint_info(
        model,
        name,
    )
    return float(
        data.qpos[qadr]
    )


def apply_hand_pose(model, data, close_fraction, amplitude):
    f = np.clip(
        close_fraction * amplitude,
        0.0,
        1.0,
    )

    # Independent six-channel control.
    set_joint_qpos(
        model,
        data,
        ACTIVE["thumb_yaw"][0],
        ACTIVE["thumb_yaw"][1] * f,
    )
    set_joint_qpos(
        model,
        data,
        ACTIVE["thumb_pitch"][0],
        ACTIVE["thumb_pitch"][1] * f,
    )
    set_joint_qpos(
        model,
        data,
        ACTIVE["index"][0],
        ACTIVE["index"][1] * f,
    )
    set_joint_qpos(
        model,
        data,
        ACTIVE["middle"][0],
        ACTIVE["middle"][1] * f,
    )
    set_joint_qpos(
        model,
        data,
        ACTIVE["ring"][0],
        ACTIVE["ring"][1] * f,
    )
    set_joint_qpos(
        model,
        data,
        ACTIVE["pinky"][0],
        ACTIVE["pinky"][1] * f,
    )

    # Apply the exact URDF mimic mapping explicitly for kinematic render.
    for dep, active, multiplier, offset in MIMIC:
        q_active = get_joint_qpos(
            model,
            data,
            active,
        )
        q_dep = (
            offset
            +
            multiplier * q_active
        )
        set_joint_qpos(
            model,
            data,
            dep,
            q_dep,
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Kinematic MuJoCo render of the full G1 with the attached "
            "RH56DFTP-2L Inspire right hand opening and closing."
        )
    )
    parser.add_argument(
        "--xml-path",
        default=(
            "/data/wx/code-IK/unitree_ros/robots/"
            "g1_description/g1_29dof_rh56dftp_right.xml"
        ),
    )
    parser.add_argument(
        "--output-mp4",
        default=(
            "/data/wx/code-IK/HybrIK/res/pick_place/"
            "g1_rh56dftp_right_hand_test.mp4"
        ),
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.8,
        help="Fraction of each active hand joint range used in the test.",
    )
    parser.add_argument(
        "--cam-lookat",
        type=float,
        nargs=3,
        default=[0.0, 0.15, 0.85],
    )
    parser.add_argument(
        "--cam-distance",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--cam-azimuth",
        type=float,
        default=230.0,
    )
    parser.add_argument(
        "--cam-elevation",
        type=float,
        default=-8.0,
    )
    args = parser.parse_args()

    xml_path = Path(args.xml_path).resolve()
    output_path = Path(args.output_mp4).resolve()
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = mujoco.MjModel.from_xml_path(
        str(xml_path)
    )
    data = mujoco.MjData(
        model
    )

    # Hold the whole G1 at its model qpos0. We are checking geometry,
    # mounting and finger kinematics here, not full-body dynamics.
    data.qpos[:] = model.qpos0.copy()

    num_frames = int(
        round(
            args.seconds
            *
            args.fps
        )
    )

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
    camera.lookat[:] = np.asarray(
        args.cam_lookat,
        dtype=np.float64,
    )
    camera.distance = args.cam_distance
    camera.azimuth = args.cam_azimuth
    camera.elevation = args.cam_elevation

    writer = cv2.VideoWriter(
        str(output_path),
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
            "Failed to open MP4 writer: "
            + str(output_path)
        )

    try:
        for frame_idx in range(
            num_frames
        ):
            phase = (
                2.0
                *
                np.pi
                *
                frame_idx
                /
                max(
                    num_frames - 1,
                    1,
                )
            )

            # 0 -> 1 -> 0 over one cycle.
            close_fraction = (
                0.5
                *
                (
                    1.0
                    -
                    np.cos(
                        phase
                    )
                )
            )

            data.qpos[:] = (
                model.qpos0.copy()
            )

            apply_hand_pose(
                model,
                data,
                close_fraction,
                args.amplitude,
            )

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

            cv2.putText(
                bgr,
                "frame={} close={:.2f}".format(
                    frame_idx,
                    close_fraction,
                ),
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            writer.write(
                bgr
            )

    finally:
        writer.release()
        renderer.close()

    print("Saved:", output_path)
    print("Frames:", num_frames)
    print(
        "This was a kinematic test: G1 body stayed at qpos0 while "
        "the right Inspire hand opened/closed."
    )


if __name__ == "__main__":
    main()
