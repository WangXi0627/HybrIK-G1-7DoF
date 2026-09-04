import mujoco
import numpy as np

XML_PATH = "/data/wx/code-IK/unitree_ros/robots/g1_description/g1_29dof.xml"
NPZ_PATH = "res/pick_place/g1_left_arm_7dof_flip_forward.npz"

CHECK_FRAMES = [0, 20, 40, 60]

traj = np.load(NPZ_PATH, allow_pickle=True)
q = np.asarray(traj["q"], dtype=np.float64)
joint_names = [str(x) for x in traj["joint_names"]]

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

qpos_adrs = []
for name in joint_names:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    qpos_adrs.append(model.jnt_qposadr[jid])

wrist_bid = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "left_wrist_yaw_link",   # 也可以改成 left_wrist_roll_link / hand body
)

print("Check wrist x position")
for f in CHECK_FRAMES:
    data.qpos[:] = model.qpos0.copy()
    for adr, val in zip(qpos_adrs, q[f]):
        data.qpos[adr] = val
    mujoco.mj_forward(model, data)

    wrist = data.xpos[wrist_bid].copy()
    print(f"frame {f:3d}: wrist xyz = {wrist}")