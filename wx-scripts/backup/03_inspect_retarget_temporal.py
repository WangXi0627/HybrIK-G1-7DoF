import numpy as np


PATH = "res/pick_place/retarget_input.npz"

d = np.load(PATH)


def angle_between_frames(x):
    dots = np.sum(x[1:] * x[:-1], axis=1)
    dots = np.clip(dots, -1, 1)
    return np.degrees(np.arccos(dots))


for key in [
    "left_upper_arm_dir",
    "left_forearm_dir",
]:
    a = angle_between_frames(d[key])

    print(key)
    print("  mean step:", a.mean())
    print("  max step :", a.max())
    print("  >10 deg  :", np.sum(a > 10))
    print("  >30 deg  :", np.sum(a > 30))
    print("  >60 deg  :", np.sum(a > 60))