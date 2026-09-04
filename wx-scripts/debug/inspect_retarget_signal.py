import numpy as np


PATH = "res/pick_place/retarget_input.npz"

d = np.load(PATH)

left_angle = d["left_elbow_angle"]
right_angle = d["right_elbow_angle"]

# 几何夹角 -> 屈肘量
left_flexion = np.pi - left_angle
right_flexion = np.pi - right_angle


print("========== Left elbow ==========")

print(
    "geometric angle:",
    np.degrees(left_angle.min()),
    "~",
    np.degrees(left_angle.max()),
)

print(
    "flexion:",
    np.degrees(left_flexion.min()),
    "~",
    np.degrees(left_flexion.max()),
)


print()
print("========== Right elbow ==========")

print(
    "geometric angle:",
    np.degrees(right_angle.min()),
    "~",
    np.degrees(right_angle.max()),
)

print(
    "flexion:",
    np.degrees(right_flexion.min()),
    "~",
    np.degrees(right_flexion.max()),
)


print()
print("========== Frame 0 ==========")

print(
    "left flexion:",
    np.degrees(left_flexion[0]),
)

print(
    "right flexion:",
    np.degrees(right_flexion[0]),
)