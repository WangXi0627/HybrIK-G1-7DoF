#!/usr/bin/env python3
"""
Attach Inspire RH56DFTP-2L RIGHT hand to Unitree G1 MuJoCo model.

Source hand model:
    EuncheolIm/Inspire_hand_description
    Inspire_hand/xml/inspire_hand_right.xml

What this script does:
1. Load the original G1 g1_29dof.xml.
2. Load the RH56DFTP-2L right-hand MJCF directly.
3. Convert all hand mesh file paths to absolute paths.
4. Remove the stock G1 right_rubber_hand visual geom.
5. Restore right_wrist_yaw_link to wrist-only inertia.
6. Remove the standalone-scene offset from right_hand_root:
       pos="0 0.1 0" -> pos="0 0 0"
7. Merge hand <asset>, <equality>, <actuator>, and hand body tree into G1.
8. Mount the hand below right_wrist_yaw_link through a tunable fixed body.
9. Compile the generated combined XML immediately for validation.

Default mount:
    position = [0.0415, -0.003, 0]
    euler    = [0, pi/2, 0]

The default rotation maps the RH56DFTP hand's longitudinal +Z direction
to the G1 wrist's distal +X direction.  If the rendered hardware mounting
needs small correction, tune --mount-pos and --mount-euler only.
"""

import argparse
import copy
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco


# ============================================================
# G1 right wrist-only inertia
#
# The stock g1_29dof.xml may fold the fixed rubber-hand mass
# into right_wrist_yaw_link.  When replacing that hand with the
# RH56DFTP model, restore the wrist-only inertial parameters.
#
# MuJoCo fullinertia order:
#   ixx iyy izz ixy ixz iyz
# ============================================================

RIGHT_WRIST_ONLY_INERTIAL = {
    "pos": "0.02200381568 -0.00049485096 0.00053861123",
    "mass": "0.08457647",
    "fullinertia": (
        "0.00004929128828 "
        "0.00005973338134 "
        "0.00003928083826 "
        "0.00000045735494 "
        "0.00000445867591 "
        "-0.00000043217198"
    ),
}


# ============================================================
# Expected RH56DFTP-2L right hand interface
# ============================================================

EXPECTED_HAND_ACTUATORS = [
    "right_thumb_yaw_act",
    "right_thumb_pitch_act",
    "right_index_act",
    "right_middle_act",
    "right_ring_act",
    "right_pinky_act",
]

EXPECTED_HAND_JOINTS = [
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


# ============================================================
# XML helpers
# ============================================================

def pretty_indent(elem, level=0):
    """
    Pretty-print ElementTree XML for easier manual inspection.
    """
    indent = "\n" + "  " * level

    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "

        for child in elem:
            pretty_indent(child, level + 1)

        if not child.tail or not child.tail.strip():
            child.tail = indent

    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = indent


def get_or_create(root, tag):
    elem = root.find(tag)

    if elem is None:
        elem = ET.SubElement(root, tag)

    return elem


def find_body_by_name(worldbody, name):
    for body in worldbody.iter("body"):
        if body.get("name") == name:
            return body

    return None


def collect_named_children(section):
    """
    Return {(tag, name): element} for named direct children.
    """
    result = {}

    if section is None:
        return result

    for child in section:
        name = child.get("name")

        if name:
            result[(child.tag, name)] = child

    return result


def resolve_asset_path(raw_file, hand_xml_dir):
    """
    Resolve a hand MJCF asset path exactly as it is meant relative
    to the hand XML directory.

    Example:
        ../visual/foo.obj
    with hand XML at:
        .../Inspire_hand/xml/inspire_hand_right.xml
    becomes:
        .../Inspire_hand/visual/foo.obj
    """
    raw_path = Path(raw_file)

    if raw_path.is_absolute():
        if raw_path.exists():
            return raw_path.resolve()

        return None

    candidate = (hand_xml_dir / raw_path).resolve()

    if candidate.exists():
        return candidate

    return None


def absolutize_hand_assets(hand_root, hand_xml_path):
    """
    Convert every <asset ... file="..."> path in the hand MJCF
    to an absolute path before merging it into the G1 XML.

    This prevents the original ../visual and ../collision paths
    from being interpreted relative to the G1 XML directory.
    """
    hand_asset = hand_root.find("asset")

    if hand_asset is None:
        raise RuntimeError(
            "RH56DFTP hand MJCF has no <asset> section."
        )

    hand_xml_dir = hand_xml_path.parent

    converted = 0

    for elem in hand_asset.iter():
        raw_file = elem.get("file")

        if not raw_file:
            continue

        resolved = resolve_asset_path(
            raw_file,
            hand_xml_dir,
        )

        if resolved is None:
            raise FileNotFoundError(
                "Cannot resolve RH56DFTP asset:\n"
                f"  XML file : {hand_xml_path}\n"
                f"  raw path : {raw_file}\n"
                f"  tried    : {(hand_xml_dir / raw_file).resolve()}"
            )

        elem.set(
            "file",
            str(resolved),
        )

        converted += 1

    return converted


def merge_named_section(g1_root, hand_root, tag):
    """
    Merge direct named children of a hand top-level section
    into the corresponding G1 section.

    Used for:
        asset
        equality
        actuator
    """
    hand_section = hand_root.find(tag)

    if hand_section is None:
        return 0

    g1_section = get_or_create(
        g1_root,
        tag,
    )

    existing = collect_named_children(
        g1_section
    )

    copied = 0

    for child in hand_section:
        child_name = child.get("name")
        key = (
            child.tag,
            child_name,
        )

        if child_name and key in existing:
            raise RuntimeError(
                f"Name collision while merging <{tag}>: "
                f"{child.tag} name={child_name}"
            )

        g1_section.append(
            copy.deepcopy(child)
        )

        copied += 1

    return copied


# ============================================================
# G1 cleanup helpers
# ============================================================

def remove_stock_right_rubber_hand(wrist_body):
    """
    Remove the visual mesh of the stock Unitree rubber hand.

    The Inspire model will replace it.
    """
    removed = 0

    for child in list(wrist_body):
        if child.tag != "geom":
            continue

        mesh_name = child.get("mesh")
        geom_name = child.get("name")

        if (
            mesh_name == "right_rubber_hand"
            or geom_name == "right_rubber_hand"
        ):
            wrist_body.remove(child)
            removed += 1

    return removed


def restore_wrist_only_inertial(wrist_body):
    """
    Restore right_wrist_yaw_link's inertial parameters so the
    removed stock rubber-hand mass is not counted together with
    the new Inspire hand.
    """
    inertial = wrist_body.find("inertial")

    if inertial is None:
        inertial = ET.Element("inertial")
        wrist_body.insert(
            0,
            inertial,
        )

    # Remove alternative inertia forms that cannot coexist with
    # fullinertia in MuJoCo.
    for attr in [
        "quat",
        "diaginertia",
        "fullinertia",
    ]:
        inertial.attrib.pop(
            attr,
            None,
        )

    inertial.set(
        "pos",
        RIGHT_WRIST_ONLY_INERTIAL["pos"],
    )

    inertial.set(
        "mass",
        RIGHT_WRIST_ONLY_INERTIAL["mass"],
    )

    inertial.set(
        "fullinertia",
        RIGHT_WRIST_ONLY_INERTIAL["fullinertia"],
    )


# ============================================================
# Hand helpers
# ============================================================

def get_single_hand_root_body(hand_root):
    """
    RH56DFTP right-hand MJCF contains one top-level hand body:
        right_hand_root

    Return a deep copy, then the caller can remove the standalone
    scene translation without modifying the source XML.
    """
    worldbody = hand_root.find("worldbody")

    if worldbody is None:
        raise RuntimeError(
            "RH56DFTP hand MJCF has no <worldbody>."
        )

    bodies = list(
        worldbody.findall("body")
    )

    if not bodies:
        raise RuntimeError(
            "RH56DFTP hand MJCF has no top-level body."
        )

    target = None

    for body in bodies:
        if body.get("name") == "right_hand_root":
            target = body
            break

    if target is None:
        if len(bodies) == 1:
            target = bodies[0]
        else:
            names = [
                b.get("name")
                for b in bodies
            ]
            raise RuntimeError(
                "Could not uniquely identify RH56DFTP right hand root. "
                f"Top-level bodies: {names}"
            )

    return copy.deepcopy(
        target
    )



def ensure_hand_materials(hand_root):
    """
    The upstream RH56DFTP right-hand XML references:
        hand_white / hand_dark / hand_gray
    but currently does not define these materials in its <asset> section.
    The left-hand XML in the same repository does define them.

    Inject the same three material definitions when missing.
    """
    asset = hand_root.find("asset")

    if asset is None:
        asset = ET.SubElement(
            hand_root,
            "asset",
        )

    existing = {
        elem.get("name")
        for elem in asset.findall("material")
        if elem.get("name")
    }

    definitions = {
        "hand_white": {
            "rgba": "0.9 0.9 0.9 1",
            "specular": "0.5",
            "shininess": "0.3",
        },
        "hand_dark": {
            "rgba": "0.15 0.15 0.15 1",
            "specular": "0.3",
            "shininess": "0.5",
        },
        "hand_gray": {
            "rgba": "0.5 0.5 0.5 1",
            "specular": "0.4",
            "shininess": "0.4",
        },
    }

    added = []

    for name, attrs in definitions.items():
        if name in existing:
            continue

        elem = ET.Element(
            "material",
            {
                "name": name,
                **attrs,
            },
        )

        asset.append(
            elem
        )

        added.append(
            name
        )

    return added


def validate_source_hand_model(hand_xml_path):
    """
    Validate the RH56DFTP right-hand MJCF.

    NOTE:
    The current upstream right-hand XML references hand_white /
    hand_dark / hand_gray but omits their definitions.  Therefore
    validate a repaired in-memory copy instead of compiling the raw
    upstream XML directly.
    """
    hand_tree = ET.parse(
        str(hand_xml_path)
    )

    hand_root = hand_tree.getroot()

    added_materials = ensure_hand_materials(
        hand_root
    )

    absolutize_hand_assets(
        hand_root,
        hand_xml_path,
    )

    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".xml",
        delete=False,
        encoding="utf-8",
    ) as f:
        temp_path = Path(
            f.name
        )

        hand_tree.write(
            f,
            encoding="unicode",
            xml_declaration=False,
        )

    try:
        model = mujoco.MjModel.from_xml_path(
            str(temp_path)
        )
    finally:
        try:
            temp_path.unlink()
        except OSError:
            pass

    if added_materials:
        print(
            "Injected missing source materials:",
            added_materials,
        )

    print()
    print(
        "=========================================="
    )
    print(
        "Source RH56DFTP right-hand model"
    )
    print(
        "=========================================="
    )
    print("nq   =", model.nq)
    print("nv   =", model.nv)
    print("njnt =", model.njnt)
    print("nu   =", model.nu)
    print("neq  =", model.neq)
    print("nbody=", model.nbody)

    missing_actuators = []

    for name in EXPECTED_HAND_ACTUATORS:
        aid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            name,
        )

        if aid < 0:
            missing_actuators.append(
                name
            )

    if missing_actuators:
        raise RuntimeError(
            "Source RH56DFTP model is missing expected actuators:\n  "
            + "\n  ".join(
                missing_actuators
            )
        )

    missing_joints = []

    for name in EXPECTED_HAND_JOINTS:
        jid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        if jid < 0:
            missing_joints.append(
                name
            )

    if missing_joints:
        raise RuntimeError(
            "Source RH56DFTP model is missing expected joints:\n  "
            + "\n  ".join(
                missing_joints
            )
        )

    if model.nu != 6:
        raise RuntimeError(
            f"Expected 6 RH56DFTP actuators, got {model.nu}."
        )

    if model.neq != 6:
        raise RuntimeError(
            f"Expected 6 RH56DFTP equality constraints, got {model.neq}."
        )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Attach RH56DFTP-2L Inspire RIGHT hand MJCF "
            "to Unitree G1 right_wrist_yaw_link."
        )
    )

    parser.add_argument(
        "--g1-xml",
        default=(
            "/data/wx/code-IK/unitree_ros/"
            "robots/g1_description/g1_29dof.xml"
        ),
        help="Original Unitree G1 MuJoCo XML.",
    )

    parser.add_argument(
        "--hand-xml",
        default=(
            "/data/wx/code-IK/"
            "Inspire_hand_description-master/"
            "Inspire_hand/xml/inspire_hand_right.xml"
        ),
        help="RH56DFTP-2L RIGHT hand MuJoCo XML.",
    )

    parser.add_argument(
        "--output",
        default=(
            "/data/wx/code-IK/unitree_ros/"
            "robots/g1_description/"
            "g1_29dof_rh56dftp_right.xml"
        ),
        help="Combined output MJCF.",
    )

    parser.add_argument(
        "--wrist-body",
        default="right_wrist_yaw_link",
        help="G1 body under which the hand is mounted.",
    )

    # --------------------------------------------------------
    # Initial mounting transform.
    #
    # RH56DFTP hand geometry extends predominantly along +Z
    # in its own root frame.
    #
    # G1 right wrist distal direction is +X.
    #
    # Ry(+pi/2):
    #     +Z_hand -> +X_G1
    #
    # Translation starts from the original G1 rubber-hand
    # attachment location.
    # --------------------------------------------------------

    parser.add_argument(
        "--mount-pos",
        type=float,
        nargs=3,
        default=[
            0.0415,
            -0.003,
            0.0,
        ],
        metavar=(
            "X",
            "Y",
            "Z",
        ),
        help=(
            "RH56DFTP mount translation in "
            "right_wrist_yaw_link coordinates."
        ),
    )

    parser.add_argument(
        "--mount-euler",
        type=float,
        nargs=3,
        default=[
            0.0,
            1.5707963267948966,
            0.0,
        ],
        metavar=(
            "RX",
            "RY",
            "RZ",
        ),
        help=(
            "RH56DFTP mount Euler rotation in radians. "
            "Default maps hand +Z to G1 +X."
        ),
    )

    parser.add_argument(
        "--keep-rubber-hand",
        action="store_true",
        help=(
            "Keep the stock G1 right_rubber_hand geom. "
            "Normally leave this OFF."
        ),
    )

    parser.add_argument(
        "--keep-combined-wrist-inertia",
        action="store_true",
        help=(
            "Keep the stock g1_29dof.xml wrist inertia. "
            "Normally leave this OFF when replacing the rubber hand."
        ),
    )

    parser.add_argument(
        "--keep-source-root-offset",
        action="store_true",
        help=(
            "Keep right_hand_root's source pos='0 0.1 0'. "
            "Normally leave this OFF; that offset belongs to "
            "the standalone hand scene, not the robot mounting."
        ),
    )

    args = parser.parse_args()

    g1_xml_path = Path(
        args.g1_xml
    ).resolve()

    hand_xml_path = Path(
        args.hand_xml
    ).resolve()

    output_path = Path(
        args.output
    ).resolve()

    if not g1_xml_path.exists():
        raise FileNotFoundError(
            f"G1 XML not found:\n  {g1_xml_path}"
        )

    if not hand_xml_path.exists():
        raise FileNotFoundError(
            f"RH56DFTP hand XML not found:\n  {hand_xml_path}"
        )

    # Keep combined XML next to original G1 XML so all G1
    # relative paths remain valid.
    if output_path.parent != g1_xml_path.parent:
        raise RuntimeError(
            "--output must be in the same directory as --g1-xml.\n"
            f"G1 directory    : {g1_xml_path.parent}\n"
            f"Output directory: {output_path.parent}"
        )

    # --------------------------------------------------------
    # Validate source hand first.
    # --------------------------------------------------------

    validate_source_hand_model(
        hand_xml_path
    )

    # --------------------------------------------------------
    # Parse source XMLs.
    # --------------------------------------------------------

    g1_tree = ET.parse(
        str(g1_xml_path)
    )
    g1_root = g1_tree.getroot()

    hand_tree = ET.parse(
        str(hand_xml_path)
    )
    hand_root = hand_tree.getroot()

    # The upstream RIGHT-hand XML references hand_white/hand_dark/
    # hand_gray but currently omits their <material> definitions.
    # Inject the definitions used by the LEFT-hand XML before merging.
    added_materials = ensure_hand_materials(
        hand_root
    )

    if added_materials:
        print(
            "Injected missing RH56DFTP materials:",
            added_materials,
        )

    # Hand asset paths such as ../visual/... would become invalid
    # after writing the combined XML in G1's directory.
    converted_assets = absolutize_hand_assets(
        hand_root,
        hand_xml_path,
    )

    print()
    print(
        "Absolute RH56DFTP asset paths:",
        converted_assets,
    )

    # --------------------------------------------------------
    # Locate G1 right wrist.
    # --------------------------------------------------------

    g1_worldbody = g1_root.find(
        "worldbody"
    )

    if g1_worldbody is None:
        raise RuntimeError(
            "G1 XML has no <worldbody>."
        )

    wrist_body = find_body_by_name(
        g1_worldbody,
        args.wrist_body,
    )

    if wrist_body is None:
        raise RuntimeError(
            f"Cannot find G1 wrist body: {args.wrist_body}"
        )

    # --------------------------------------------------------
    # Remove stock rubber hand and correct inertia.
    # --------------------------------------------------------

    if not args.keep_rubber_hand:
        removed = remove_stock_right_rubber_hand(
            wrist_body
        )

        print(
            "Removed stock right_rubber_hand geoms:",
            removed,
        )

    else:
        print(
            "Keeping stock right_rubber_hand geom."
        )

    if not args.keep_combined_wrist_inertia:
        restore_wrist_only_inertial(
            wrist_body
        )

        print(
            "Restored right_wrist_yaw_link wrist-only inertia."
        )

    else:
        print(
            "Keeping original combined wrist inertia."
        )

    # --------------------------------------------------------
    # Merge hand assets, mimic equality constraints and motors.
    # --------------------------------------------------------

    num_assets = merge_named_section(
        g1_root,
        hand_root,
        "asset",
    )

    num_equalities = merge_named_section(
        g1_root,
        hand_root,
        "equality",
    )

    num_actuators = merge_named_section(
        g1_root,
        hand_root,
        "actuator",
    )

    print(
        "Merged hand asset entries:",
        num_assets,
    )
    print(
        "Merged hand equality entries:",
        num_equalities,
    )
    print(
        "Merged hand actuator entries:",
        num_actuators,
    )

    # --------------------------------------------------------
    # Prepare the hand body tree.
    # --------------------------------------------------------

    hand_body = get_single_hand_root_body(
        hand_root
    )

    source_root_pos = hand_body.get(
        "pos",
        "0 0 0",
    )

    if not args.keep_source_root_offset:
        hand_body.set(
            "pos",
            "0 0 0",
        )

    print()
    print(
        "RH56DFTP source right_hand_root pos:",
        source_root_pos,
    )
    print(
        "RH56DFTP attached right_hand_root pos:",
        hand_body.get(
            "pos",
            "0 0 0",
        ),
    )

    # --------------------------------------------------------
    # Create tunable fixed mount.
    # --------------------------------------------------------

    mount_body = ET.Element(
        "body",
        {
            "name": "right_rh56dftp_mount",
            "pos": "{} {} {}".format(
                args.mount_pos[0],
                args.mount_pos[1],
                args.mount_pos[2],
            ),
            "euler": "{} {} {}".format(
                args.mount_euler[0],
                args.mount_euler[1],
                args.mount_euler[2],
            ),
        },
    )

    mount_body.append(
        hand_body
    )

    wrist_body.append(
        mount_body
    )

    g1_root.set(
        "model",
        "g1_29dof_rh56dftp_right",
    )

    # --------------------------------------------------------
    # Write combined XML.
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pretty_indent(
        g1_root
    )

    g1_tree.write(
        str(output_path),
        encoding="utf-8",
        xml_declaration=True,
    )

    print()
    print(
        "Combined XML written:"
    )
    print(
        " ",
        output_path
    )

    # --------------------------------------------------------
    # Compile the final model immediately.
    # --------------------------------------------------------

    model = mujoco.MjModel.from_xml_path(
        str(output_path)
    )

    print()
    print(
        "=========================================="
    )
    print(
        "Combined G1 + RH56DFTP RIGHT"
    )
    print(
        "=========================================="
    )
    print("nq   =", model.nq)
    print("nv   =", model.nv)
    print("njnt =", model.njnt)
    print("nu   =", model.nu)
    print("neq  =", model.neq)
    print("nbody=", model.nbody)

    print()
    print(
        "Mount:"
    )
    print(
        "  body  =",
        args.wrist_body,
    )
    print(
        "  pos   =",
        args.mount_pos,
    )
    print(
        "  euler =",
        args.mount_euler,
    )

    # --------------------------------------------------------
    # Validate hand joints/actuators in combined model.
    # --------------------------------------------------------

    print()
    print(
        "RH56DFTP hand actuators:"
    )

    found_actuators = 0

    for name in EXPECTED_HAND_ACTUATORS:
        aid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            name,
        )

        print(
            f"  {name:24s} -> id={aid}"
        )

        if aid >= 0:
            found_actuators += 1

    if found_actuators != 6:
        raise RuntimeError(
            "Expected 6 RH56DFTP hand actuators in the combined "
            f"model, got {found_actuators}."
        )

    print()
    print(
        "RH56DFTP hand joints:"
    )

    found_joints = 0

    for name in EXPECTED_HAND_JOINTS:
        jid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        if jid >= 0:
            found_joints += 1

        print(
            f"  {name:36s} -> id={jid}"
        )

    if found_joints != 12:
        raise RuntimeError(
            "Expected 12 RH56DFTP hand joints in the combined "
            f"model, got {found_joints}."
        )

    mount_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "right_rh56dftp_mount",
    )

    hand_root_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "right_hand_root",
    )

    if mount_id < 0:
        raise RuntimeError(
            "right_rh56dftp_mount is missing after compilation."
        )

    if hand_root_id < 0:
        raise RuntimeError(
            "right_hand_root is missing after compilation."
        )

    print()
    print(
        "PASS: G1 + RH56DFTP right-hand model compiles successfully."
    )
    print()
    print(
        "Next step:"
    )
    print(
        "  Render the combined model before changing any mount values."
    )
    print(
        "  If orientation is correct but translation is slightly off,"
    )
    print(
        "  tune only --mount-pos first."
    )


if __name__ == "__main__":
    main()
