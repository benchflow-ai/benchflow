"""Compose the Metal arm's kinematic chain for the simulator and the web viewer.

Joints 1-6 come from the vendor's canonical kinematics URDF (``metal_sdk/example/urdf``); the
gripper mount and the two prismatic finger joints come from the ``metal_description`` URDF, which
is only used for visualisation. Names are lower-cased (``JOINT1`` -> ``joint1``, ``Link1`` ->
``link1``) so mesh file names and link names agree.

Usage: python metal-sim/scripts/urdf_to_chain.py sim-assets/metal_with_gripper.canonical.urdf \
           sim-assets/metal_with_gripper.description.urdf \
           metal-sim/src/inspect_robots_metal_sim/data/metal_chain.json
"""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET


def _joint(j: ET.Element) -> dict:
    origin, axis, limit = j.find("origin"), j.find("axis"), j.find("limit")
    return {
        "name": j.get("name").lower(),
        "type": j.get("type"),
        "parent": j.find("parent").get("link").lower(),
        "child": j.find("child").get("link").lower(),
        "xyz": [float(v) for v in (origin.get("xyz") if origin is not None else "0 0 0").split()],
        "rpy": [float(v) for v in (origin.get("rpy") if origin is not None else "0 0 0").split()],
        "axis": [float(v) for v in axis.get("xyz").split()] if axis is not None and j.get("type") != "fixed" else None,
        "limit": {"lower": float(limit.get("lower")), "upper": float(limit.get("upper"))} if limit is not None else None,
    }


def _link(link: ET.Element) -> dict:
    mesh = link.find("visual/geometry/mesh")
    return {"name": link.get("name").lower(), "mesh": (mesh.get("filename").split("/")[-1].rsplit(".", 1)[0].lower() + ".STL") if mesh is not None else None}


def main(canonical: str, description: str, out_path: str) -> None:
    arm = ET.parse(canonical).getroot()
    desc = ET.parse(description).getroot()
    joints = [_joint(j) for j in arm.findall("joint")]
    links = [_link(link) for link in arm.findall("link")]
    arm_links = {link["name"] for link in links}
    # Gripper mount and fingers (visualisation only) from the description URDF.
    for j in desc.findall("joint"):
        entry = _joint(j)
        if entry["name"] in ("gripper_base", "joint7", "joint8"):
            joints.append(entry)
    for link in desc.findall("link"):
        entry = _link(link)
        if entry["name"] not in arm_links and entry["name"] in ("gripper_base", "link7", "link8"):
            links.append(entry)
    chain = {
        "robot": arm.get("name"),
        "sources": {"arm": canonical.split("/")[-1], "gripper": description.split("/")[-1]},
        "joint_mapping": "identity: motor degrees -> radians on joint1..joint6, no offsets",
        "joints": joints,
        "links": links,
    }
    with open(out_path, "w") as fh:
        json.dump(chain, fh, indent=1)
    print(f"wrote {out_path}: {len(joints)} joints, {len(links)} links")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
