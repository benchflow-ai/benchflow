"""Camera calibration helpers (numpy; MuJoCo only when the MuJoCo helper is called)."""

from __future__ import annotations

import math
from typing import Any


def pinhole(width: int, height: int, fovy_deg: float) -> dict[str, float]:
    """Intrinsics of a pinhole camera with square pixels and the principal point at the image centre."""
    f = (height / 2) / math.tan(math.radians(fovy_deg) / 2)
    return {"fx": f, "fy": f, "cx": width / 2, "cy": height / 2, "fovy_deg": fovy_deg}


def mujoco_camera(
    model: Any,
    data: Any,
    name: str,
    width: int,
    height: int,
    *,
    image_flipped: bool = False,
    calibrated: bool = True,
) -> dict | None:
    """Calibration of a MuJoCo camera for images of `width` x `height` pixels.

    ``projection`` is a 3x4 matrix P with [u*w, v*w, w] = P @ [x, y, z, 1] in the saved image's pixels (u right,
    v down). ``image_flipped``: the backend's render() flips the image vertically. None when there is no such camera.
    """
    import mujoco
    import numpy as np

    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    if cid < 0:
        return None
    fovy = float(model.cam_fovy[cid])
    out: dict[str, Any] = {
        "name": name,
        "width": width,
        "height": height,
        "fovy_deg": round(fovy, 2),
    }
    if not calibrated:
        out["projection"] = None
        out["calibrated"] = False
        return out
    f = (height / 2) / np.tan(np.deg2rad(fovy) / 2)
    R = data.cam_xmat[cid].reshape(3, 3)
    t = data.cam_xpos[cid]
    ext = np.hstack([R.T, (-R.T @ t)[:, None]])
    K = np.array([[f, 0, -width / 2], [0, -f, -height / 2], [0, 0, -1.0]])
    P = K @ ext
    if image_flipped:
        P[1] = (height - 1) * P[2] - P[1]
    out.update(
        {
            "calibrated": True,
            "intrinsics": {
                k: round(float(v), 4) for k, v in pinhole(width, height, fovy).items()
            },
            "position": [round(float(x), 4) for x in t],
            "rotation": [[round(float(x), 6) for x in row] for row in R],
            "projection": [[round(float(x), 5) for x in row] for row in P],
        }
    )
    return out
