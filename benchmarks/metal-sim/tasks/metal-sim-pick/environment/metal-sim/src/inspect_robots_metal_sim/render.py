"""A tiny numpy software renderer for the simulator's synthetic ``front`` camera.

Draws the ground grid, the cube, and the arm as a thick polyline through its joint origins with a
pinhole projection. No third-party graphics dependencies so the embodiment runs anywhere.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Vec = npt.NDArray[np.float64]
Img = npt.NDArray[np.uint8]

SKY = np.array([236, 240, 246], dtype=np.uint8)
GROUND = np.array([214, 220, 228], dtype=np.uint8)
GRID = np.array([176, 184, 196], dtype=np.uint8)
ARM = np.array([38, 62, 120], dtype=np.uint8)
JOINT = np.array([242, 150, 40], dtype=np.uint8)
CUBE = np.array([214, 48, 48], dtype=np.uint8)
CUBE_HELD = np.array([150, 40, 170], dtype=np.uint8)
TIP = np.array([255, 255, 255], dtype=np.uint8)
TIP_OK = np.array([40, 200, 90], dtype=np.uint8)


class FrontCamera:
    """Pinhole camera looking at the workspace from the front-right, slightly above."""

    def __init__(
        self,
        width: int,
        height: int,
        *,
        eye: tuple[float, float, float] = (1.05, -0.65, 0.55),
        look_at: tuple[float, float, float] = (0.12, 0.0, 0.18),
        fov_deg: float = 48.0,
    ) -> None:
        self.w, self.h = int(width), int(height)
        self.eye = np.asarray(eye, dtype=np.float64)
        target = np.asarray(look_at, dtype=np.float64)
        fwd = target - self.eye
        fwd /= np.linalg.norm(fwd)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(fwd, up)
        right /= np.linalg.norm(right)
        cam_up = np.cross(right, fwd)
        self._rot = np.stack([right, cam_up, fwd])  # rows: camera x, y, z (forward)
        self._f = 0.5 * self.w / np.tan(np.deg2rad(fov_deg) / 2)

    def project(self, pts: Vec) -> tuple[Vec, Vec]:
        """World points (N, 3) to pixel coordinates (N, 2) and depths (N,)."""
        rel = (np.atleast_2d(pts) - self.eye) @ self._rot.T
        depth = np.maximum(rel[:, 2], 1e-3)
        u = self.w / 2 + self._f * rel[:, 0] / depth
        v = self.h / 2 - self._f * rel[:, 1] / depth
        return np.stack([u, v], axis=1), depth

    # -- primitives -----------------------------------------------------------

    def _segment(self, img: Img, a: Vec, b: Vec, radius: float, color: np.ndarray) -> None:
        x0, y0 = a
        x1, y1 = b
        r = radius + 0.5
        lo_x, hi_x = int(np.floor(min(x0, x1) - r)), int(np.ceil(max(x0, x1) + r))
        lo_y, hi_y = int(np.floor(min(y0, y1) - r)), int(np.ceil(max(y0, y1) + r))
        lo_x, lo_y = max(lo_x, 0), max(lo_y, 0)
        hi_x, hi_y = min(hi_x, self.w - 1), min(hi_y, self.h - 1)
        if lo_x > hi_x or lo_y > hi_y:
            return
        ys, xs = np.mgrid[lo_y : hi_y + 1, lo_x : hi_x + 1]
        dx, dy = x1 - x0, y1 - y0
        seg_len2 = dx * dx + dy * dy
        if seg_len2 < 1e-9:
            t = np.zeros_like(xs, dtype=np.float64)
        else:
            t = np.clip(((xs - x0) * dx + (ys - y0) * dy) / seg_len2, 0.0, 1.0)
        px, py = x0 + t * dx, y0 + t * dy
        mask = (xs - px) ** 2 + (ys - py) ** 2 <= radius * radius
        img[lo_y : hi_y + 1, lo_x : hi_x + 1][mask] = color

    def _disc(self, img: Img, c: Vec, radius: float, color: np.ndarray) -> None:
        self._segment(img, c, c, radius, color)

    def _fill_hull(self, img: Img, pts2d: Vec, color: np.ndarray) -> None:
        hull = _convex_hull(pts2d)
        if len(hull) < 3:
            return
        ys = np.arange(max(int(np.floor(hull[:, 1].min())), 0), min(int(np.ceil(hull[:, 1].max())), self.h - 1) + 1)
        n = len(hull)
        for y in ys:
            xs: list[float] = []
            for i in range(n):
                (xa, ya), (xb, yb) = hull[i], hull[(i + 1) % n]
                if (ya <= y < yb) or (yb <= y < ya):
                    xs.append(xa + (y - ya) * (xb - xa) / (yb - ya))
            if len(xs) >= 2:
                x_lo, x_hi = max(int(np.ceil(min(xs))), 0), min(int(np.floor(max(xs))), self.w - 1)
                if x_lo <= x_hi:
                    img[y, x_lo : x_hi + 1] = color

    # -- scene ----------------------------------------------------------------

    def render(self, arm_pts: Vec, cube_center: Vec, cube_size: float, *, grasped: bool, success: bool) -> Img:
        """Rasterise one frame. ``arm_pts`` are the (9, 3) points from ``MetalKinematics``."""
        img = np.empty((self.h, self.w, 3), dtype=np.uint8)
        img[:] = SKY
        # Ground plane as a filled quad plus a 10 cm grid.
        corners = np.array([[-0.45, -0.55, 0.0], [0.65, -0.55, 0.0], [0.65, 0.55, 0.0], [-0.45, 0.55, 0.0]])
        c2d, _ = self.project(corners)
        self._fill_hull(img, c2d, GROUND)
        for x in np.arange(-0.4, 0.61, 0.1):
            p, _ = self.project(np.array([[x, -0.5, 0.0], [x, 0.5, 0.0]]))
            self._segment(img, p[0], p[1], 0.6, GRID)
        for y in np.arange(-0.5, 0.51, 0.1):
            p, _ = self.project(np.array([[-0.4, y, 0.0], [0.6, y, 0.0]]))
            self._segment(img, p[0], p[1], 0.6, GRID)
        # Cube.
        h = cube_size / 2
        offs = np.array([[sx, sy, sz] for sx in (-h, h) for sy in (-h, h) for sz in (-h, h)])
        cube2d, _ = self.project(cube_center + offs)
        self._fill_hull(img, cube2d, CUBE_HELD if grasped else CUBE)
        # Arm: thick near the base, thinner at the wrist.
        pts2d, _ = self.project(arm_pts)
        radii = np.linspace(7.0, 3.0, len(pts2d) - 1)
        for i in range(len(pts2d) - 1):
            self._segment(img, pts2d[i], pts2d[i + 1], float(radii[i]), ARM)
        for i in range(1, len(pts2d) - 2):
            self._disc(img, pts2d[i], 4.0, JOINT)
        self._disc(img, pts2d[-1], 3.5, TIP_OK if success else TIP)
        return img


def _convex_hull(points: Vec) -> Vec:
    """Andrew's monotone chain; returns hull vertices in counter-clockwise order."""
    pts = sorted({(float(x), float(y)) for x, y in np.atleast_2d(points)})
    if len(pts) <= 2:
        return np.asarray(pts, dtype=np.float64)

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)
