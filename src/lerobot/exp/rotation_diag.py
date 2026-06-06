"""RTC chunk rotation diagnostic.

For each RTC inference chunk, compare ADJACENT-STEP rotvec distance against the
actual angular distance between the two rotations they represent. Distinguishes:

  - REPRESENTATION_FLIP : rotvec vector distance is large but the real angular
    distance between rotations is small. Linear interpolation of these rotvecs
    (as done by ``ActionInterpolator``) will sweep through identity/non-physical
    intermediate orientations -> the gripper appears to "tumble".

  - REAL_FAST_ROTATION  : both the rotvec distance AND the angular distance are
    large -> the policy genuinely commanded a fast rotation.

Run by exporting ``LEROBOT_DIAG_RTC_ROTVEC=1`` before launching the rollout;
the rotvec column indices are inferred from the layout name passed to
``log_chunk_diagnostics``.

This is a throwaway probe; delete when no longer useful.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from lerobot.utils.rotation import Rotation

logger = logging.getLogger(__name__)


# Indices of (wx, wy, wz) in the flat action vector, per action layout.
_LAYOUTS: dict[str, tuple[tuple[int, int, int] | None, tuple[int, int, int] | None]] = {
    # (left_wxyz, right_wxyz)
    "dual_arm_with_gripper":       ((3, 4, 5),  (10, 11, 12)),  # [lxyz lwxyz lg rxyz rwxyz rg]
    "dual_arm_no_gripper":         ((3, 4, 5),  (9, 10, 11)),   # [lxyz lwxyz rxyz rwxyz]
    "left_arm_only_with_gripper":  ((3, 4, 5),  None),
    "right_arm_only_with_gripper": (None,       (3, 4, 5)),
}


def analyze_chunk_rotation(
    chunk: torch.Tensor,
    wxyz_columns: tuple[int, int, int],
    tag: str,
    flip_rotvec_threshold: float = 0.5,
    flip_ratio_threshold: float = 3.0,
    max_details: int = 3,
) -> str:
    """Summary line + at most ``max_details`` detail lines for one chunk + one arm.

    Args:
        chunk: ``(T, action_dim)`` tensor (a single chunk, squeezed of the batch dim).
        wxyz_columns: indices of (wx, wy, wz) in the action vector.
        tag: short label for the log line, e.g. ``"[r] processed"``.
        flip_rotvec_threshold: minimum ``|delta_rotvec|`` (rad) to consider a step worth flagging.
        flip_ratio_threshold: a step is classified as a representation flip when
            ``|delta_rotvec| > flip_ratio_threshold * actual_angular_distance``.
        max_details: cap on per-step detail lines.
    """
    rv = chunk[:, list(wxyz_columns)].detach().cpu().numpy().astype(float)  # (T, 3)
    T = len(rv)
    if T < 2:
        return f"{tag}: T={T}, skipping"

    # What linear interpolation would sweep through.
    rotvec_jumps = np.linalg.norm(np.diff(rv, axis=0), axis=1)  # (T-1,)

    # The true angular distance between adjacent rotations.
    angular_jumps = np.full(T - 1, np.nan)
    for i in range(T - 1):
        if not (np.all(np.isfinite(rv[i])) and np.all(np.isfinite(rv[i + 1]))):
            continue
        try:
            r1 = Rotation.from_rotvec(rv[i])
            r2 = Rotation.from_rotvec(rv[i + 1])
            angular_jumps[i] = float(np.linalg.norm((r2 * r1.inv()).as_rotvec()))
        except Exception:
            pass

    finite_ang = np.where(np.isfinite(angular_jumps), angular_jumps, 0.0)
    flip_mask = (rotvec_jumps > flip_rotvec_threshold) & (
        rotvec_jumps > flip_ratio_threshold * np.maximum(finite_ang, 1e-6)
    )
    real_fast_mask = (rotvec_jumps > flip_rotvec_threshold) & (~flip_mask)
    n_flips = int(flip_mask.sum())
    n_real_fast = int(real_fast_mask.sum())

    flag = (
        "REPRESENTATION_FLIP"
        if n_flips > 0
        else ("REAL_FAST_ROTATION" if n_real_fast > 0 else "ok")
    )
    max_rv = float(rotvec_jumps.max())
    max_ang = float(np.nanmax(angular_jumps)) if np.any(np.isfinite(angular_jumps)) else float("nan")
    summary = (
        f"{tag}: T={T} max_rv_jump={max_rv:.3f} max_angular_jump={max_ang:.3f} "
        f"flips={n_flips} real_fast={n_real_fast} {flag}"
    )

    if n_flips > 0 or n_real_fast > 0:
        ranked = np.argsort(-rotvec_jumps)
        lines = []
        for s in ranked[:max_details]:
            kind = "FLIP" if flip_mask[s] else ("FAST" if real_fast_mask[s] else "")
            r1, r2 = rv[s], rv[s + 1]
            lines.append(
                f"  [{kind}] step{s}: ({r1[0]:+.3f},{r1[1]:+.3f},{r1[2]:+.3f}) "
                f"-> ({r2[0]:+.3f},{r2[1]:+.3f},{r2[2]:+.3f}) "
                f"|delta_rv|={rotvec_jumps[s]:.3f} actual_angular={angular_jumps[s]:.3f}"
            )
        if lines:
            summary += "\n" + "\n".join(lines)

    return summary


def log_chunk_diagnostics(
    original: torch.Tensor,
    processed: torch.Tensor,
    chunk_idx: int | None = None,
    layout: str = "dual_arm_with_gripper",
) -> None:
    """Run ``analyze_chunk_rotation`` on raw + post-processed chunks for both arms.

    Pass ``original`` and ``processed`` from the RTC inference loop. ``layout``
    picks the rotvec column indices; defaults to G2 dual-arm with gripper.
    """
    l_idx, r_idx = _LAYOUTS.get(layout, _LAYOUTS["dual_arm_with_gripper"])
    chunk_tag = f"chunk#{chunk_idx}" if chunk_idx is not None else ""

    def _log(name: str, summary: str) -> None:
        if "ok" in summary.splitlines()[0]:
            logger.debug("ROTVEC_DIAG %s %s", chunk_tag, summary)
        else:
            logger.warning("ROTVEC_DIAG %s %s", chunk_tag, summary)

    if l_idx is not None:
        _log("l/original",  analyze_chunk_rotation(original,  l_idx, "[l] original "))
        _log("l/processed", analyze_chunk_rotation(processed, l_idx, "[l] processed"))
    if r_idx is not None:
        _log("r/original",  analyze_chunk_rotation(original,  r_idx, "[r] original "))
        _log("r/processed", analyze_chunk_rotation(processed, r_idx, "[r] processed"))
