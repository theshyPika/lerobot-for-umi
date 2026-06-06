# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Action interpolation for smoother robot control.

Provides configurable Nx control rate by interpolating between consecutive actions.
Useful with RTC and action-chunking policies to reduce jerkiness.
"""

import math

import torch
from torch import Tensor

# temp fix @ck
def detect_rotvec_indices_from_keys(action_keys: list[str]) -> list[tuple[int, int, int]]:
    """Detect axis-angle (``wx``/``wy``/``wz``) rotvec triplets in an action key list.

    Returns a list of ``(idx_wx, idx_wy, idx_wz)`` index triplets for every
    distinct prefix in ``action_keys`` that has all three of ``<prefix>.wx``,
    ``<prefix>.wy``, ``<prefix>.wz`` present. Used to tell ``ActionInterpolator``
    which columns of the action vector encode rotations so it can avoid
    linearly interpolating across the antipodal-twin discontinuity of the
    rotvec representation.
    """
    key_to_idx = {k: i for i, k in enumerate(action_keys)}
    groups: list[tuple[int, int, int]] = []
    seen_prefixes: set[str] = set()
    for k in action_keys:
        if not k.endswith(".wx"):
            continue
        prefix = k[:-3]
        if prefix in seen_prefixes:
            continue
        wy = f"{prefix}.wy"
        wz = f"{prefix}.wz"
        if wy in key_to_idx and wz in key_to_idx:
            groups.append((key_to_idx[k], key_to_idx[wy], key_to_idx[wz]))
            seen_prefixes.add(prefix)
    return groups

# temp fix @ck
def _align_rotvec_to_ref(r_new: Tensor, r_ref: Tensor) -> Tensor:
    """Replace ``r_new`` with its antipodal twin if that twin is closer to ``r_ref``.

    A rotation ``R`` represented by rotvec ``r = θ·u`` (``θ = ‖r‖``, unit axis ``u``)
    can equivalently be written as ``r_twin = (2π − θ)·(−u)``. Both vectors represent
    the same physical rotation but lie exactly ``2π`` apart in 3-vector space — i.e.
    a linear blend between them sweeps through the origin (identity rotation).
    Picking the representation that is closer to ``r_ref`` lets a subsequent linear
    blend take the geometric short path instead of tumbling through identity.

    Operates on the last dimension being length 3.
    """
    # eps guard for the degenerate "no rotation" case where the axis is undefined.
    norm_new = torch.linalg.norm(r_new)
    if float(norm_new) < 1e-6:
        return r_new
    r_twin = (2.0 * math.pi - norm_new) * (-r_new / norm_new)
    if torch.linalg.norm(r_twin - r_ref) < torch.linalg.norm(r_new - r_ref):
        return r_twin
    return r_new


class ActionInterpolator:
    """Interpolates between consecutive actions for smoother control.

    When enabled with multiplier N, produces N actions per policy action
    by linearly interpolating between the previous and current action.

    Example with multiplier=3:
        prev_action -> [1/3 interpolated, 2/3 interpolated, current_action]

    This effectively multiplies the control rate for smoother motion.

    Usage:
        interpolator = ActionInterpolator(multiplier=2)  # 2x control rate

        # In control loop:
        if interpolator.needs_new_action():
            new_action = queue.get()
            if new_action:
                interpolator.add(new_action.cpu())

        action = interpolator.get()
        if action:
            robot.send_action(action)

    Rotation handling:
        When the action vector contains rotvec (axis-angle) columns,
        component-wise linear interpolation is geometrically incorrect near
        ``‖r‖ = π``: two adjacent actions can be antipodal twins of the same
        rotation but lie ~``2π`` apart as 3-vectors, so the blend sweeps through
        identity and the gripper tumbles. Pass ``rotation_indices`` to align each
        rotvec triplet to ``_prev`` before linear interpolation; this gives the
        geometric short path without changing anything for other columns.
        The companion helper :func:`detect_rotvec_indices_from_keys` infers the
        triplets from the action key naming convention.
    """

    def __init__(
        self,
        multiplier: int = 1,
        rotation_indices: list[tuple[int, int, int]] | None = None,
    ):
        """Initialize the interpolator.

        Args:
            multiplier: Control rate multiplier (1 = no interpolation, 2 = 2x, 3 = 3x, etc.)
            rotation_indices: Optional list of ``(idx_wx, idx_wy, idx_wz)`` index
                triplets specifying rotvec columns of the action vector. Each
                triplet is aligned against ``_prev`` (antipodal-twin pick) before
                the linear interpolation step. ``None`` (default) keeps the pure
                linear behaviour.
        """
        if multiplier < 1:
            raise ValueError(f"multiplier must be >= 1, got {multiplier}")
        self.multiplier = multiplier
        self.rotation_indices: list[tuple[int, int, int]] = list(rotation_indices or [])
        self._prev: Tensor | None = None
        self._buffer: list[Tensor] = []
        self._idx = 0

    @property
    def enabled(self) -> bool:
        """Whether interpolation is active (multiplier > 1)."""
        return self.multiplier > 1

    def reset(self):
        """Reset interpolation state (call between episodes)."""
        self._prev = None
        self._buffer = []
        self._idx = 0

    def needs_new_action(self) -> bool:
        """Check if a new action is needed from the queue."""
        return self._idx >= len(self._buffer)

    def add(self, action: Tensor) -> None:
        """Add a new action and compute interpolated sequence.

        Args:
            action: New action tensor from policy/queue (already on CPU).
        """
        if self.multiplier > 1 and self._prev is not None:
            # Antipodal-twin alignment of each rotvec triplet against ``_prev`` so the
            # subsequent linear blend takes the geometric short path instead of
            # tumbling through identity. No-op for the all-linear case
            # (``rotation_indices`` empty) and for triplets whose new rotvec is already
            # the closer representation.
            if self.rotation_indices:
                action = action.clone()
                for ix, iy, iz in self.rotation_indices:
                    r_new = action[[ix, iy, iz]]
                    r_ref = self._prev[[ix, iy, iz]]
                    r_aligned = _align_rotvec_to_ref(r_new, r_ref)
                    action[ix] = r_aligned[0]
                    action[iy] = r_aligned[1]
                    action[iz] = r_aligned[2]

            self._buffer = []
            for i in range(1, self.multiplier + 1):
                t = i / self.multiplier
                interp = self._prev + t * (action - self._prev)
                self._buffer.append(interp)
        else:
            # First step: no previous action yet, so run at base FPS without interpolation.
            self._buffer = [action.clone()]
        self._prev = action.clone()
        self._idx = 0

    def get(self) -> Tensor | None:
        """Get the next interpolated action.

        Returns:
            Next action tensor, or None if buffer is exhausted.
        """
        if self._idx >= len(self._buffer):
            return None
        action = self._buffer[self._idx]
        self._idx += 1
        return action

    def get_control_interval(self, fps: float) -> float:
        """Get the control interval based on interpolation multiplier.

        Args:
            fps: Base frames per second.

        Returns:
            Control interval in seconds (divided by multiplier).
        """
        return 1.0 / (fps * self.multiplier)
