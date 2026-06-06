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

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.rotation import (
    quat_canonicalize,
    quat_conjugate,
    quat_multiply,
    quat_normalize,
)

from .delta_action_processor import MapDeltaActionToRobotActionStep, MapTensorToDeltaActionDictStep
from .pipeline import ProcessorStep, ProcessorStepRegistry

# Re-export for backward compatibility
__all__ = [
    "MapDeltaActionToRobotActionStep",
    "MapTensorToDeltaActionDictStep",
    "RelativeActionsProcessorStep",
    "AbsoluteActionsProcessorStep",
    "to_relative_actions",
    "to_absolute_actions",
    "detect_quaternion_indices_from_names",
]


def detect_quaternion_indices_from_names(
    action_names: Sequence[str] | None,
) -> list[tuple[int, int, int, int]]:
    """Detect xyzw quaternion column quadruplets from action dimension names.

    Returns one ``(idx_qx, idx_qy, idx_qz, idx_qw)`` tuple per distinct prefix that
    has all four of ``<prefix>.qx/.qy/.qz/.qw`` present (e.g. ``l.ee`` -> the four
    ``l.ee.q*`` indices). Empty list when names are absent or contain no quaternion
    columns, in which case the relative/absolute conversion stays purely additive.

    Orientation dims must be composed multiplicatively rather than subtracted/added:
    quaternion subtraction has no geometric meaning. The quadruplet grouping is
    semantically required — each arm's 4 components form one rotation and must be
    multiplied as a unit.
    """
    if not action_names:
        return []
    name_to_idx = {str(n): i for i, n in enumerate(action_names)}
    indices: list[tuple[int, int, int, int]] = []
    seen: set[str] = set()
    for n in action_names:
        name = str(n)
        if not name.endswith(".qx"):
            continue
        prefix = name[:-3]
        if prefix in seen:
            continue
        comps = [f"{prefix}.qx", f"{prefix}.qy", f"{prefix}.qz", f"{prefix}.qw"]
        if all(c in name_to_idx for c in comps):
            indices.append(tuple(name_to_idx[c] for c in comps))  # type: ignore[arg-type]
            seen.add(prefix)
    return indices


def _additive_mask(
    mask: Sequence[bool], quaternion_indices: Sequence[tuple[int, int, int, int]]
) -> list[bool]:
    """Drop quaternion dims from the additive mask so they are handled multiplicatively."""
    mask_list = list(mask)
    quat_dims = {i for quad in quaternion_indices for i in quad}
    if not quat_dims:
        return mask_list
    return [bool(m) and (i not in quat_dims) for i, m in enumerate(mask_list)]


def to_relative_actions(
    actions: Tensor,
    state: Tensor,
    mask: Sequence[bool],
    quaternion_indices: Sequence[tuple[int, int, int, int]] | None = None,
) -> Tensor:
    """Convert absolute actions to relative for masked dims.

    Position/scalar dims: ``relative = action - state``. Quaternion quadruplets (if
    any): ``q_rel = q_action ⊗ q_state⁻¹``, canonicalized to qw >= 0 (short arc).

    Args:
        actions: (B, T, action_dim) or (B, action_dim).
        state: (B, state_dim). Broadcast across time dimension.
        mask: Which dims to convert additively. Can be shorter than action_dim.
        quaternion_indices: xyzw index quadruplets handled multiplicatively. These
            dims are excluded from the additive path.
    """
    quaternion_indices = list(quaternion_indices or [])
    # Align state to the same device/dtype as actions. _last_state is cached before
    # DeviceProcessorStep moves the transition, so it can be on CPU while actions are on CUDA.
    if state.device != actions.device or state.dtype != actions.dtype:
        state = state.to(device=actions.device, dtype=actions.dtype)

    mask_t = torch.tensor(
        _additive_mask(mask, quaternion_indices), dtype=actions.dtype, device=actions.device
    )
    dims = mask_t.shape[0]
    state_offset = state[..., :dims] * mask_t
    if actions.ndim == 3:
        state_offset = state_offset.unsqueeze(-2)
    actions = actions.clone()
    actions[..., :dims] -= state_offset

    for quad in quaternion_indices:
        idx = list(quad)
        q_act = actions[..., idx]  # untouched by the additive step (masked out)
        q_st = state[..., idx]
        if q_act.ndim == 3 and q_st.ndim == 2:
            q_st = q_st.unsqueeze(-2)
        q_rel = quat_multiply(q_act, quat_conjugate(quat_normalize(q_st)))
        actions[..., idx] = quat_canonicalize(quat_normalize(q_rel))
    return actions


def to_absolute_actions(
    actions: Tensor,
    state: Tensor,
    mask: Sequence[bool],
    quaternion_indices: Sequence[tuple[int, int, int, int]] | None = None,
) -> Tensor:
    """Convert relative actions back to absolute for masked dims.

    Position/scalar dims: ``absolute = relative + state``. Quaternion quadruplets (if
    any): ``q_action = q_rel ⊗ q_state``, with the (possibly non-unit) predicted q_rel
    normalized first and the result renormalized to a unit quaternion.

    Args:
        actions: (B, T, action_dim) or (B, action_dim).
        state: (B, state_dim). Broadcast across time dimension.
        mask: Which dims to convert additively. Can be shorter than action_dim.
        quaternion_indices: xyzw index quadruplets handled multiplicatively.
    """
    quaternion_indices = list(quaternion_indices or [])
    # Align state to the same device/dtype as actions. _last_state is cached before
    # DeviceProcessorStep moves the transition, so it can be on CPU while actions are on CUDA.
    if state.device != actions.device or state.dtype != actions.dtype:
        state = state.to(device=actions.device, dtype=actions.dtype)

    mask_t = torch.tensor(
        _additive_mask(mask, quaternion_indices), dtype=actions.dtype, device=actions.device
    )
    dims = mask_t.shape[0]
    state_offset = state[..., :dims] * mask_t
    if actions.ndim == 3:
        state_offset = state_offset.unsqueeze(-2)
    actions = actions.clone()
    actions[..., :dims] += state_offset

    for quad in quaternion_indices:
        idx = list(quad)
        q_rel = quat_normalize(actions[..., idx])  # model output is not exactly unit
        q_st = state[..., idx]
        if q_rel.ndim == 3 and q_st.ndim == 2:
            q_st = q_st.unsqueeze(-2)
        q_abs = quat_multiply(q_rel, quat_normalize(q_st))
        actions[..., idx] = quat_normalize(q_abs)
    return actions


@ProcessorStepRegistry.register("delta_actions_processor")
@dataclass
class RelativeActionsProcessorStep(ProcessorStep):
    """Converts absolute actions to relative actions (action -= state) for masked dimensions.

    Mirrors OpenPI's DeltaActions transform. Applied during preprocessing so the model
    trains on relative offsets instead of absolute positions.
    Caches the last seen state so a paired AbsoluteActionsProcessorStep can reverse
    the conversion during postprocessing.

    Attributes:
        enabled: Whether to apply the relative conversion.
        exclude_joints: Joint names to keep absolute (not converted to relative).
        action_names: Action dimension names from dataset metadata, used to build
            the mask from exclude_joints. If None, all dims are converted.
    """

    enabled: bool = False
    exclude_joints: list[str] = field(default_factory=list)
    action_names: list[str] | None = None
    _last_state: torch.Tensor | None = field(default=None, init=False, repr=False)

    def _build_mask(self, action_dim: int) -> list[bool]:
        if not self.exclude_joints or self.action_names is None:
            return [True] * action_dim

        exclude_tokens = [str(name).lower() for name in self.exclude_joints if name]
        if not exclude_tokens:
            return [True] * action_dim

        mask = []
        for name in self.action_names[:action_dim]:
            action_name = str(name).lower()
            is_excluded = any(token == action_name or token in action_name for token in exclude_tokens)
            mask.append(not is_excluded)

        if len(mask) < action_dim:
            mask.extend([True] * (action_dim - len(mask)))

        return mask

    def _quaternion_indices(self) -> list[tuple[int, int, int, int]]:
        """xyzw quaternion column quadruplets inferred from action_names (cached)."""
        if self.action_names is None:
            return []
        return detect_quaternion_indices_from_names(self.action_names)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION, {})
        state = observation.get(OBS_STATE) if observation else None

        # Always cache state for the paired AbsoluteActionsProcessorStep
        if state is not None:
            self._last_state = state

        if not self.enabled:
            return transition

        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        if action is None or state is None:
            return new_transition

        mask = self._build_mask(action.shape[-1])
        new_transition[TransitionKey.ACTION] = to_relative_actions(
            action, state, mask, self._quaternion_indices()
        )
        return new_transition

    def get_cached_state(self) -> torch.Tensor | None:
        """Return the cached ``observation.state`` used as the reference point for relative/absolute action conversions."""
        return self._last_state

    def get_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "exclude_joints": self.exclude_joints,
            "action_names": self.action_names,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("absolute_actions_processor")
@dataclass
class AbsoluteActionsProcessorStep(ProcessorStep):
    """Converts relative actions back to absolute actions (action += state) for all dimensions.

    Mirrors OpenPI's AbsoluteActions transform. Applied during postprocessing so
    predicted relative offsets are converted back to absolute positions for execution.
    Reads the cached state from its paired RelativeActionsProcessorStep.

    Attributes:
        enabled: Whether to apply the absolute conversion.
        relative_step: Reference to the paired RelativeActionsProcessorStep that caches state.
    """

    enabled: bool = False
    relative_step: RelativeActionsProcessorStep | None = field(default=None, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition

        if self.relative_step is None:
            raise RuntimeError(
                "AbsoluteActionsProcessorStep requires a paired RelativeActionsProcessorStep "
                "but relative_step is None. Ensure relative_step is set when constructing the postprocessor."
            )

        cached_state = self.relative_step.get_cached_state()
        if cached_state is None:
            raise RuntimeError(
                "AbsoluteActionsProcessorStep requires state from RelativeActionsProcessorStep "
                "but no state has been cached. Ensure the preprocessor runs before the postprocessor."
            )

        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            return new_transition

        mask = self.relative_step._build_mask(action.shape[-1])
        new_transition[TransitionKey.ACTION] = to_absolute_actions(
            action, cached_state, mask, self.relative_step._quaternion_indices()
        )
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
