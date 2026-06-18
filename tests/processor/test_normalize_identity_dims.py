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

"""Per-dimension IDENTITY normalization (``identity_dims``) on the real G2 dual-arm
+ gripper EE layout.

EE-orientation quaternion dims must bypass the per-component QUANTILES remap (stay raw),
while position/gripper dims still normalize. The quaternion indices are derived the same
way the production code does — ``detect_quaternion_indices_from_names`` over the ordered
action names — so this test also guards against the indices being scrambled.
"""

import numpy as np
import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.processor import (
    NormalizerProcessorStep,
    TransitionKey,
    UnnormalizerProcessorStep,
    create_transition,
)
from lerobot.processor.relative_action_processor import detect_quaternion_indices_from_names
from lerobot.utils.constants import ACTION

# Real dual-arm + gripper EE action layout (16 dims), same order as the dataset.
NAMES = [
    "l.ee.x", "l.ee.y", "l.ee.z", "l.ee.qx", "l.ee.qy", "l.ee.qz", "l.ee.qw", "l.ee.gripper.pos",
    "r.ee.x", "r.ee.y", "r.ee.z", "r.ee.qx", "r.ee.qy", "r.ee.qz", "r.ee.qw", "r.ee.gripper.pos",
]
DIM = len(NAMES)
QUAT_DIMS = sorted({i for quad in detect_quaternion_indices_from_names(NAMES) for i in quad})
NON_QUAT = [i for i in range(DIM) if i not in QUAT_DIMS]


def test_detect_lands_on_the_quaternion_columns_not_scrambled():
    # The whole scheme hinges on these indices: left q at 3-6, right q at 11-14.
    assert detect_quaternion_indices_from_names(NAMES) == [(3, 4, 5, 6), (11, 12, 13, 14)]
    assert QUAT_DIMS == [3, 4, 5, 6, 11, 12, 13, 14]


def _stats():
    # Non-trivial, asymmetric quantile ranges so normalization is clearly a no-op only
    # where identity_dims says so.
    q01 = np.array([-2, -2, -2, -0.3, 0.4, -0.5, 0.8, 0.0, -2, -2, -2, -0.5, -0.1, -0.7, 0.1, 0.0], dtype=np.float32)
    q99 = np.array([2, 2, 2, 0.5, 0.7, -0.1, 1.0, 1.0, 2, 2, 2, -0.1, 0.4, -0.3, 0.6, 1.0], dtype=np.float32)
    return {ACTION: {"q01": q01, "q99": q99}}


def _steps(identity_dims):
    feats = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(DIM,))}
    nm = {FeatureType.ACTION: NormalizationMode.QUANTILES}
    return (
        NormalizerProcessorStep(features=feats, norm_map=nm, stats=_stats(), identity_dims=identity_dims),
        UnnormalizerProcessorStep(features=feats, norm_map=nm, stats=_stats(), identity_dims=identity_dims),
    )


def _apply(step, x):
    return step(create_transition(action=x))[TransitionKey.ACTION]


def _sample():
    # in-range unit-ish quaternions + arbitrary position/gripper
    return torch.tensor([[
        1.0, -1.0, 0.5, -0.21, 0.55, -0.30, 0.93, 0.4,
        -0.7, 0.8, 1.3, -0.30, 0.10, -0.50, 0.35, 0.6,
    ]])


def test_quaternion_dims_bypass_only_those_columns():
    x = _sample()
    norm, _ = _steps({ACTION: QUAT_DIMS})
    out = _apply(norm, x.clone())
    # quaternion columns pass through unchanged
    torch.testing.assert_close(out[:, QUAT_DIMS], x[:, QUAT_DIMS])
    # every non-quaternion column is actually remapped (changed)
    for i in NON_QUAT:
        assert not torch.allclose(out[:, i], x[:, i]), f"dim {i} should have been normalized"


def test_without_identity_dims_quaternions_are_normalized():
    x = _sample()
    norm, _ = _steps(None)
    out = _apply(norm, x.clone())
    assert not torch.allclose(out[:, QUAT_DIMS], x[:, QUAT_DIMS])


def test_roundtrip_normalize_then_unnormalize():
    x = _sample()
    norm, unnorm = _steps({ACTION: QUAT_DIMS})
    back = _apply(unnorm, _apply(norm, x.clone()).clone())
    torch.testing.assert_close(back, x, atol=1e-5, rtol=1e-5)


def test_identity_dims_round_trips_through_get_config():
    norm, _ = _steps({ACTION: QUAT_DIMS})
    cfg = norm.get_config()
    assert cfg["identity_dims"] == {ACTION: QUAT_DIMS}
    rebuilt = NormalizerProcessorStep(
        features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(DIM,))},
        norm_map={FeatureType.ACTION: NormalizationMode.QUANTILES},
        stats=_stats(),
        identity_dims=cfg["identity_dims"],
    )
    x = _sample()
    out = _apply(rebuilt, x.clone())
    torch.testing.assert_close(out[:, QUAT_DIMS], x[:, QUAT_DIMS])
