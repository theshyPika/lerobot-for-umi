#!/usr/bin/env python

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

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.types import EnvTransition, PolicyAction, TransitionKey

from .pipeline import ProcessorStep, ProcessorStepRegistry


@ProcessorStepRegistry.register("clip_action_processor")
@dataclass
class ClipActionProcessorStep(ProcessorStep):
    """
    Clips action values to a specified range.
    
    This processor is useful when policy outputs actions that are outside
    the expected range, which can happen with models like pi0.5 that may
    output normalized values outside the [-1, 1] range.
    
    Attributes:
        clip_min: Minimum value to clip actions to. Can be a scalar or a dictionary
                  mapping action names to specific min values.
        clip_max: Maximum value to clip actions to. Can be a scalar or a dictionary
                  mapping action names to specific max values.
        clip_normalized: If True, clips normalized actions (in [-1, 1] range).
                         If False, clips unnormalized actions.
    """
    
    clip_min: float | dict[str, float] = -1.0
    clip_max: float | dict[str, float] = 1.0
    clip_normalized: bool = True
    
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        
        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            return new_transition
        
        if not isinstance(action, PolicyAction):
            raise ValueError(f"Action should be a PolicyAction type got {type(action)}")
        
        # Clip the action
        clipped_action = self._clip_action(action)
        new_transition[TransitionKey.ACTION] = clipped_action
        
        return new_transition
    
    def _clip_action(self, action: PolicyAction) -> PolicyAction:
        """Clip action values to the specified range."""
        clipped_action = action.clone() if hasattr(action, 'clone') else action.copy()
        
        for key, value in action.items():
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value)
            
            # Get clip bounds for this action key
            min_val = self.clip_min
            max_val = self.clip_max
            
            if isinstance(min_val, dict):
                min_val = min_val.get(key, -1.0)
            
            if isinstance(max_val, dict):
                max_val = max_val.get(key, 1.0)
            
            # Note: The clip_normalized parameter is informational only in this implementation
            # It indicates whether the clipping is applied to normalized or unnormalized actions
            # but the actual clipping range is determined by clip_min and clip_max
            # Clip the value
            clipped_value = torch.clamp(value, min_val, max_val)
            clipped_action[key] = clipped_value
        
        return clipped_action
    
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """This step does not alter the feature definitions."""
        return features