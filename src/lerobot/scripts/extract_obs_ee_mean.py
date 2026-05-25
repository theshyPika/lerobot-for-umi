# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
Extract the observation EE mean from a pretrained pi05 model's normalizer.

The pi05 G2 policy stores per-feature dataset statistics inside its preprocessor
``policy_preprocessor_step_3_normalizer_processor.safetensors``. For a model
trained with EE-pose observations, ``observation.state.mean`` is a 14-dim
vector that aligns with the standard G2 EE feature names
(l.ee.x, l.ee.y, l.ee.z, l.ee.wx, l.ee.wy, l.ee.wz, l.ee.gripper.pos and the
r.* counterparts).

This script loads that mean vector, maps it back to named keys, prints it,
and optionally writes a JSON file consumable by ``reset_g2_to_ee_mean.py``.

Example:
    python -m lerobot.scripts.extract_obs_ee_mean \
        --model_path=/home/ck/models/finetune_models/pi05_g2_dual_arm_g1_7_relstats_clean/038000/pretrained_model \
        --output=/tmp/g2_ee_mean.json
"""

import argparse
import json
import logging
from pathlib import Path

from safetensors.torch import load_file

from lerobot.utils.utils import init_logging


DEFAULT_EE_NAMES_14 = [
    "l.ee.x",
    "l.ee.y",
    "l.ee.z",
    "l.ee.wx",
    "l.ee.wy",
    "l.ee.wz",
    "l.ee.gripper.pos",
    "r.ee.x",
    "r.ee.y",
    "r.ee.z",
    "r.ee.wx",
    "r.ee.wy",
    "r.ee.wz",
    "r.ee.gripper.pos",
]

DEFAULT_EE_NAMES_7_LEFT = DEFAULT_EE_NAMES_14[:7]
DEFAULT_EE_NAMES_7_RIGHT = DEFAULT_EE_NAMES_14[7:]


def _resolve_normalizer_path(model_path: Path) -> Path:
    """Locate the normalizer safetensors inside a pretrained_model directory."""
    if model_path.is_file():
        return model_path
    preferred = sorted(model_path.glob("policy_preprocessor*normalizer*.safetensors"))
    if preferred:
        return preferred[0]
    candidates = sorted(model_path.glob("*normalizer*.safetensors"))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find a normalizer safetensors file under {model_path}. "
            "Expected something like 'policy_preprocessor_step_*_normalizer_processor.safetensors'."
        )
    if len(candidates) > 1:
        logging.warning(
            "No preprocessor-normalizer file found; falling back to %s. Others: %s",
            candidates[0].name,
            [c.name for c in candidates[1:]],
        )
    return candidates[0]


def _pick_feature_names(num_values: int, override: list[str] | None) -> list[str]:
    if override is not None:
        if len(override) != num_values:
            raise ValueError(
                f"--feature_names has {len(override)} entries but observation.state.mean has {num_values}."
            )
        return list(override)
    if num_values == 14:
        return DEFAULT_EE_NAMES_14
    if num_values == 7:
        # Cannot infer single-arm side from stats alone; default to left and warn.
        logging.warning(
            "observation.state.mean has 7 entries — assuming left arm. "
            "Pass --feature_names if this is the right arm."
        )
        return DEFAULT_EE_NAMES_7_LEFT
    raise ValueError(
        f"observation.state.mean has {num_values} entries; no default naming. "
        "Pass --feature_names explicitly."
    )


def extract_obs_ee_mean(
    model_path: Path,
    feature_names: list[str] | None = None,
    stat: str = "mean",
) -> dict[str, float]:
    """Return the per-feature observation EE statistic from the normalizer."""
    normalizer_path = _resolve_normalizer_path(model_path)
    logging.info("Loading normalizer state from %s", normalizer_path)
    state = load_file(str(normalizer_path))

    key = f"observation.state.{stat}"
    if key not in state:
        available = sorted(k for k in state if k.startswith("observation.state."))
        raise KeyError(
            f"'{key}' not found in normalizer state. Available observation.state.* keys: {available}"
        )

    tensor = state[key].detach().cpu().flatten()
    values = [float(v) for v in tensor.tolist()]
    names = _pick_feature_names(len(values), feature_names)
    return dict(zip(names, values))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--model_path",
        type=Path,
        required=True,
        help="Path to a pretrained_model directory or directly to the normalizer safetensors file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON path to write the {feature_name: value} mapping.",
    )
    parser.add_argument(
        "--stat",
        type=str,
        default="mean",
        choices=["mean", "q50", "q10", "q90", "q01", "q99", "min", "max", "std"],
        help="Which observation.state statistic to extract (default: mean).",
    )
    parser.add_argument(
        "--feature_names",
        nargs="+",
        default=None,
        help="Override feature name list (must match the vector length).",
    )
    return parser.parse_args()


def main() -> None:
    init_logging()
    args = _parse_args()

    ee_stats = extract_obs_ee_mean(
        model_path=args.model_path,
        feature_names=args.feature_names,
        stat=args.stat,
    )

    print(f"observation.state.{args.stat} (from {args.model_path}):")
    for name, value in ee_stats.items():
        print(f"  {name:>20s} = {value: .6f}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_path": str(args.model_path),
            "stat": args.stat,
            "values": ee_stats,
        }
        with args.output.open("w") as f:
            json.dump(payload, f, indent=2)
        logging.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
