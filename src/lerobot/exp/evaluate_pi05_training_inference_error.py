#!/usr/bin/env python
"""Evaluate PI05 action inference error on LeRobot training episodes.

The script samples one episode per task by default, runs policy chunk inference
from each current observation, postprocesses the chunk back to the dataset's
absolute action space, and compares it with the recorded future actions.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def _set_default_writable_caches() -> None:
    os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf-datasets-cache")
    os.environ.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


_set_default_writable_caches()

from lerobot.configs import PreTrainedConfig  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: E402,F401
from lerobot.processor.pipeline import PolicyProcessorPipeline  # noqa: E402
from lerobot.types import TransitionKey  # noqa: E402
from lerobot.utils.constants import ACTION  # noqa: E402


DEFAULT_DATASET_ROOT = Path(
    "/home/ck/finetune-data/nostill_ep_fr_quat/g2_dual_arm_g1_7_recomputed_stats"
)
DEFAULT_CHECKPOINT = Path(
    "/home/ck/models/finetune_models/pi05_g2_dual_arm_g1_7_recomputed_stats_clean_nostill_ep_fr_quat/checkpoints/025000/pretrained_model"
)


@dataclass
class RunningStats:
    count: int = 0
    abs_sum: np.ndarray | None = None
    sq_sum: np.ndarray | None = None
    max_abs: np.ndarray | None = None
    horizon_abs_sum: dict[int, float] = field(default_factory=lambda: defaultdict(float))
    horizon_count: dict[int, int] = field(default_factory=lambda: defaultdict(int))

    def update(self, error: np.ndarray) -> None:
        if error.size == 0:
            return
        abs_err = np.abs(error)
        sq_err = error * error
        if self.abs_sum is None:
            self.abs_sum = np.zeros(abs_err.shape[-1], dtype=np.float64)
            self.sq_sum = np.zeros(abs_err.shape[-1], dtype=np.float64)
            self.max_abs = np.zeros(abs_err.shape[-1], dtype=np.float64)
        self.count += int(abs_err.shape[0])
        self.abs_sum += abs_err.sum(axis=0)
        self.sq_sum += sq_err.sum(axis=0)
        self.max_abs = np.maximum(self.max_abs, abs_err.max(axis=0))

        per_horizon_mae = abs_err.mean(axis=1)
        for horizon, mae in enumerate(per_horizon_mae):
            self.horizon_abs_sum[horizon] += float(mae)
            self.horizon_count[horizon] += 1

    def update_from(self, other: "RunningStats") -> None:
        if other.count == 0 or other.abs_sum is None or other.sq_sum is None or other.max_abs is None:
            return
        if self.abs_sum is None:
            self.abs_sum = np.zeros_like(other.abs_sum)
            self.sq_sum = np.zeros_like(other.sq_sum)
            self.max_abs = np.zeros_like(other.max_abs)
        self.count += other.count
        self.abs_sum += other.abs_sum
        self.sq_sum += other.sq_sum
        self.max_abs = np.maximum(self.max_abs, other.max_abs)
        for horizon, value in other.horizon_abs_sum.items():
            self.horizon_abs_sum[horizon] += value
        for horizon, value in other.horizon_count.items():
            self.horizon_count[horizon] += value

    def summary(self, action_names: list[str]) -> dict[str, Any]:
        if self.count == 0 or self.abs_sum is None or self.sq_sum is None or self.max_abs is None:
            return {"count": 0}

        mae_per_dim = self.abs_sum / self.count
        rmse_per_dim = np.sqrt(self.sq_sum / self.count)
        horizon_mae = {
            str(h): self.horizon_abs_sum[h] / self.horizon_count[h]
            for h in sorted(self.horizon_count)
            if self.horizon_count[h] > 0
        }
        dims = [
            {
                "name": action_names[i] if i < len(action_names) else str(i),
                "mae": float(mae_per_dim[i]),
                "rmse": float(rmse_per_dim[i]),
                "max_abs": float(self.max_abs[i]),
            }
            for i in range(len(mae_per_dim))
        ]
        return {
            "count": self.count,
            "mae": float(mae_per_dim.mean()),
            "rmse": float(np.sqrt(self.sq_sum.sum() / (self.count * len(mae_per_dim)))),
            "max_abs": float(self.max_abs.max()),
            "first_step_mae": float(horizon_mae.get("0", float("nan"))),
            "horizon_mae": horizon_mae,
            "per_dim": dims,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--repo-id", default=None, help="Defaults to dataset root directory name.")
    parser.add_argument("--device", default=None, help="Override checkpoint device, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument("--video-backend", default="torchcodec")
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--task-index", type=int, action="append", default=None)
    parser.add_argument("--episode-index", type=int, action="append", default=None)
    parser.add_argument("--max-frames-per-episode", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-horizon", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample", choices=["first", "random"], default="first")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--print-top-dims", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=20, help="Print progress every N evaluated frames.")
    parser.add_argument("--print-subtasks", action="store_true", help="Print selected episode subtask counts.")
    parser.add_argument(
        "--debug-prompts",
        type=int,
        default=0,
        help="Print N pre-tokenizer PI05 prompts from each selected episode before evaluating.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print selected episodes; do not load policy.")
    parser.add_argument("--offline", action="store_true", help="Force Hugging Face/transformers offline mode.")
    return parser.parse_args()


def load_frame_table(dataset_root: Path) -> pd.DataFrame:
    data_paths = sorted((dataset_root / "data").glob("*/*.parquet"))
    if not data_paths:
        raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}")
    features = json.loads((dataset_root / "meta" / "info.json").read_text())["features"]
    columns = ["index", "episode_index", "frame_index", "task_index", ACTION]
    if "subtask_index" in features:
        columns.append("subtask_index")
    return pd.concat((pd.read_parquet(path, columns=columns) for path in data_paths), ignore_index=True)


def load_task_names(dataset_root: Path) -> dict[int, str]:
    tasks = pd.read_parquet(dataset_root / "meta" / "tasks.parquet")
    return {int(row.task_index): str(task) for task, row in tasks.iterrows()}


def load_subtask_names(dataset_root: Path) -> dict[int, str]:
    path = dataset_root / "meta" / "subtasks.parquet"
    if not path.exists():
        return {}
    subtasks = pd.read_parquet(path)
    return {int(row.subtask_index): str(subtask) for subtask, row in subtasks.iterrows()}


def select_episodes(
    frame_table: pd.DataFrame,
    *,
    task_indices: list[int] | None,
    episode_indices: list[int] | None,
    episodes_per_task: int,
    sample: str,
    seed: int,
) -> list[tuple[int, int]]:
    if episode_indices:
        pairs = []
        for ep_idx in episode_indices:
            rows = frame_table[frame_table["episode_index"] == ep_idx]
            if rows.empty:
                raise ValueError(f"episode_index={ep_idx} not found")
            pairs.append((int(rows["task_index"].iloc[0]), int(ep_idx)))
        return pairs

    available = frame_table[["task_index", "episode_index"]].drop_duplicates()
    if task_indices is not None:
        available = available[available["task_index"].isin(task_indices)]
    rng = np.random.default_rng(seed)
    pairs = []
    for task_idx, group in available.groupby("task_index", sort=True):
        episodes = sorted(int(v) for v in group["episode_index"].unique())
        if not episodes:
            continue
        if sample == "random":
            chosen = sorted(rng.choice(episodes, size=min(episodes_per_task, len(episodes)), replace=False))
        else:
            chosen = episodes[:episodes_per_task]
        pairs.extend((int(task_idx), int(ep_idx)) for ep_idx in chosen)
    return pairs


def load_policy_and_processors(
    checkpoint: Path, device: str | None
) -> tuple[PI05Policy, PolicyProcessorPipeline, PolicyProcessorPipeline]:
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    if device is not None:
        cfg.device = device
    cfg.pretrained_path = None
    cfg.gradient_checkpointing = False
    cfg.compile_model = False

    policy = PI05Policy.from_pretrained(checkpoint, config=cfg, strict=True, local_files_only=True)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        cfg,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": cfg.device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    return policy, preprocessor, postprocessor


def to_numpy_action(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.float32)
    return np.asarray(value, dtype=np.float32)


def get_episode_actions(frame_table: pd.DataFrame, episode_index: int) -> np.ndarray:
    rows = frame_table[frame_table["episode_index"] == episode_index].sort_values("frame_index")
    if rows.empty:
        raise ValueError(f"episode_index={episode_index} has no rows")
    return np.stack([to_numpy_action(v) for v in rows[ACTION].to_list()], axis=0)


def get_episode_subtask_counts(
    frame_table: pd.DataFrame, episode_index: int, subtask_names: dict[int, str]
) -> list[dict[str, Any]]:
    if "subtask_index" not in frame_table.columns:
        return []
    rows = frame_table[frame_table["episode_index"] == episode_index].sort_values("frame_index")
    counts = []
    for subtask_idx, group in rows.groupby("subtask_index", sort=True):
        subtask_idx = int(subtask_idx)
        counts.append(
            {
                "subtask_index": subtask_idx,
                "frames": int(len(group)),
                "first_frame": int(group["frame_index"].min()),
                "last_frame": int(group["frame_index"].max()),
                "subtask": subtask_names.get(subtask_idx, ""),
            }
        )
    return counts


def print_subtask_counts(
    frame_table: pd.DataFrame, episode_index: int, subtask_names: dict[int, str], indent: str = "  "
) -> None:
    counts = get_episode_subtask_counts(frame_table, episode_index, subtask_names)
    if not counts:
        print(f"{indent}subtasks: <none>")
        return
    print(f"{indent}subtasks:")
    for item in counts:
        print(
            f"{indent}  subtask_index={item['subtask_index']} "
            f"frames={item['frames']} frame_range={item['first_frame']}:{item['last_frame']} "
            f"subtask={item['subtask']}"
        )


def postprocess_chunk(postprocessor: PolicyProcessorPipeline, chunk: torch.Tensor) -> np.ndarray:
    processed = postprocessor(chunk.detach())
    if isinstance(processed, torch.Tensor):
        return processed.squeeze(0).detach().cpu().numpy().astype(np.float32)
    return np.asarray(processed, dtype=np.float32)


def debug_prompts(dataset: LeRobotDataset, preprocessor: PolicyProcessorPipeline, limit: int) -> None:
    if limit <= 0:
        return
    for frame_idx in range(min(limit, len(dataset))):
        item = dataset[frame_idx]
        prompt = None
        raw_subtask = item.get("subtask")
        for transition in preprocessor.step_through(item):
            complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}) or {}
            task_value = complementary_data.get("task")
            if isinstance(task_value, list) and task_value and isinstance(task_value[0], str):
                if task_value[0].startswith("Task:") and "State:" in task_value[0]:
                    prompt = task_value[0]
        print(f"  debug_prompt frame_idx={frame_idx} raw_subtask={raw_subtask}")
        if prompt is None:
            print("    <PI05 prompt not found>")
        else:
            print(f"    {prompt}")


def summarize_top_dims(summary: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
    dims = summary.get("per_dim", [])
    return sorted(dims, key=lambda item: item["mae"], reverse=True)[:top_k]


def evaluate_episode(
    *,
    dataset: LeRobotDataset,
    policy: PI05Policy,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    true_actions: np.ndarray,
    max_frames: int | None,
    frame_stride: int,
    max_horizon: int | None,
    log_every: int,
) -> RunningStats:
    stats = RunningStats()
    last_start = len(dataset) - 1
    frame_indices = range(0, last_start + 1, frame_stride)
    if max_frames is not None:
        frame_indices = list(frame_indices)[:max_frames]

    total = len(frame_indices)
    with torch.inference_mode():
        for eval_idx, frame_idx in enumerate(frame_indices, start=1):
            item = dataset[frame_idx]
            model_batch = preprocessor(item)
            pred_chunk = policy.predict_action_chunk(model_batch)
            pred_abs = postprocess_chunk(postprocessor, pred_chunk)

            available = min(pred_abs.shape[0], true_actions.shape[0] - frame_idx)
            if max_horizon is not None:
                available = min(available, max_horizon)
            if available <= 0:
                continue

            target = true_actions[frame_idx : frame_idx + available]
            stats.update(pred_abs[:available] - target)
            if log_every > 0 and (eval_idx == 1 or eval_idx % log_every == 0 or eval_idx == total):
                print(f"  frame_progress={eval_idx}/{total} frame_idx={frame_idx}", flush=True)
    return stats


def main() -> None:
    args = parse_args()
    if args.offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")
    if args.episodes_per_task < 1:
        raise ValueError("--episodes-per-task must be >= 1")

    repo_id = args.repo_id or args.dataset_root.name
    frame_table = load_frame_table(args.dataset_root)
    task_names = load_task_names(args.dataset_root)
    subtask_names = load_subtask_names(args.dataset_root)
    action_names = json.loads((args.dataset_root / "meta" / "info.json").read_text())["features"][ACTION][
        "names"
    ]

    pairs = select_episodes(
        frame_table,
        task_indices=args.task_index,
        episode_indices=args.episode_index,
        episodes_per_task=args.episodes_per_task,
        sample=args.sample,
        seed=args.seed,
    )
    if not pairs:
        raise ValueError("No episodes selected")

    print(f"dataset_root={args.dataset_root}")
    print(f"checkpoint={args.checkpoint}")
    print(f"selected_episodes={pairs}")
    print(f"device_override={args.device or '<checkpoint/default>'}")
    print(f"video_backend={args.video_backend}")

    if args.dry_run:
        print("dry_run episode selection:")
        for task_idx, episode_idx in pairs:
            rows = frame_table[frame_table["episode_index"] == episode_idx]
            print(
                f"  task_index={task_idx} episode={episode_idx} "
                f"frames={len(rows)} task={task_names.get(task_idx, '')}"
            )
            if args.print_subtasks:
                print_subtask_counts(frame_table, episode_idx, subtask_names, indent="    ")
        return

    start_load = time.perf_counter()
    policy, preprocessor, postprocessor = load_policy_and_processors(args.checkpoint, args.device)
    print(f"loaded policy/processors in {time.perf_counter() - start_load:.1f}s")

    overall = RunningStats()
    task_stats: dict[int, RunningStats] = defaultdict(RunningStats)
    episode_summaries = []

    for task_idx, episode_idx in pairs:
        ep_start = time.perf_counter()
        dataset = LeRobotDataset(
            repo_id,
            root=args.dataset_root,
            episodes=[episode_idx],
            video_backend=args.video_backend,
            download_videos=False,
        )
        subtask_counts = get_episode_subtask_counts(frame_table, episode_idx, subtask_names)
        if args.print_subtasks:
            print_subtask_counts(frame_table, episode_idx, subtask_names)
        if args.debug_prompts > 0:
            debug_prompts(dataset, preprocessor, args.debug_prompts)
        true_actions = get_episode_actions(frame_table, episode_idx)
        ep_stats = evaluate_episode(
            dataset=dataset,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            true_actions=true_actions,
            max_frames=args.max_frames_per_episode,
            frame_stride=args.frame_stride,
            max_horizon=args.max_horizon,
            log_every=args.log_every,
        )
        elapsed = time.perf_counter() - ep_start
        summary = ep_stats.summary(action_names)
        summary.update(
            {
                "task_index": task_idx,
                "task": task_names.get(task_idx, ""),
                "episode_index": episode_idx,
                "episode_frames": len(dataset),
                "subtasks": subtask_counts,
                "elapsed_s": elapsed,
                "top_dims_by_mae": summarize_top_dims(summary, args.print_top_dims),
            }
        )
        episode_summaries.append(summary)
        overall.update_from(ep_stats)
        task_stats[task_idx].update_from(ep_stats)

        print(
            f"task_index={task_idx} episode={episode_idx} frames={len(dataset)} "
            f"count={summary.get('count', 0)} mae={summary.get('mae', float('nan')):.6f} "
            f"first_step_mae={summary.get('first_step_mae', float('nan')):.6f} "
            f"rmse={summary.get('rmse', float('nan')):.6f} elapsed={elapsed:.1f}s"
        )
        if task_idx == 3:
            print("task_index=3 top dims by MAE:")
            for item in summary["top_dims_by_mae"]:
                print(
                    f"  {item['name']}: mae={item['mae']:.6f} "
                    f"rmse={item['rmse']:.6f} max_abs={item['max_abs']:.6f}"
                )

    task_summaries = {}
    for task_idx, stats in sorted(task_stats.items()):
        summary = stats.summary(action_names)
        summary["task"] = task_names.get(task_idx, "")
        summary["top_dims_by_mae"] = summarize_top_dims(summary, args.print_top_dims)
        task_summaries[str(task_idx)] = summary

    overall_summary = overall.summary(action_names)
    overall_summary["top_dims_by_mae"] = summarize_top_dims(overall_summary, args.print_top_dims)

    result = {
        "dataset_root": str(args.dataset_root),
        "checkpoint": str(args.checkpoint),
        "episodes": episode_summaries,
        "tasks": task_summaries,
        "overall": overall_summary,
    }

    print("overall:")
    print(
        f"  count={overall_summary.get('count', 0)} "
        f"mae={overall_summary.get('mae', float('nan')):.6f} "
        f"first_step_mae={overall_summary.get('first_step_mae', float('nan')):.6f} "
        f"rmse={overall_summary.get('rmse', float('nan')):.6f} "
        f"max_abs={overall_summary.get('max_abs', float('nan')):.6f}"
    )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
