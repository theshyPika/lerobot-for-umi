#!/usr/bin/env python3
"""Filter short still-frame runs from a local LeRobot v3 dataset.

When to use:
    Use this after converting raw G2 task folders into LeRobot v3 datasets and
    after removing full observation-only episodes.  This tool targets the
    remaining brief repeated/frozen frames inside otherwise valid teleoperation
    episodes.  Run it before merging task datasets or, if your datasets are
    already merged, before recomputing relative-action statistics.

What it does:
    It detects brief still-frame runs from low-dimensional parquet features
    such as ``action`` and ``observation.state``, then rewrites a new dataset
    while preserving episode boundaries, task metadata, and subtask metadata.

Examples:
    # 1) Dry run on the default G2 cleaned dataset and write a JSON report.
    python tools/filter_still_frames.py \
        --dry-run \
        --json-out /tmp/g2_still_frame_report.json

    # 2) Export a filtered dataset with the default detection thresholds.
    python tools/filter_still_frames.py \
        --root /data1/training_data/teleop/g2/nostill/g2_dual_arm_g1_7_clean \
        --new-root /data1/training_data/teleop/g2/nostill/g2_dual_arm_g1_7_clean_no_sf \
        --video-backend pyav \
        --vcodec h264

    # 3) Use a stricter setting when you only want to remove near-exact
    # repeated frames.
    python tools/filter_still_frames.py \
        --dry-run \
        --threshold 1e-6 \
        --max-run 2 \
        --json-out /tmp/g2_still_frame_report_tight.json
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lerobot.datasets.io_utils import write_subtasks
from lerobot.datasets.lerobot_dataset import LeRobotDataset


DEFAULT_ROOT = Path("/data1/training_data/teleop/g2/nostill/g2_dual_arm_g1_7_clean")


@dataclass
class EpisodePlan:
    episode_index: int
    total_frames: int
    removed_frames: list[int]
    kept_indices: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove short still-frame runs from a local LeRobot v3 dataset. "
            "By default this is a dry run and only reports what would be removed."
        )
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Input LeRobot dataset root.")
    parser.add_argument("--repo-id", default=None, help="Input repo_id. Defaults to input root name.")
    parser.add_argument(
        "--new-root",
        type=Path,
        default=None,
        help="Output dataset root. Required unless --dry-run is true.",
    )
    parser.add_argument(
        "--new-repo-id",
        default=None,
        help="Output repo_id. Defaults to output root name, or '<repo-id>_no_sf'.",
    )
    parser.add_argument(
        "--feature-keys",
        nargs="+",
        default=["action", "observation.state"],
        help="Numeric features used for still-frame detection.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-5,
        help="L2 threshold for a frame to be considered unchanged from the previous frame.",
    )
    parser.add_argument(
        "--per-dim-threshold",
        type=float,
        default=None,
        help="Optional max absolute per-dimension threshold. If set, both L2 and per-dim checks must pass.",
    )
    parser.add_argument(
        "--min-run",
        type=int,
        default=1,
        help="Minimum length of a still run to remove. A run contains duplicate frames after the first frame.",
    )
    parser.add_argument(
        "--max-run",
        type=int,
        default=3,
        help="Maximum still-run length to remove. Longer pauses are kept as intentional behavior.",
    )
    parser.add_argument(
        "--min-episode-frames",
        type=int,
        default=2,
        help="Drop an episode if filtering would leave fewer than this many frames.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only report removals; do not write a dataset.")
    parser.add_argument("--overwrite", action="store_true", help="Delete --new-root first if it already exists.")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path for a JSON report.")
    parser.add_argument("--vcodec", default="h264", help="Video codec for output videos.")
    parser.add_argument("--video-backend", default=None, help="Video backend used by LeRobot when reading input.")
    parser.add_argument("--encoder-threads", type=int, default=None, help="Threads per output video encoder.")
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=8,
        help="Async image writer threads used before video encoding.",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def as_matrix(series: pd.Series, key: str) -> np.ndarray:
    values = series.to_list()
    try:
        return np.asarray([np.asarray(v, dtype=np.float64).reshape(-1) for v in values])
    except ValueError as exc:
        raise ValueError(f"Feature {key!r} has inconsistent vector shapes") from exc


def unchanged_from_previous(
    episode_df: pd.DataFrame,
    feature_keys: list[str],
    threshold: float,
    per_dim_threshold: float | None,
) -> np.ndarray:
    unchanged = np.zeros(len(episode_df), dtype=bool)
    if len(episode_df) <= 1:
        return unchanged

    checks = []
    for key in feature_keys:
        if key not in episode_df.columns:
            raise ValueError(f"Missing feature column {key!r} in data parquet")
        matrix = as_matrix(episode_df[key], key)
        diffs = matrix[1:] - matrix[:-1]
        l2_ok = np.linalg.norm(diffs, axis=1) <= threshold
        if per_dim_threshold is not None:
            dim_ok = np.max(np.abs(diffs), axis=1) <= per_dim_threshold
            checks.append(l2_ok & dim_ok)
        else:
            checks.append(l2_ok)

    unchanged[1:] = np.logical_and.reduce(checks)
    return unchanged


def find_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def build_filter_plan(
    root: Path,
    feature_keys: list[str],
    threshold: float,
    per_dim_threshold: float | None,
    min_run: int,
    max_run: int,
    min_episode_frames: int,
) -> tuple[list[EpisodePlan], list[int]]:
    columns = ["index", "episode_index", "frame_index", *feature_keys]
    episode_frames: dict[int, list[pd.DataFrame]] = {}

    data_files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No data parquet files found under {root / 'data'}")

    for path in tqdm(data_files, desc="Scanning parquet"):
        df = pd.read_parquet(path, columns=columns)
        for episode_index, ep_df in df.groupby("episode_index", sort=True):
            episode_frames.setdefault(int(episode_index), []).append(ep_df)

    plans: list[EpisodePlan] = []
    dropped_episodes: list[int] = []
    for episode_index in sorted(episode_frames):
        ep_df = pd.concat(episode_frames[episode_index], ignore_index=True)
        ep_df = ep_df.sort_values("frame_index").reset_index(drop=True)
        unchanged = unchanged_from_previous(ep_df, feature_keys, threshold, per_dim_threshold)

        remove_positions: set[int] = set()
        for start, end in find_true_runs(unchanged):
            run_len = end - start
            if min_run <= run_len <= max_run:
                remove_positions.update(range(start, end))

        kept_df = ep_df.drop(index=sorted(remove_positions))
        if len(kept_df) < min_episode_frames:
            dropped_episodes.append(episode_index)
            kept_indices: list[int] = []
            removed_frames = ep_df["frame_index"].astype(int).tolist()
        else:
            kept_indices = kept_df["index"].astype(int).tolist()
            removed_frames = ep_df.iloc[sorted(remove_positions)]["frame_index"].astype(int).tolist()

        plans.append(
            EpisodePlan(
                episode_index=episode_index,
                total_frames=len(ep_df),
                removed_frames=removed_frames,
                kept_indices=kept_indices,
            )
        )

    return plans, dropped_episodes


def frame_for_writer(dataset: LeRobotDataset, index: int) -> dict[str, Any]:
    item = dataset[index]
    frame: dict[str, Any] = {"task": item["task"]}
    auto_keys = {"index", "timestamp", "frame_index", "episode_index", "task_index"}
    for key, feature in dataset.meta.features.items():
        if key in auto_keys:
            continue
        value = item[key]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        elif hasattr(value, "cpu"):
            value = value.cpu().numpy()
        else:
            value = np.asarray(value)

        if feature["dtype"] in {"image", "video"} and value.ndim == 3 and value.shape[0] in {1, 3}:
            value = np.moveaxis(value, 0, -1)
        elif feature["dtype"] not in {"image", "video"}:
            value = np.asarray(value, dtype=np.dtype(feature["dtype"])).reshape(feature["shape"])
        frame[key] = value
    return frame


def write_filtered_dataset(
    src: LeRobotDataset,
    plans: list[EpisodePlan],
    new_root: Path,
    new_repo_id: str,
    vcodec: str,
    video_backend: str | None,
    encoder_threads: int | None,
    image_writer_threads: int,
) -> LeRobotDataset:
    dst = LeRobotDataset.create(
        repo_id=new_repo_id,
        root=new_root,
        fps=src.meta.fps,
        features=src.meta.features,
        robot_type=src.meta.robot_type,
        use_videos=len(src.meta.video_keys) > 0,
        vcodec=vcodec,
        video_backend=video_backend,
        encoder_threads=encoder_threads,
        image_writer_threads=image_writer_threads,
    )

    if src.meta.subtasks is not None:
        dst.meta.subtasks = src.meta.subtasks.copy()
        write_subtasks(dst.meta.subtasks, dst.meta.root)

    try:
        for plan in tqdm(plans, desc="Writing episodes"):
            if not plan.kept_indices:
                continue
            for index in tqdm(plan.kept_indices, desc=f"episode {plan.episode_index}", leave=False):
                dst.add_frame(frame_for_writer(src, index))
            dst.save_episode()
    finally:
        dst.finalize()

    return LeRobotDataset(repo_id=new_repo_id, root=new_root, video_backend=video_backend)


def make_report(plans: list[EpisodePlan], dropped_episodes: list[int]) -> dict[str, Any]:
    total_frames = sum(plan.total_frames for plan in plans)
    removed_frames = sum(len(plan.removed_frames) for plan in plans)
    kept_frames = sum(len(plan.kept_indices) for plan in plans)
    touched = [plan for plan in plans if plan.removed_frames]
    return {
        "total_episodes": len(plans),
        "dropped_episodes": dropped_episodes,
        "touched_episodes": len(touched),
        "total_frames": total_frames,
        "removed_frames": removed_frames,
        "kept_frames": kept_frames,
        "removed_ratio": removed_frames / total_frames if total_frames else 0.0,
        "episodes": [
            {
                "episode_index": plan.episode_index,
                "total_frames": plan.total_frames,
                "removed_count": len(plan.removed_frames),
                "removed_frames": plan.removed_frames,
                "kept_count": len(plan.kept_indices),
            }
            for plan in plans
            if plan.removed_frames or not plan.kept_indices
        ],
    }


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    if args.threshold < 0:
        raise ValueError("--threshold must be non-negative")
    if args.per_dim_threshold is not None and args.per_dim_threshold < 0:
        raise ValueError("--per-dim-threshold must be non-negative")
    if args.min_run <= 0 or args.max_run < args.min_run:
        raise ValueError("--min-run must be positive and --max-run must be >= --min-run")

    repo_id = args.repo_id or args.root.name
    plans, dropped_episodes = build_filter_plan(
        root=args.root,
        feature_keys=args.feature_keys,
        threshold=args.threshold,
        per_dim_threshold=args.per_dim_threshold,
        min_run=args.min_run,
        max_run=args.max_run,
        min_episode_frames=args.min_episode_frames,
    )
    report = make_report(plans, dropped_episodes)

    logging.info(
        "Still-frame plan: remove %d/%d frames (%.4f), touched episodes=%d, dropped episodes=%d",
        report["removed_frames"],
        report["total_frames"],
        report["removed_ratio"],
        report["touched_episodes"],
        len(dropped_episodes),
    )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info("Wrote report to %s", args.json_out)

    if args.dry_run:
        return

    if args.new_root is None:
        raise ValueError("--new-root is required when not using --dry-run")
    if args.new_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.new_root} already exists; pass --overwrite to replace it")
        shutil.rmtree(args.new_root)

    new_repo_id = args.new_repo_id or args.new_root.name if args.new_root is not None else f"{repo_id}_no_sf"
    src = LeRobotDataset(repo_id=repo_id, root=args.root, video_backend=args.video_backend)
    dst = write_filtered_dataset(
        src=src,
        plans=plans,
        new_root=args.new_root,
        new_repo_id=new_repo_id,
        vcodec=args.vcodec,
        video_backend=args.video_backend,
        encoder_threads=args.encoder_threads,
        image_writer_threads=args.image_writer_threads,
    )
    logging.info("Wrote filtered dataset to %s (%d episodes, %d frames)", dst.root, dst.num_episodes, dst.num_frames)


if __name__ == "__main__":
    main()
