#!/usr/bin/env python3
"""Find episodes dominated by observation subtasks in LeRobot datasets.

This script is read-only. It scans data parquet files, maps ``subtask_index`` to
names from ``meta/subtasks.parquet``, and prints per-dataset episode indices that
match an observation subtask pattern.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


DEFAULT_BASE_ROOT = Path("/data1/training_data/teleop/g2")
DEFAULT_DATASETS = [
    "g2_dual_arm_g1",
    "g2_dual_arm_g1_2",
    "g2_dual_arm_g2",
    "g2_dual_arm_g3",
    "g2_dual_arm_g4",
    "g2_dual_arm_g5",
    "g2_dual_arm_g6",
    "g2_dual_arm_g7",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List LeRobot episodes whose frames are observation subtasks."
    )
    parser.add_argument(
        "--base-root",
        type=Path,
        default=DEFAULT_BASE_ROOT,
        help=f"Base directory used with --dataset-names. Default: {DEFAULT_BASE_ROOT}",
    )
    parser.add_argument(
        "--dataset-names",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Dataset directory names under --base-root.",
    )
    parser.add_argument(
        "--roots",
        type=Path,
        nargs="+",
        default=None,
        help="Explicit dataset roots. Overrides --base-root/--dataset-names.",
    )
    parser.add_argument(
        "--pattern",
        default=r"观察|<观察>",
        help="Regex matched against subtask names. Default matches observation subtasks.",
    )
    parser.add_argument(
        "--min-ratio",
        type=float,
        default=1.0,
        help="Minimum fraction of frames in an episode that must match the pattern. Default: 1.0.",
    )
    parser.add_argument(
        "--output-suffix",
        default="_nostill",
        help="Suffix used when printing lerobot-edit-dataset output paths.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional JSON file for machine-readable results.",
    )
    return parser.parse_args()


def load_subtasks(root: Path) -> tuple[pd.DataFrame, dict[int, str]]:
    path = root / "meta/subtasks.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing subtasks metadata: {path}")

    subtasks = pd.read_parquet(path)
    if subtasks.index.name != "subtask":
        if "subtask" not in subtasks.columns:
            raise ValueError(f"{path} must have a 'subtask' index or column")
        subtasks = subtasks.set_index("subtask")

    if "subtask_index" not in subtasks.columns:
        raise ValueError(f"{path} must contain a 'subtask_index' column")

    mapping = {int(row["subtask_index"]): str(name) for name, row in subtasks.iterrows()}
    return subtasks.sort_values("subtask_index"), mapping


def find_matching_episodes(root: Path, pattern: re.Pattern[str], min_ratio: float) -> dict:
    subtasks, index_to_name = load_subtasks(root)
    matching_subtask_indices = {
        idx for idx, name in index_to_name.items() if pattern.search(name) is not None
    }

    data_files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No data parquet files found under {root / 'data'}")

    episode_counts: dict[int, int] = {}
    episode_match_counts: dict[int, int] = {}
    for path in data_files:
        df = pd.read_parquet(path, columns=["episode_index", "subtask_index"])
        if "episode_index" not in df.columns or "subtask_index" not in df.columns:
            raise ValueError(f"{path} must contain episode_index and subtask_index columns")

        subtask_values = df["subtask_index"].astype(int)
        unknown = sorted(set(subtask_values.unique()) - set(index_to_name))
        if unknown:
            raise ValueError(f"{path} contains unmapped subtask_index values: {unknown}")

        match_mask = subtask_values.isin(matching_subtask_indices)
        counts = df.groupby("episode_index").size()
        match_counts = df[match_mask].groupby("episode_index").size()

        for ep_idx, count in counts.items():
            ep_idx = int(ep_idx)
            episode_counts[ep_idx] = episode_counts.get(ep_idx, 0) + int(count)
        for ep_idx, count in match_counts.items():
            ep_idx = int(ep_idx)
            episode_match_counts[ep_idx] = episode_match_counts.get(ep_idx, 0) + int(count)

    matches = []
    for ep_idx in sorted(episode_counts):
        total = episode_counts[ep_idx]
        matched = episode_match_counts.get(ep_idx, 0)
        ratio = matched / total if total else 0.0
        if ratio >= min_ratio and matched > 0:
            matches.append(
                {
                    "episode_index": ep_idx,
                    "total_frames": total,
                    "matching_frames": matched,
                    "match_ratio": ratio,
                }
            )

    return {
        "root": str(root),
        "repo_id": root.name,
        "matching_subtasks": [
            {"subtask_index": idx, "subtask": index_to_name[idx]}
            for idx in sorted(matching_subtask_indices)
        ],
        "episodes": matches,
    }


def print_result(result: dict, output_suffix: str) -> None:
    root = Path(result["root"])
    repo_id = result["repo_id"]
    episodes = [item["episode_index"] for item in result["episodes"]]

    print(f"\n== {repo_id} ==")
    if result["matching_subtasks"]:
        print("matching subtasks:")
        for item in result["matching_subtasks"]:
            print(f"  {item['subtask_index']}: {item['subtask']}")
    else:
        print("matching subtasks: none")

    if not episodes:
        print("episodes to delete: []")
        return

    print(f"episodes to delete: {episodes}")
    for item in result["episodes"]:
        print(
            f"  ep={item['episode_index']} frames={item['matching_frames']}/"
            f"{item['total_frames']} ratio={item['match_ratio']:.3f}"
        )

    new_repo_id = repo_id + output_suffix
    new_root = root.with_name(root.name + output_suffix)
    print("delete command:")
    print(
        "  lerobot-edit-dataset "
        f"--repo_id {repo_id} "
        f"--root {root} "
        f"--new_repo_id {new_repo_id} "
        f"--new_root {new_root} "
        "--operation.type delete_episodes "
        f"--operation.episode_indices \"{json.dumps(episodes)}\""
    )


def main() -> None:
    args = parse_args()
    if not 0.0 < args.min_ratio <= 1.0:
        raise ValueError("--min-ratio must be in (0, 1]")

    roots = args.roots if args.roots is not None else [args.base_root / name for name in args.dataset_names]
    pattern = re.compile(args.pattern)

    results = []
    for root in roots:
        result = find_matching_episodes(root, pattern, args.min_ratio)
        results.append(result)
        print_result(result, args.output_suffix)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
