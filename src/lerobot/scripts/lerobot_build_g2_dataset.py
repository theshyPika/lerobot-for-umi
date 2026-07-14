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

"""Build a training-ready G2 LeRobot dataset in one command.

This orchestrator turns the scattered four-stage G2 data pipeline into a single
deterministic pipeline:

    raw source
      ──[convert]──> per-group datasets          (subprocess, per-group parallel)
      ──[prune*]──>  drop observation episodes   (optional, in-process)
      ──[filter]──>  drop still-frame runs        (subprocess, per-group parallel)
      ──[merge]──>   one merged dataset           (library call)
      ──[recompute_stats]──> final dataset        (library call, relative-action stats)

It does not reimplement any stage: convert/filter shell out to the existing
``tools/`` scripts; prune/merge/recompute_stats call ``lerobot.datasets`` library
functions directly. All intermediate paths live under a single ``--output-base``
tree with a ``.pipeline_done`` sentinel per stage unit, so an interrupted run
resumes by skipping finished units (``--force`` reruns).

Mirror of ``lerobot_edit_dataset`` for the draccus ``@parser.wrap()`` style
(supports ``--config_path build.yaml``).

Examples:
    # Full quaternion (ee) dual-arm build, parallel over groups.
    # NOTE: draccus uses underscores and JSON for list fields, e.g. --groups '["G7"]'.
    lerobot-build-g2-dataset \\
        --source_dir /data1/training_data/sourceFile \\
        --output_base /data1/training_data/teleop/g2/build \\
        --groups '["G1","G2","G3","G4","G5","G6","G7"]' \\
        --action_type ee --arm_mode dual --vcodec h264 \\
        --filter.enabled true \\
        --stats.relative_action true --stats.chunk_size 50 \\
        --stats.relative_exclude_joints '["gripper"]' --stats.num_workers 24 \\
        --parallel_groups 4 \\
        --final_name g2_dual_arm_g1_7_quat

    # Reproduce an experiment from a YAML config.
    lerobot-build-g2-dataset --config_path build_quat.yaml

    # Dry run: print the plan + subprocess commands, write nothing.
    lerobot-build-g2-dataset --dry_run true ...

    # Resume / partial: skip finished units, or start from a stage.
    lerobot-build-g2-dataset --from_stage filter ...
    lerobot-build-g2-dataset --stages '["merge","recompute_stats"]' ...
"""

import logging
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from lerobot.configs import parser
from lerobot.datasets import LeRobotDataset, delete_episodes, merge_datasets, recompute_stats
from lerobot.utils.utils import init_logging

# The convert and filter stages shell out to these standalone scripts.
REPO_ROOT = Path(__file__).resolve().parents[3]
CONVERT_SCRIPT = REPO_ROOT / "tools" / "create_g2_dataset_using_lerobot.py"
FILTER_SCRIPT = REPO_ROOT / "tools" / "filter_still_frames.py"
FIND_EPISODES_SCRIPT = REPO_ROOT / "tools" / "find_observation_subtask_episodes.py"

# Sentinel file touched inside each stage output dir once it completes.
SENTINEL = ".pipeline_done"

# Observation subtask episodes are fully-observation episodes (every frame's
# subtask matches the observation pattern). Prune deletes all of them.
_OBSERVATION_PATTERN_DEFAULT = r"观察|<观察>"
_OBSERVATION_MIN_RATIO = 1.0

STAGE_ORDER = ["convert", "prune", "filter", "merge", "recompute_stats"]


# --------------------------------------------------------------------------- #
# Config dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class FilterConfig:
    enabled: bool = True
    # L2 norm threshold below which a frame is "unchanged" from the previous one.
    threshold: float = 1e-5
    # Optional max |delta| per dimension; if set, both L2 and per-dim must pass.
    per_dim_threshold: float | None = None
    # Remove still-runs whose length is in [min_run, max_run].
    min_run: int = 1
    max_run: int = 3
    # Drop an episode entirely if filtering would leave fewer frames than this.
    min_episode_frames: int = 2
    # Numeric parquet features used for still-frame detection.
    feature_keys: list[str] = field(default_factory=lambda: ["action", "observation.state"])
    # Output video codec / backend for the filtered dataset.
    vcodec: str = "h264"
    video_backend: str | None = None
    encoder_threads: int | None = None
    image_writer_threads: int = 8


@dataclass
class PruneConfig:
    """Drop episodes that are entirely observation subtasks.

    Observation-only ("still observation") episodes hurt training. New data
    collection no longer records them, so this stage only matters for legacy
    data. When enabled, ALL matching episodes are deleted (no ratio knob).
    """

    enabled: bool = False
    pattern: str = _OBSERVATION_PATTERN_DEFAULT


@dataclass
class StatsConfig:
    """Relative-action statistics recomputation (required for relative-action training)."""

    relative_action: bool = True
    chunk_size: int = 50
    relative_exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])
    num_workers: int = 24
    skip_image_video: bool = True


@dataclass
class G2BuildConfig:
    # Raw G2 teleoperation source (task folders -> episode folders w/ metaInfo.json).
    source_dir: str = "/data1/training_data/sourceFile"
    # Root of the structured output tree (convert/prune/filter/merge/final live under here).
    output_base: str = ""
    # Task groups to convert (one convert subprocess per group).
    groups: list[str] = field(
        default_factory=lambda: ["G1", "G2", "G3", "G4", "G5", "G6", "G7"]
    )
    # Final dataset name (repo_id) — also names the merge/ and final/ dirs.
    final_name: str = ""

    # --- convert stage ---
    arm_mode: str = "dual"
    action_type: str = "ee"  # "joint" | "ee" (ee == quaternion EE pose)
    video_storage: str = "video"  # "video" | "image"
    vcodec: str = "libsvtav1"
    fps: int = 30
    max_episodes_per_group: int | None = None

    # --- stage sub-configs ---
    filter: FilterConfig = field(default_factory=FilterConfig)
    prune: PruneConfig = field(default_factory=PruneConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)

    # --- orchestration ---
    parallel_groups: int = 4
    # Ordered subset of STAGE_ORDER to run (default: all).
    stages: list[str] = field(default_factory=lambda: list(STAGE_ORDER))
    # Start from this stage (skips earlier ones); ignored if empty.
    from_stage: str | None = None
    # Rerun finished units (removes their output + sentinel first).
    force: bool = False
    # Print the plan + commands; write nothing.
    dry_run: bool = False


# --------------------------------------------------------------------------- #
# Path / sentinel helpers
# --------------------------------------------------------------------------- #
def _slug(group: str) -> str:
    return group.lower()


def _sentinel(path: Path) -> Path:
    return path / SENTINEL


def _is_done(path: Path) -> bool:
    return _sentinel(path).exists()


def _mark_done(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _sentinel(path).touch()


def _reset(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def convert_out(cfg: G2BuildConfig, group: str) -> Path:
    return Path(cfg.output_base) / "convert" / _slug(group)


def prune_out(cfg: G2BuildConfig, group: str) -> Path:
    return Path(cfg.output_base) / "prune" / _slug(group)


def filter_out(cfg: G2BuildConfig, group: str) -> Path:
    return Path(cfg.output_base) / "filter" / _slug(group)


def merge_out(cfg: G2BuildConfig) -> Path:
    return Path(cfg.output_base) / "merge" / cfg.final_name


def final_out(cfg: G2BuildConfig) -> Path:
    return Path(cfg.output_base) / "final" / cfg.final_name


def _validate(cfg: G2BuildConfig) -> None:
    if not cfg.output_base:
        raise ValueError("--output-base is required")
    if not cfg.final_name:
        raise ValueError("--final-name is required")
    if not Path(cfg.source_dir).exists():
        raise FileNotFoundError(f"--source-dir not found: {cfg.source_dir}")
    if cfg.arm_mode.lower() not in {"dual", "left", "right", "both", "bimanual", "l", "r"}:
        raise ValueError(f"Unsupported arm_mode: {cfg.arm_mode}")
    if cfg.action_type not in {"joint", "ee"}:
        raise ValueError(f"Unsupported action_type: {cfg.action_type}")
    if cfg.video_storage not in {"video", "image"}:
        raise ValueError(f"Unsupported video_storage: {cfg.video_storage}")
    if not CONVERT_SCRIPT.exists():
        raise FileNotFoundError(f"Convert script missing: {CONVERT_SCRIPT}")
    if not FILTER_SCRIPT.exists():
        raise FileNotFoundError(f"Filter script missing: {FILTER_SCRIPT}")
    unknown = [s for s in cfg.stages if s not in STAGE_ORDER]
    if unknown:
        raise ValueError(f"Unknown stages {unknown}; valid: {STAGE_ORDER}")
    if cfg.from_stage and cfg.from_stage not in STAGE_ORDER:
        raise ValueError(f"Unknown from_stage {cfg.from_stage}; valid: {STAGE_ORDER}")


def _effective_stages(cfg: G2BuildConfig) -> list[str]:
    stages = list(cfg.stages)
    if cfg.from_stage:
        idx = STAGE_ORDER.index(cfg.from_stage)
        stages = [s for s in STAGE_ORDER[idx:] if s in cfg.stages]
    # prune only runs when enabled
    if not cfg.prune.enabled:
        stages = [s for s in stages if s != "prune"]
    return stages


# --------------------------------------------------------------------------- #
# Subprocess command builders (for dry-run printing + execution)
# --------------------------------------------------------------------------- #
def _convert_cmd(cfg: G2BuildConfig, group: str, resume: bool) -> list[str]:
    cmd: list[str] = [
        sys.executable,
        str(CONVERT_SCRIPT),
        "--source-dir",
        cfg.source_dir,
        "--output-dir",
        str(Path(cfg.output_base) / "convert"),
        "--groups",
        group,
        "--arm-mode",
        cfg.arm_mode,
        "--action-type",
        cfg.action_type,
        "--video-storage",
        cfg.video_storage,
        "--dataset-name",
        _slug(group),
        "--fps",
        str(cfg.fps),
        "--vcodec",
        cfg.vcodec,
    ]
    if cfg.max_episodes_per_group is not None:
        cmd += ["--max-episodes-per-group", str(cfg.max_episodes_per_group)]
    # Only pass --resume to the convert script when an interrupted run already
    # exists on disk; a fresh run with --resume raises "dataset path does not exist".
    if resume:
        cmd += ["--resume"]
    return cmd


def _filter_cmd(cfg: G2BuildConfig, group: str, input_root: Path) -> list[str]:
    fc = cfg.filter
    cmd: list[str] = [
        sys.executable,
        str(FILTER_SCRIPT),
        "--root",
        str(input_root),
        "--new-root",
        str(filter_out(cfg, group)),
        "--new-repo-id",
        _slug(group),
        "--threshold",
        str(fc.threshold),
        "--min-run",
        str(fc.min_run),
        "--max-run",
        str(fc.max_run),
        "--min-episode-frames",
        str(fc.min_episode_frames),
        "--vcodec",
        fc.vcodec,
        "--image-writer-threads",
        str(fc.image_writer_threads),
        "--feature-keys",
        *fc.feature_keys,
    ]
    if fc.per_dim_threshold is not None:
        cmd += ["--per-dim-threshold", str(fc.per_dim_threshold)]
    if fc.video_backend is not None:
        cmd += ["--video-backend", fc.video_backend]
    if fc.encoder_threads is not None:
        cmd += ["--encoder-threads", str(fc.encoder_threads)]
    if cfg.force:
        cmd += ["--overwrite"]
    return cmd


# --------------------------------------------------------------------------- #
# Stage runners
# --------------------------------------------------------------------------- #
def _run_subprocess(cmd: list[str], group: str) -> None:
    logging.info("[%s] running: %s", group, " ".join(cmd))
    subprocess.run(cmd, check=True)


def _stage_convert(cfg: G2BuildConfig, groups: list[str]) -> None:
    jobs = []
    for group in groups:
        out = convert_out(cfg, group)
        if _is_done(out) and not cfg.force:
            logging.info("[convert:%s] done, skipping", group)
            continue
        # An existing-but-undone output means an interrupted convert run; resume it.
        resume = out.exists() and not cfg.force
        if cfg.force:
            _reset(out)
            resume = False
        jobs.append((group, out, resume))

    if not jobs:
        return
    if cfg.dry_run:
        for group, _, resume in jobs:
            logging.info("[convert:%s] (dry-run) %s", group, " ".join(_convert_cmd(cfg, group, resume)))
        return

    with ThreadPoolExecutor(max_workers=max(1, cfg.parallel_groups)) as pool:
        futs = {
            pool.submit(_run_subprocess, _convert_cmd(cfg, group, resume), group): group
            for group, _, resume in jobs
        }
        for fut in as_completed(futs):
            group = futs[fut]
            fut.result()  # raise on failure
            _mark_done(convert_out(cfg, group))
            logging.info("[convert:%s] done", group)


def _filter_input(cfg: G2BuildConfig, group: str) -> Path:
    """Resolve the input dataset root for the filter stage.

    If prune ran for this group, use its output; otherwise fall back to convert.
    """
    if cfg.prune.enabled and _is_done(prune_out(cfg, group)):
        return prune_out(cfg, group)
    return convert_out(cfg, group)


def _stage_prune(cfg: G2BuildConfig, groups: list[str]) -> None:
    pattern = re.compile(cfg.prune.pattern)
    for group in groups:
        src = convert_out(cfg, group)
        out = prune_out(cfg, group)
        if not _is_done(src):
            logging.warning("[prune:%s] convert output missing, skipping prune", group)
            continue
        if _is_done(out) and not cfg.force:
            logging.info("[prune:%s] done, skipping", group)
            continue
        if cfg.force:
            _reset(out)

        # Import the find tool lazily (it's a tools/ script, importable via path).
        sys.path.insert(0, str(FIND_EPISODES_SCRIPT.parent))
        try:
            import find_observation_subtask_episodes as find_tool  # type: ignore
        finally:
            sys.path.pop(0)

        result = find_tool.find_matching_episodes(src, pattern, _OBSERVATION_MIN_RATIO)
        episode_indices = [e["episode_index"] for e in result["episodes"]]

        if not episode_indices:
            logging.info("[prune:%s] no observation episodes; filter will read convert output", group)
            # No output to produce — filter falls back to convert via _filter_input.
            continue

        if cfg.dry_run:
            logging.info(
                "[prune:%s] (dry-run) would delete %d observation episodes",
                group,
                len(episode_indices),
            )
            continue

        if cfg.force:
            _reset(out)
        dataset = LeRobotDataset(repo_id=_slug(group), root=src)
        total = int(dataset.meta.total_episodes)
        if len(episode_indices) >= total:
            # delete_episodes refuses to empty a dataset. If every selected
            # episode is observation-only, skip pruning this group and let the
            # filter stage read the unpruned convert output instead of crashing.
            logging.warning(
                "[prune:%s] all %d episodes match the observation pattern; "
                "skipping prune (delete_episodes cannot empty the dataset). "
                "Filter will read the convert output instead.",
                group,
                total,
            )
            continue
        delete_episodes(dataset, episode_indices=episode_indices, output_dir=out, repo_id=_slug(group))
        _mark_done(out)
        logging.info("[prune:%s] deleted %d/%d observation episodes", group, len(episode_indices), total)


def _stage_filter(cfg: G2BuildConfig, groups: list[str]) -> None:
    if not cfg.filter.enabled:
        logging.info("filter disabled; merge will read %s", "prune/convert outputs")
        return

    jobs = []
    for group in groups:
        out = filter_out(cfg, group)
        if _is_done(out) and not cfg.force:
            logging.info("[filter:%s] done, skipping", group)
            continue
        if cfg.force:
            _reset(out)
        jobs.append((group, _filter_input(cfg, group)))

    if not jobs:
        return
    if cfg.dry_run:
        for group, inp in jobs:
            logging.info("[filter:%s] (dry-run) in=%s %s", group, inp, " ".join(_filter_cmd(cfg, group, inp)))
        return

    with ThreadPoolExecutor(max_workers=max(1, cfg.parallel_groups)) as pool:
        futs = {
            pool.submit(_run_subprocess, _filter_cmd(cfg, group, inp), f"filter:{group}"): group
            for group, inp in jobs
        }
        for fut in as_completed(futs):
            group = futs[fut]
            fut.result()
            _mark_done(filter_out(cfg, group))
            logging.info("[filter:%s] done", group)


def _discover_merge_inputs(cfg: G2BuildConfig) -> list[Path]:
    """Glob all completed per-group filter outputs (or convert/prune if filter disabled)."""
    base = Path(cfg.output_base)
    if cfg.filter.enabled:
        stage_dir = base / "filter"
    elif cfg.prune.enabled:
        stage_dir = base / "prune"
    else:
        stage_dir = base / "convert"
    roots = []
    if stage_dir.exists():
        for child in sorted(stage_dir.iterdir()):
            if child.is_dir() and _is_done(child):
                roots.append(child)
    return roots


def _stage_merge(cfg: G2BuildConfig) -> None:
    out = merge_out(cfg)
    if _is_done(out) and not cfg.force:
        logging.info("[merge] done, skipping")
        return
    if cfg.force:
        _reset(out)

    if cfg.dry_run:
        roots = _discover_merge_inputs(cfg)
        if roots:
            logging.info("[merge] (dry-run) would merge %d datasets -> %s", len(roots), out)
        else:
            expected = [str(filter_out(cfg, g)) for g in cfg.groups] if cfg.filter.enabled else [
                str(prune_out(cfg, g)) for g in cfg.groups
            ]
            logging.info("[merge] (dry-run) no completed inputs yet; would merge -> %s", out)
            logging.info("[merge] (dry-run) expected inputs: %s", expected)
        return

    roots = _discover_merge_inputs(cfg)
    if not roots:
        raise FileNotFoundError(
            f"No completed per-group datasets found under {Path(cfg.output_base)} to merge"
        )
    logging.info("[merge] merging %d datasets: %s", len(roots), [r.name for r in roots])

    datasets = [LeRobotDataset(repo_id=r.name, root=r) for r in roots]
    merged = merge_datasets(datasets, output_repo_id=cfg.final_name, output_dir=out)
    _mark_done(out)
    logging.info(
        "[merge] done: %d episodes, %d frames -> %s",
        merged.meta.total_episodes,
        merged.meta.total_frames,
        out,
    )


def _stage_recompute_stats(cfg: G2BuildConfig) -> None:
    src = merge_out(cfg)
    out = final_out(cfg)
    if _is_done(out) and not cfg.force:
        logging.info("[recompute_stats] done, skipping")
        return
    if cfg.force or out.exists():
        _reset(out)

    sc = cfg.stats
    logging.info(
        "[recompute_stats] %s -> %s (relative=%s, chunk=%d, exclude=%s, workers=%d)",
        src,
        out,
        sc.relative_action,
        sc.chunk_size,
        sc.relative_exclude_joints,
        sc.num_workers,
    )
    if cfg.dry_run:
        logging.info("[recompute_stats] (dry-run) would recompute stats -> %s", out)
        return

    if not _is_done(src):
        raise FileNotFoundError(f"Merge output missing (run merge first): {src}")

    # recompute_stats writes stats into dataset.root, so recompute in-place on the
    # merge output, then rename merge -> final. This avoids copying GBs of video
    # (which a copytree-then-recompute would do) while keeping merge reproducible.
    dataset = LeRobotDataset(repo_id=cfg.final_name, root=src)
    recompute_stats(
        dataset,
        skip_image_video=sc.skip_image_video,
        relative_action=sc.relative_action,
        relative_exclude_joints=sc.relative_exclude_joints,
        chunk_size=sc.chunk_size,
        num_workers=sc.num_workers,
    )
    shutil.move(str(src), str(out))
    _mark_done(out)
    verified = LeRobotDataset(repo_id=cfg.final_name, root=out)
    logging.info(
        "[recompute_stats] done: %d episodes, %d frames -> %s",
        verified.meta.total_episodes,
        verified.meta.total_frames,
        out,
    )


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
@parser.wrap()
def build_g2_dataset(cfg: G2BuildConfig) -> None:
    _validate(cfg)
    init_logging()

    logging.info(
        "G2 build pipeline: output_base=%s final=%s groups=%s action=%s arm=%s parallel=%d",
        cfg.output_base,
        cfg.final_name,
        cfg.groups,
        cfg.action_type,
        cfg.arm_mode,
        cfg.parallel_groups,
    )
    if cfg.dry_run:
        logging.info("DRY RUN — nothing will be written")

    stages = _effective_stages(cfg)
    logging.info("Stages: %s", stages)

    for stage in stages:
        logging.info("=== stage: %s ===", stage)
        if stage == "convert":
            _stage_convert(cfg, cfg.groups)
        elif stage == "prune":
            _stage_prune(cfg, cfg.groups)
        elif stage == "filter":
            _stage_filter(cfg, cfg.groups)
        elif stage == "merge":
            _stage_merge(cfg)
        elif stage == "recompute_stats":
            _stage_recompute_stats(cfg)

    final = final_out(cfg)
    if "recompute_stats" in stages and not cfg.dry_run:
        logging.info("✓ Training-ready dataset: %s", final)
    else:
        logging.info("Pipeline finished. Final dataset target: %s", final)


def main() -> None:
    build_g2_dataset()


if __name__ == "__main__":
    main()
