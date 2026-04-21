#!/usr/bin/env python

"""
Create a LeRobot v3.0 dataset from G2 teleoperation episodes.

This script converts the raw G2 teleoperation data under
`/data1/training_data/sourceFile` into a LeRobot v3.0 dataset suitable for
training or fine-tuning VLA-style models.

Input expectations:
- The source directory contains task folders, and each task folder contains
  episode folders with a `metaInfo.json`.
- Each episode contains trajectory data under `data/**/*.parquet`.
- Visual data is required. It is loaded from episode-local videos first, and
  falls back to image-like parquet columns only if present.

This script always exports 3 visual features:
- `observation.images.hand_left_color`
- `observation.images.hand_right_color`
- `observation.images.head_color`

It supports two action/state representations:
- `joint`: left/right single-arm 8D vectors, or dual-arm 16D vectors
- `ee`: left/right single-arm 7D vectors, or dual-arm 14D vectors

Arm selection:
- `--arm-mode dual`: export both arms
- `--arm-mode left`: export left arm only
- `--arm-mode right`: export right arm only

Task-related fields:
- `task` is derived from `metaInfo["taskDesc"]`, because `taskName` like
  `G1-2` is not suitable as a semantic language instruction for VLA training.
- `subtask_index` is written into the frame data.
- `meta/subtasks.parquet` is generated from `metaInfo["taskStep"]`.

Storage behavior:
- `--video-storage image`: images are written as image features and embedded
  into the dataset representation expected by LeRobot.
- `--video-storage video`: frames are still decoded during export, but the
  output dataset stores visual observations under `videos/`.

Performance notes:
- The main bottleneck is usually source video decoding.
- This script already fetches the 3 cameras in parallel per frame using
  threads. Dataset writing remains single-threaded on purpose, which is the
  safer choice with LeRobot's writer.
- If your server has NVIDIA GPU video encoding available, `--video-storage
  video --vcodec h264_nvenc` is usually much faster than CPU video encoding.
- `--video-storage image` avoids output video encoding entirely, which can be
  useful for debugging or small-scale validation, but output size will usually
  be larger.

Examples:
    # 1) Small smoke test: only export up to 2 episodes from G7.
    # Useful for validating schema, task/subtask fields, and visual loading.
    python create_g2_dataset_using_lerobot.py \
        --source-dir /data1/training_data/sourceFile \
        --output-dir /tmp/g2_exports \
        --groups G7 \
        --max-episodes-per-group 2 \
        --action-type ee \
        --video-storage video \
        --dataset-name g2_g7_smoke

    # 2) Standard full export in joint space, saving camera streams as videos.
    # Good default for full dataset generation.
    python create_g2_dataset_using_lerobot.py \
        --source-dir /data1/training_data/sourceFile \
        --output-dir /data1/training_data/lerobot_exports \
        --groups G1 G2 G3 G4 G5 G6 G7 \
        --arm-mode dual \
        --action-type joint \
        --video-storage video \
        --dataset-name g2_merged

    # 3) Right-arm-only export in joint space.
    # Useful when the downstream policy should only model the right arm.
    python create_g2_dataset_using_lerobot.py \
        --source-dir /data1/training_data/sourceFile \
        --output-dir /data1/training_data/lerobot_exports \
        --groups G1 G2 \
        --arm-mode right \
        --action-type joint \
        --video-storage video \
        --dataset-name g2_right_arm_g1_g2

    # 4) Left-arm-only export in end-effector space.
    python create_g2_dataset_using_lerobot.py \
        --source-dir /data1/training_data/sourceFile \
        --output-dir /data1/training_data/lerobot_exports \
        --groups G3 G4 \
        --arm-mode left \
        --action-type ee \
        --video-storage video \
        --dataset-name g2_left_arm_ee_g3_g4

    # 5) Faster video export on machines with NVIDIA encoder support.
    # This speeds up output video encoding, but does not remove source decode cost.
    python create_g2_dataset_using_lerobot.py \
        --source-dir /data1/training_data/sourceFile \
        --output-dir /data1/training_data/lerobot_exports \
        --groups G1 G2 G3 G4 G5 G6 G7 \
        --arm-mode dual \
        --action-type joint \
        --video-storage video \
        --vcodec h264_nvenc \
        --dataset-name g2_merged_nvenc
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import (
    VideoDecoderCache,
    decode_video_frames,
    decode_video_frames_torchcodec,
    get_safe_default_codec,
)


FPS = 30
CAMERA_SPECS = {
    "hand_left_color": (1056, 1280, 3),
    "hand_right_color": (1056, 1280, 3),
    "head_color": (400, 640, 3),
}
JOINT_NAMES = [
    *(f"l.joint{i}.pos" for i in range(1, 8)),
    "l.gripper.pos",
    *(f"r.joint{i}.pos" for i in range(1, 8)),
    "r.gripper.pos",
]
LEFT_JOINT_NAMES = JOINT_NAMES[:8]
RIGHT_JOINT_NAMES = JOINT_NAMES[8:]
EE_NAMES = [
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
LEFT_EE_NAMES = EE_NAMES[:7]
RIGHT_EE_NAMES = EE_NAMES[7:]

ARM_MODE_ALIASES = {
    "dual": "dual",
    "both": "dual",
    "bimanual": "dual",
    "left": "left",
    "l": "left",
    "right": "right",
    "r": "right",
}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_visual_features(storage: str) -> dict[str, dict[str, Any]]:
    dtype = "video" if storage == "video" else "image"
    return {
        f"observation.images.{camera}": {
            "dtype": dtype,
            "shape": shape,
            "names": ["height", "width", "channels"],
        }
        for camera, shape in CAMERA_SPECS.items()
    }


def normalize_arm_mode(arm_mode: str) -> str:
    normalized = ARM_MODE_ALIASES.get(arm_mode.lower())
    if normalized is None:
        raise ValueError(f"Unsupported arm_mode: {arm_mode}")
    return normalized


def get_joint_names_for_arm_mode(arm_mode: str) -> list[str]:
    arm_mode = normalize_arm_mode(arm_mode)
    if arm_mode == "left":
        return LEFT_JOINT_NAMES
    if arm_mode == "right":
        return RIGHT_JOINT_NAMES
    return JOINT_NAMES


def get_ee_names_for_arm_mode(arm_mode: str) -> list[str]:
    arm_mode = normalize_arm_mode(arm_mode)
    if arm_mode == "left":
        return LEFT_EE_NAMES
    if arm_mode == "right":
        return RIGHT_EE_NAMES
    return EE_NAMES


def build_joint_features(storage: str, arm_mode: str) -> dict[str, dict[str, Any]]:
    joint_names = get_joint_names_for_arm_mode(arm_mode)
    return {
        "subtask_index": {
            "dtype": "int64",
            "shape": (1,),
            "names": None,
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (len(joint_names),),
            "names": joint_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(joint_names),),
            "names": joint_names,
        },
        **build_visual_features(storage),
    }


def build_ee_features(storage: str, arm_mode: str) -> dict[str, dict[str, Any]]:
    ee_names = get_ee_names_for_arm_mode(arm_mode)
    return {
        "subtask_index": {
            "dtype": "int64",
            "shape": (1,),
            "names": None,
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (len(ee_names),),
            "names": ee_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(ee_names),),
            "names": ee_names,
        },
        **build_visual_features(storage),
    }


def find_task_folders(source_dir: Path) -> dict[str, list[Path]]:
    task_folders: dict[str, list[Path]] = {}
    for item in sorted(source_dir.iterdir()):
        if not item.is_dir():
            continue
        try:
            episode_dirs = [sub for sub in sorted(item.iterdir()) if sub.is_dir() and (sub / "metaInfo.json").exists()]
        except FileNotFoundError:
            continue
        if not episode_dirs:
            continue

        try:
            with open(episode_dirs[0] / "metaInfo.json", "r", encoding="utf-8") as f:
                meta_info = json.load(f)
            task_name = meta_info.get("taskName", "") or item.name
            group_name = task_name.split("-")[0] if "-" in task_name else task_name
            task_folders.setdefault(group_name, []).append(item)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Could not determine task name for %s: %s", item, exc)
            task_folders.setdefault(item.name, []).append(item)
    return task_folders


def clean_step_text(step: str) -> str:
    step = step.strip()
    step = re.sub(r"^\s*\d+\.\s*", "", step)
    return step.strip()


def parse_task_step(raw_task_step: Any) -> list[str]:
    if raw_task_step is None:
        return []

    if isinstance(raw_task_step, list):
        return [clean_step_text(str(step)) for step in raw_task_step if str(step).strip()]

    raw_text = str(raw_task_step).strip()
    if not raw_text:
        return []

    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, list):
            return [clean_step_text(str(step)) for step in parsed if str(step).strip()]
    except json.JSONDecodeError:
        pass

    return [clean_step_text(raw_text)]


def make_subtask_text(meta_info: dict[str, Any]) -> str:
    steps = parse_task_step(meta_info.get("taskStep"))
    if steps:
        return "；".join(steps)
    task_desc = str(meta_info.get("taskDesc", "")).strip()
    if task_desc:
        return task_desc
    return str(meta_info.get("taskName", "unknown")).strip() or "unknown"


def make_task_text(meta_info: dict[str, Any]) -> str:
    task_desc = str(meta_info.get("taskDesc", "")).strip()
    if task_desc:
        return task_desc
    task_name = str(meta_info.get("taskName", "")).strip()
    if task_name:
        return task_name
    return "unknown"


def load_episode_data(episode_path: Path) -> pd.DataFrame:
    parquet_files = sorted(episode_path.glob("data/**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {episode_path}")

    df = pd.read_parquet(parquet_files[0])
    if df.empty:
        raise ValueError(f"Episode parquet is empty: {parquet_files[0]}")

    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
    elif "index" in df.columns:
        df = df.sort_values("index").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    return df


def quaternion_to_axis_angle(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64)
    norm = np.linalg.norm(quat_xyzw)
    if norm < 1e-8:
        return np.zeros(3, dtype=np.float32)

    q = quat_xyzw / norm
    if q[3] < 0:
        q = -q

    angle = 2.0 * np.arccos(np.clip(q[3], -1.0, 1.0))
    sin_half = np.sqrt(max(1.0 - q[3] * q[3], 0.0))
    if sin_half < 1e-8:
        rotvec = 2.0 * q[:3]
    else:
        rotvec = angle * (q[:3] / sin_half)
    return rotvec.astype(np.float32)


def ensure_float32_vector(value: Any, expected_len: int, field_name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape != (expected_len,):
        raise ValueError(f"{field_name} expected shape ({expected_len},), got {arr.shape}")
    return arr


def get_first_present(row: pd.Series, keys: list[str], field_name: str, required: bool = True) -> Any:
    for key in keys:
        if key in row.index and row[key] is not None:
            return row[key]
    if required:
        raise KeyError(f"Missing {field_name}. Tried columns: {keys}")
    return None


def extract_gripper_pair(row: pd.Series, prefix: str) -> np.ndarray:
    raw = get_first_present(row, [f"{prefix}.effector.position"], f"{prefix} gripper pair")
    return ensure_float32_vector(raw, 2, f"{prefix}.effector.position")


def extract_joint_vector(row: pd.Series, prefix: str) -> np.ndarray:
    raw = get_first_present(row, [f"{prefix}.joint.position"], f"{prefix} dual-arm joint positions")
    joints = ensure_float32_vector(raw, 14, f"{prefix}.joint.position")
    grippers = extract_gripper_pair(row, prefix)
    return np.concatenate([joints[:7], grippers[:1], joints[7:14], grippers[1:2]]).astype(np.float32)


def select_arm_slice(vector: np.ndarray, arm_mode: str, per_arm_dim: int) -> np.ndarray:
    arm_mode = normalize_arm_mode(arm_mode)
    if arm_mode == "left":
        return vector[:per_arm_dim].astype(np.float32)
    if arm_mode == "right":
        return vector[per_arm_dim : 2 * per_arm_dim].astype(np.float32)
    return vector.astype(np.float32)


def vector_to_dual_rotvec(vec: np.ndarray, field_name: str) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    if vec.shape == (12,):
        return vec.astype(np.float32)
    if vec.shape == (14,):
        left = np.concatenate([vec[:3], quaternion_to_axis_angle(vec[3:7])])
        right = np.concatenate([vec[7:10], quaternion_to_axis_angle(vec[10:14])])
        return np.concatenate([left, right]).astype(np.float32)
    raise ValueError(f"{field_name} must be length 12 (rotvec) or 14 (quaternion), got {vec.shape}")


def extract_ee_vector(row: pd.Series, prefix: str) -> np.ndarray:
    raw = get_first_present(row, [f"{prefix}.end.position"], f"{prefix} dual-arm EE pose")
    ee_vec = vector_to_dual_rotvec(np.asarray(raw), f"{prefix}.end.position")
    grippers = extract_gripper_pair(row, prefix)
    return np.concatenate([ee_vec[:6], grippers[:1], ee_vec[6:12], grippers[1:2]]).astype(np.float32)


def to_uint8_image(array: np.ndarray, expected_shape: tuple[int, int, int], field_name: str) -> np.ndarray:
    image = np.asarray(array)
    if image.ndim != 3:
        raise ValueError(f"{field_name} expected 3 dimensions, got shape {image.shape}")

    if image.shape == expected_shape:
        out = image
    elif image.shape == (expected_shape[2], expected_shape[0], expected_shape[1]):
        out = np.transpose(image, (1, 2, 0))
    else:
        raise ValueError(f"{field_name} expected shape {expected_shape} or {(expected_shape[2], expected_shape[0], expected_shape[1])}, got {image.shape}")

    if np.issubdtype(out.dtype, np.floating):
        max_value = float(np.nanmax(out)) if out.size else 0.0
        if max_value <= 1.0:
            out = np.clip(out, 0.0, 1.0) * 255.0
        out = np.clip(out, 0.0, 255.0).astype(np.uint8)
    else:
        out = np.clip(out, 0, 255).astype(np.uint8)

    return out


def decode_image_bytes(blob: bytes, expected_shape: tuple[int, int, int], field_name: str) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError(f"Pillow is required to decode image bytes for {field_name}") from exc

    with Image.open(io.BytesIO(blob)) as image:
        rgb = image.convert("RGB")
        arr = np.asarray(rgb)
    return to_uint8_image(arr, expected_shape, field_name)


def normalize_image_value(value: Any, expected_shape: tuple[int, int, int], field_name: str) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return to_uint8_image(value, expected_shape, field_name)
    if isinstance(value, (list, tuple)):
        return to_uint8_image(np.asarray(value), expected_shape, field_name)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return decode_image_bytes(bytes(value), expected_shape, field_name)
    if isinstance(value, str):
        return decode_image_bytes(Path(value).read_bytes(), expected_shape, field_name)
    raise TypeError(f"Unsupported image value type for {field_name}: {type(value)}")


@dataclass
class EpisodeVisualSource:
    episode_path: Path
    df_columns: list[str]
    fps: int
    column_sources: dict[str, str]
    video_sources: dict[str, Path]
    video_backend: str
    decoder_cache: VideoDecoderCache | None = None

    @classmethod
    def discover(cls, episode_path: Path, df: pd.DataFrame, fps: int) -> "EpisodeVisualSource":
        column_sources: dict[str, str] = {}
        video_sources: dict[str, Path] = {}
        columns = list(df.columns)

        for camera in CAMERA_SPECS:
            video_path = find_camera_video_file(episode_path, camera)
            if video_path is not None:
                video_sources[camera] = video_path
                continue

            column_candidates = [
                f"observation.images.{camera}",
                f"observation.image.{camera}",
                f"images.{camera}",
                camera,
            ]
            source_col = next((col for col in column_candidates if col in df.columns), None)
            if source_col is not None:
                column_sources[camera] = source_col
                continue

            raise FileNotFoundError(
                f"Could not locate visual source for camera '{camera}' in episode {episode_path}. "
                f"Tried parquet columns {column_candidates} and episode-local video files."
            )

        return cls(
            episode_path=episode_path,
            df_columns=columns,
            fps=fps,
            column_sources=column_sources,
            video_sources=video_sources,
            video_backend=get_safe_default_codec(),
            decoder_cache=VideoDecoderCache(),
        )

    def get_frame(self, camera: str, row: pd.Series, source_timestamp: float) -> np.ndarray:
        expected_shape = CAMERA_SPECS[camera]
        if camera in self.column_sources:
            col = self.column_sources[camera]
            return normalize_image_value(row[col], expected_shape, f"{self.episode_path}:{col}")

        video_path = self.video_sources[camera]
        timestamp = float(source_timestamp)
        tolerance_s = 0.5 / float(self.fps)
        if self.video_backend == "torchcodec":
            frames = decode_video_frames_torchcodec(
                video_path=video_path,
                timestamps=[timestamp],
                tolerance_s=tolerance_s,
                decoder_cache=self.decoder_cache,
            )
        else:
            frames = decode_video_frames(
                video_path=video_path,
                timestamps=[timestamp],
                tolerance_s=tolerance_s,
                backend=self.video_backend,
            )
        frame = frames[0].permute(1, 2, 0).cpu().numpy()
        return to_uint8_image(frame, expected_shape, f"{video_path}@{timestamp:.3f}s")

    def close(self) -> None:
        if self.decoder_cache is not None:
            self.decoder_cache.clear()


def find_camera_video_file(episode_path: Path, camera: str) -> Path | None:
    exact_candidates = [
        episode_path / "videos" / "chunk-000" / f"observation.images.{camera}" / "episode_000000.mp4",
        episode_path / "videos" / "chunk-000" / f"observation.images.{camera}" / "episode_000000.avi",
        episode_path / "videos" / "chunk-000" / f"observation.images.{camera}" / "episode_000000.mov",
        episode_path / "videos" / "chunk-000" / f"observation.images.{camera}" / "episode_000000.mkv",
    ]
    for candidate in exact_candidates:
        if candidate.is_file():
            return candidate

    matches: list[Path] = []
    for suffix in (".mp4", ".avi", ".mov", ".mkv"):
        for path in episode_path.rglob(f"*{suffix}"):
            path_str = str(path)
            if camera in path_str or f"observation.images.{camera}" in path_str or camera in path.parent.name:
                matches.append(path)

    unique_matches = sorted({path for path in matches if path.is_file()})
    return unique_matches[0] if unique_matches else None


def build_frame(
    row: pd.Series,
    task_text: str,
    subtask_index: int,
    action_type: str,
    arm_mode: str,
    visuals: EpisodeVisualSource,
) -> dict[str, Any]:
    frame: dict[str, Any] = {
        "task": task_text,
        "subtask_index": np.array([subtask_index], dtype=np.int64),
    }
    source_timestamp = float(row["timestamp"]) if "timestamp" in row.index else 0.0

    if action_type == "joint":
        frame["observation.state"] = select_arm_slice(
            extract_joint_vector(row, "observation.state"), arm_mode=arm_mode, per_arm_dim=8
        )
        frame["action"] = select_arm_slice(extract_joint_vector(row, "action"), arm_mode=arm_mode, per_arm_dim=8)
    elif action_type == "ee":
        frame["observation.state"] = select_arm_slice(
            extract_ee_vector(row, "observation.state"), arm_mode=arm_mode, per_arm_dim=7
        )
        frame["action"] = select_arm_slice(extract_ee_vector(row, "action"), arm_mode=arm_mode, per_arm_dim=7)
    else:
        raise ValueError(f"Unsupported action_type: {action_type}")

    with ThreadPoolExecutor(max_workers=len(CAMERA_SPECS)) as executor:
        future_to_camera = {
            executor.submit(visuals.get_frame, camera, row, source_timestamp): camera
            for camera in CAMERA_SPECS
        }
        for future, camera in [(future, future_to_camera[future]) for future in future_to_camera]:
            frame[f"observation.images.{camera}"] = future.result()

    return frame


def process_episode_to_dataset(
    dataset: LeRobotDataset,
    episode_path: Path,
    df: pd.DataFrame,
    task_text: str,
    subtask_index: int,
    action_type: str,
    arm_mode: str,
    fps: int,
) -> None:
    visuals = EpisodeVisualSource.discover(episode_path, df, fps)
    try:
        for frame_index, (_, row) in enumerate(df.iterrows()):
            frame = build_frame(
                row=row,
                task_text=task_text,
                subtask_index=subtask_index,
                action_type=action_type,
                arm_mode=arm_mode,
                visuals=visuals,
            )
            dataset.add_frame(frame)
        dataset.save_episode()
    finally:
        visuals.close()


def get_resume_state_path(dataset_path: Path) -> Path:
    return dataset_path / "meta" / "export_resume_state.json"


def build_export_config(
    source_dir: Path,
    groups: list[str],
    arm_mode: str,
    action_type: str,
    video_storage: str,
    fps: int,
    vcodec: str,
    max_episodes_per_group: int | None,
) -> dict[str, Any]:
    return {
        "source_dir": str(source_dir.resolve()),
        "groups": list(groups),
        "arm_mode": arm_mode,
        "action_type": action_type,
        "video_storage": video_storage,
        "fps": fps,
        "vcodec": vcodec,
        "max_episodes_per_group": max_episodes_per_group,
    }


def load_resume_state(state_path: Path) -> dict[str, Any]:
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    if not isinstance(state, dict):
        raise ValueError(f"Invalid resume state format in {state_path}")
    return state


def save_resume_state(
    state_path: Path,
    export_config: dict[str, Any],
    completed_episodes: set[str],
    subtask_to_index: dict[str, int],
) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "export_config": export_config,
        "completed_episodes": sorted(completed_episodes),
        "subtask_to_index": subtask_to_index,
    }
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2, sort_keys=True)


def create_g2_dataset(
    source_dir: Path,
    output_dir: Path,
    groups: list[str],
    arm_mode: str,
    action_type: str,
    dataset_name: str,
    video_storage: str,
    fps: int,
    vcodec: str,
    max_episodes_per_group: int | None,
    resume: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    arm_mode = normalize_arm_mode(arm_mode)

    logging.info("Scanning %s for task folders...", source_dir)
    task_folders = find_task_folders(source_dir)
    logging.info("Found %d task groups: %s", len(task_folders), list(task_folders.keys()))

    selected_folders = {group: task_folders[group] for group in groups if group in task_folders}
    missing_groups = [group for group in groups if group not in task_folders]
    for group in missing_groups:
        logging.warning("Group %s not found in task folders", group)
    if not selected_folders:
        raise ValueError(f"No selected groups found. Available groups: {list(task_folders.keys())}")

    episodes_by_group: dict[str, list[Path]] = {}
    for group_name, task_folder_list in selected_folders.items():
        episode_folders: list[Path] = []
        for task_folder in task_folder_list:
            episode_folders.extend(
                [item for item in sorted(task_folder.iterdir()) if item.is_dir() and (item / "metaInfo.json").exists()]
            )
        if max_episodes_per_group is not None:
            episode_folders = episode_folders[:max_episodes_per_group]
        episodes_by_group[group_name] = episode_folders

    total_selected_episodes = sum(len(episode_folders) for episode_folders in episodes_by_group.values())
    logging.info("Selected %d episodes across %d groups", total_selected_episodes, len(episodes_by_group))

    features_spec = (
        build_ee_features(video_storage, arm_mode)
        if action_type == "ee"
        else build_joint_features(video_storage, arm_mode)
    )

    subtask_to_index: dict[str, int] = {}
    completed_episodes: set[str] = set()

    dataset_path = output_dir / dataset_name
    state_path = get_resume_state_path(dataset_path)
    export_config = build_export_config(
        source_dir=source_dir,
        groups=groups,
        arm_mode=arm_mode,
        action_type=action_type,
        video_storage=video_storage,
        fps=fps,
        vcodec=vcodec,
        max_episodes_per_group=max_episodes_per_group,
    )

    if resume:
        if not dataset_path.exists():
            raise FileNotFoundError(f"Cannot resume: dataset path does not exist: {dataset_path}")
        if not state_path.exists():
            raise FileNotFoundError(
                f"Cannot resume without checkpoint state: {state_path}. "
                "This dataset was likely created before resume support was added."
            )
        state = load_resume_state(state_path)
        saved_config = state.get("export_config")
        if saved_config != export_config:
            raise ValueError(
                "Resume configuration does not match the existing export state.\n"
                f"Expected: {saved_config}\n"
                f"Current:  {export_config}"
            )
        completed_episodes = set(state.get("completed_episodes", []))
        subtask_to_index = {
            str(key): int(value) for key, value in state.get("subtask_to_index", {}).items()
        }
        logging.info(
            "Resuming dataset export from %s with %d completed episodes",
            dataset_path,
            len(completed_episodes),
        )
        dataset = LeRobotDataset.resume(
            repo_id=dataset_name,
            root=dataset_path,
            vcodec=vcodec,
        )
        total_episodes = int(dataset.meta.total_episodes)
        total_frames = int(dataset.meta.total_frames)
    else:
        if dataset_path.exists():
            logging.warning("Dataset path %s already exists, removing...", dataset_path)
            shutil.rmtree(dataset_path)

        dataset = LeRobotDataset.create(
            repo_id=dataset_name,
            fps=fps,
            features=features_spec,
            root=dataset_path,
            robot_type="g2",
            use_videos=(video_storage == "video"),
            vcodec=vcodec,
            streaming_encoding=False,
            metadata_buffer_size=1,
        )
        save_resume_state(state_path, export_config, completed_episodes, subtask_to_index)
        total_episodes = 0
        total_frames = 0
    overall_progress = tqdm(
        total=total_selected_episodes,
        desc="Processing episodes",
        unit="episode",
        dynamic_ncols=True,
    )

    for group_name, task_folder_list in selected_folders.items():
        episode_folders = episodes_by_group[group_name]
        logging.info(
            "Processing group %s with %d episodes from %d task folders",
            group_name,
            len(episode_folders),
            len(task_folder_list),
        )

        for episode_folder in episode_folders:
            episode_key = str(episode_folder.resolve().relative_to(source_dir.resolve()))
            if episode_key in completed_episodes:
                overall_progress.update(1)
                overall_progress.set_postfix(group=group_name, frames=total_frames, refresh=False)
                continue
            try:
                with open(episode_folder / "metaInfo.json", "r", encoding="utf-8") as f:
                    meta_info = json.load(f)
                task_text = make_task_text(meta_info)
                subtask_text = make_subtask_text(meta_info)
                if subtask_text not in subtask_to_index:
                    subtask_to_index[subtask_text] = len(subtask_to_index)
                df = load_episode_data(episode_folder)
                process_episode_to_dataset(
                    dataset=dataset,
                    episode_path=episode_folder,
                    df=df,
                    task_text=task_text,
                    subtask_index=subtask_to_index[subtask_text],
                    action_type=action_type,
                    arm_mode=arm_mode,
                    fps=fps,
                )
                total_episodes += 1
                total_frames += len(df)
                completed_episodes.add(episode_key)
                save_resume_state(state_path, export_config, completed_episodes, subtask_to_index)
            except Exception as exc:  # noqa: BLE001
                logging.error("Error processing episode %s: %s", episode_folder, exc)
                dataset.clear_episode_buffer()
            finally:
                overall_progress.update(1)
                overall_progress.set_postfix(group=group_name, frames=total_frames, refresh=False)

    overall_progress.close()

    logging.info("Finalizing dataset...")
    dataset.finalize()
    write_subtasks_metadata(dataset_path, subtask_to_index)
    save_resume_state(state_path, export_config, completed_episodes, subtask_to_index)

    logging.info("Dataset created successfully")
    logging.info("Total episodes: %d", total_episodes)
    logging.info("Total frames: %d", total_frames)
    logging.info("Dataset saved to: %s", dataset_path)

    try:
        logging.info("Verifying dataset can be loaded...")
        verified = LeRobotDataset(repo_id=dataset_name, root=dataset_path)
        logging.info(
            "Verification successful: %d episodes, %d frames",
            verified.num_episodes,
            verified.num_frames,
        )
    except Exception as exc:  # noqa: BLE001
        logging.warning("Dataset verification failed: %s", exc)


def write_subtasks_metadata(dataset_path: Path, subtask_to_index: dict[str, int]) -> None:
    if not subtask_to_index:
        return

    subtasks = sorted(subtask_to_index.items(), key=lambda item: item[1])
    subtasks_df = pd.DataFrame(
        {"subtask_index": [index for _, index in subtasks]},
        index=pd.Index([name for name, _ in subtasks], name="subtask"),
    )
    subtasks_path = dataset_path / "meta" / "subtasks.parquet"
    subtasks_path.parent.mkdir(parents=True, exist_ok=True)
    subtasks_df.to_parquet(subtasks_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a G2 LeRobot v3.0 dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  1. Smoke test, small output, easiest for validation:
     python create_g2_dataset_using_lerobot.py \\
         --output-dir /tmp/g2_exports \\
         --groups G7 \\
         --max-episodes-per-group 2 \\
         --video-storage image \\
         --dataset-name g2_g7_smoke

  2. Full joint-space export with videos:
     python create_g2_dataset_using_lerobot.py \\
         --output-dir /data1/training_data/lerobot_exports \\
         --groups G1 G2 G3 G4 G5 G6 G7 \\
         --arm-mode dual \\
         --action-type joint \\
         --video-storage video \\
         --dataset-name g2_merged

  3. Right-arm-only export:
     python create_g2_dataset_using_lerobot.py \\
         --output-dir /data1/training_data/lerobot_exports \\
         --groups G7 \\
         --arm-mode right \\
         --action-type joint \\
         --video-storage image \\
         --dataset-name g2_right_arm_g7

  4. Faster video encoding on NVIDIA GPUs:
     python create_g2_dataset_using_lerobot.py \\
         --output-dir /data1/training_data/lerobot_exports \\
         --groups G1 G2 G3 G4 G5 G6 G7 \\
         --arm-mode dual \\
         --video-storage video \\
         --vcodec h264_nvenc \\
         --dataset-name g2_merged_nvenc

Notes:
  - task comes from metaInfo.taskDesc
  - subtask_index is stored in frame data
  - meta/subtasks.parquet is generated from metaInfo.taskStep
  - arm-mode controls whether state/action are exported as left-only, right-only, or dual-arm vectors
  - image mode is convenient for debugging; video mode is the normal full-export choice
""",
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default="/data1/training_data/sourceFile",
        help="Source directory containing raw G2 episodes",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where the output LeRobot dataset folder will be created",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["G1", "G2", "G3", "G4", "G5", "G6", "G7"],
        help="Task groups to include, for example: G1 G2 G7",
    )
    parser.add_argument(
        "--arm-mode",
        choices=sorted(ARM_MODE_ALIASES.keys()),
        default="dual",
        help="Export left arm only, right arm only, or both arms",
    )
    parser.add_argument(
        "--action-type",
        choices=["joint", "ee"],
        default="ee",
        help="Export joint-space or end-effector state/action representation",
    )
    parser.add_argument(
        "--video-storage",
        choices=["video", "image"],
        default="video",
        help="Store visual features as output videos or as image features",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="g2_merged",
        help="Name of the output dataset directory and repo_id",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=FPS,
        help="Dataset FPS used for output timestamps and timestamp-based video frame lookup",
    )
    parser.add_argument(
        "--vcodec",
        type=str,
        default="libsvtav1",
        help="Video codec used when --video-storage=video, for example libsvtav1 or h264_nvenc",
    )
    parser.add_argument(
        "--max-episodes-per-group",
        type=int,
        default=None,
        help="Optional cap per group, useful for smoke tests and partial exports",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previously interrupted export using the checkpoint state stored under meta/.",
    )

    args = parser.parse_args()
    setup_logging()

    create_g2_dataset(
        source_dir=Path(args.source_dir),
        output_dir=Path(args.output_dir),
        groups=args.groups,
        arm_mode=args.arm_mode,
        action_type=args.action_type,
        dataset_name=args.dataset_name,
        video_storage=args.video_storage,
        fps=args.fps,
        vcodec=args.vcodec,
        max_episodes_per_group=args.max_episodes_per_group,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
