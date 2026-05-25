# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""
Replay a native AgiBot/G2 episode parquet on the G2 robot by directly driving every
joint (waist + head + both arms + both grippers — 24 joints total). The chassis is
intentionally not controlled.

Unlike :mod:`lerobot.scripts.replay_g2_dataset`, this script does NOT read a
LeRobotDataset and does NOT run inverse kinematics. The native parquet already
contains absolute joint targets in radians (grippers in normalized [0, 1]).

Example::

    uv run python -m lerobot.scripts.replay_g2_native_dataset \
        --root=~/replay-data/g2_min_replay_cases/G1-1/episode_000000

    # half speed, skip waist/head:
    uv run python -m lerobot.scripts.replay_g2_native_dataset \
        --root=~/replay-data/g2_min_replay_cases/G2-1/episode_000000 \
        --speed_factor=0.5 --use_head=false --use_waist=false

    # dry-run (no robot connection, just validate the parquet + targets):
    uv run python -m lerobot.scripts.replay_g2_native_dataset \
        --root=~/replay-data/g2_min_replay_cases/G1-1/episode_000000 --dry_run=true
"""

import json
import logging
import math
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

import numpy as np
import pyarrow.parquet as pq
from pynput import keyboard

from lerobot.configs import parser
from lerobot.robots.g2.g2_constants import (
    G2_GRIPPER_JOINT_NAMES,
    LEFT_ARM_JOINT_NAMES,
    RIGHT_ARM_JOINT_NAMES,
    g2_gripper_normalized_to_rad,
)
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say

WAIST_JOINT_NAMES: list[str] = [
    "idx01_body_joint1",
    "idx02_body_joint2",
    "idx03_body_joint3",
    "idx04_body_joint4",
    "idx05_body_joint5",
]

HEAD_JOINT_NAMES: list[str] = [
    "idx11_head_joint1",
    "idx12_head_joint2",
    "idx13_head_joint3",
]

GRIPPER_JOINT_NAMES: list[str] = [
    G2_GRIPPER_JOINT_NAMES["left"],
    G2_GRIPPER_JOINT_NAMES["right"],
]

# Limits (rad) — pulled verbatim from the GDK control documentation.
JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "idx01_body_joint1": (-1.082104, 0.000174),
    "idx02_body_joint2": (-0.000174, 2.652900),
    "idx03_body_joint3": (-1.919862, 1.570970),
    "idx04_body_joint4": (-0.436332, 0.436332),
    "idx05_body_joint5": (-3.045599, 3.045599),
    "idx11_head_joint1": (-1.570970, 1.570970),
    "idx12_head_joint2": (-0.349240, 0.349240),
    "idx13_head_joint3": (-0.534773, 0.534773),
    "idx21_arm_l_joint1": (-3.071796, 3.071796),
    "idx22_arm_l_joint2": (-2.059505, 2.059505),
    "idx23_arm_l_joint3": (-3.071796, 3.071796),
    "idx24_arm_l_joint4": (-2.495838, 1.012308),
    "idx25_arm_l_joint5": (-3.071796, 3.071796),
    "idx26_arm_l_joint6": (-1.012308, 1.012308),
    "idx27_arm_l_joint7": (-1.535907, 1.535907),
    "idx61_arm_r_joint1": (-3.071796, 3.071796),
    "idx62_arm_r_joint2": (-2.059505, 2.059505),
    "idx63_arm_r_joint3": (-3.071796, 3.071796),
    "idx64_arm_r_joint4": (-2.495838, 1.012308),
    "idx65_arm_r_joint5": (-3.071796, 3.071796),
    "idx66_arm_r_joint6": (-1.012308, 1.012308),
    "idx67_arm_r_joint7": (-1.535907, 1.535907),
    G2_GRIPPER_JOINT_NAMES["left"]: (-0.785, 0.0),
    G2_GRIPPER_JOINT_NAMES["right"]: (-0.785, 0.0),
}


@dataclass
class G2NativeReplayConfig:
    # Episode directory (auto-discovers data/chunk-000/episode_000000.parquet) or a
    # direct path to the parquet file.
    root: str | Path

    speed_factor: float = 1.0
    start_frame: int = 0

    # Calibrate-to-first-frame: velocity-capped, dynamically sized.
    calibrate_start: bool = True
    calibrate_rate_hz: float = 100.0
    calibrate_max_joint_speed_rad_s: float = 0.15
    calibrate_max_gripper_speed_rad_s: float = 0.6
    calibrate_min_duration_s: float = 2.0
    calibrate_max_duration_s: float = 20.0
    calibrate_settle_s: float = 0.3

    # Per-step delta caps during replay (sent every 1/effective_fps).
    max_step_joint_delta_rad: float = 0.05
    max_step_gripper_delta_rad: float = 0.1

    control_period: float = 0.012
    enable_low_latency: bool = False

    use_gripper: bool = True
    use_head: bool = True
    use_waist: bool = True

    play_sounds: bool = False
    init_delay_s: float = 2.0

    # Skip the GDK lifecycle and only validate the parquet + targets.
    dry_run: bool = False


@dataclass
class _EpisodeData:
    waist_rad: np.ndarray | None   # (N, 5) or None
    head_rad: np.ndarray | None    # (N, 3) or None
    arm_rad: np.ndarray            # (N, 14)
    gripper_norm: np.ndarray | None  # (N, 2) or None
    timestamps: np.ndarray         # (N,)
    fps: float
    meta: dict


def _resolve_parquet(root: Path) -> tuple[Path, Path | None]:
    """Return (parquet_path, meta_info_path_or_None) given an episode dir or parquet."""
    if root.is_file() and root.suffix == ".parquet":
        meta_candidate = root.parent.parent.parent / "metaInfo.json"
        return root, meta_candidate if meta_candidate.exists() else None

    if not root.is_dir():
        raise FileNotFoundError(f"Episode root not found: {root}")

    # Native layout: <root>/data/chunk-000/episode_000000.parquet
    chunk_dir = root / "data" / "chunk-000"
    if not chunk_dir.is_dir():
        raise FileNotFoundError(
            f"Expected '{chunk_dir}' under episode root; is this a native G2 dataset directory?"
        )
    parquets = sorted(chunk_dir.glob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No parquet found in {chunk_dir}")
    if len(parquets) > 1:
        logging.warning("Multiple parquet files in %s; using %s", chunk_dir, parquets[0].name)
    meta_path = root / "metaInfo.json"
    return parquets[0], meta_path if meta_path.exists() else None


def _load_episode(cfg: G2NativeReplayConfig) -> _EpisodeData:
    root = Path(cfg.root).expanduser()
    parquet_path, meta_path = _resolve_parquet(root)
    logging.info("Loading parquet: %s", parquet_path)

    table = pq.read_table(parquet_path).to_pandas()
    n_frames = len(table)
    if n_frames == 0:
        raise ValueError(f"Empty parquet: {parquet_path}")

    def _stack(col: str, expected_dim: int) -> np.ndarray:
        if col not in table.columns:
            raise KeyError(f"Required column '{col}' missing in {parquet_path}")
        arr = np.stack([np.asarray(v, dtype=np.float64) for v in table[col].to_numpy()])
        if arr.shape != (n_frames, expected_dim):
            raise ValueError(
                f"Column '{col}' has shape {arr.shape}, expected ({n_frames}, {expected_dim})"
            )
        return arr

    arm_rad = _stack("action.joint.position", 14)
    waist_rad = _stack("action.waist.position", 5) if cfg.use_waist else None
    head_rad = _stack("action.head.position", 3) if cfg.use_head else None
    gripper_norm = _stack("action.effector.position", 2) if cfg.use_gripper else None

    timestamps = (
        table["timestamp"].to_numpy(dtype=np.float64)
        if "timestamp" in table.columns
        else np.arange(n_frames, dtype=np.float64) / 30.0
    )
    if n_frames >= 2:
        diffs = np.diff(timestamps)
        median_dt = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 1 / 30.0
        fps = 1.0 / median_dt if median_dt > 0 else 30.0
    else:
        fps = 30.0

    meta: dict = {}
    if meta_path is not None:
        try:
            meta = json.loads(meta_path.read_text())
        except Exception as e:
            logging.warning("Failed to parse %s: %s", meta_path, e)

    return _EpisodeData(
        waist_rad=waist_rad,
        head_rad=head_rad,
        arm_rad=arm_rad,
        gripper_norm=gripper_norm,
        timestamps=timestamps,
        fps=fps,
        meta=meta,
    )


def _build_joint_name_list(cfg: G2NativeReplayConfig) -> list[str]:
    names: list[str] = []
    if cfg.use_waist:
        names += WAIST_JOINT_NAMES
    if cfg.use_head:
        names += HEAD_JOINT_NAMES
    names += LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES
    if cfg.use_gripper:
        names += GRIPPER_JOINT_NAMES
    return names


def _frame_targets(
    cfg: G2NativeReplayConfig, episode: _EpisodeData, idx: int
) -> tuple[np.ndarray, list[bool]]:
    """Return (targets_rad, is_gripper_mask) for frame `idx`, in joint-name order."""
    parts: list[np.ndarray] = []
    grip_flags: list[bool] = []
    if cfg.use_waist:
        parts.append(episode.waist_rad[idx])
        grip_flags += [False] * 5
    if cfg.use_head:
        parts.append(episode.head_rad[idx])
        grip_flags += [False] * 3
    parts.append(episode.arm_rad[idx])
    grip_flags += [False] * 14
    if cfg.use_gripper:
        eff = episode.gripper_norm[idx]
        parts.append(np.array([g2_gripper_normalized_to_rad(eff[0]), g2_gripper_normalized_to_rad(eff[1])]))
        grip_flags += [True] * 2
    return np.concatenate(parts), grip_flags


def _validate_targets(names: list[str], targets: np.ndarray) -> list[str]:
    """Return list of offending joint names (empty if all in range)."""
    offenders: list[str] = []
    for name, value in zip(names, targets, strict=True):
        lo, hi = JOINT_LIMITS[name]
        if not (lo <= float(value) <= hi):
            offenders.append(f"{name}={float(value):.4f} (limits [{lo:.4f}, {hi:.4f}])")
    return offenders


def _clip_step(
    raw: np.ndarray, prev: np.ndarray, grip_flags: list[bool],
    max_joint_delta: float, max_gripper_delta: float,
) -> tuple[np.ndarray, list[tuple[int, float, float]]]:
    """Clip per-step delta. Returns (clipped_targets, list of (index, requested_Δ, allowed_Δ))."""
    delta = raw - prev
    clipped: list[tuple[int, float, float]] = []
    out = raw.copy()
    for i, (d, is_grip) in enumerate(zip(delta, grip_flags, strict=True)):
        cap = max_gripper_delta if is_grip else max_joint_delta
        if abs(d) > cap:
            allowed = math.copysign(cap, d)
            out[i] = prev[i] + allowed
            clipped.append((i, float(d), float(allowed)))
    return out, clipped


def _read_gripper_rad_via_end_state(robot) -> dict[str, float]:
    """Best-effort read of omnipicker gripper joint angles from ``get_end_state``.

    Returns a mapping ``{joint_name: rad}`` for whichever side(s) reported a position.
    Grippers reach us via this API on some GDK builds even though the arm joints
    are reported through ``get_joint_states``; we treat the ``position`` field as
    rad (matching the [-0.785, 0] servo range documented for omnipicker).
    """
    out: dict[str, float] = {}
    try:
        end_state = robot.get_end_state()
    except Exception as e:
        logging.warning("get_end_state() failed during readback: %s", e)
        return out
    for side in ("left", "right"):
        side_state = end_state.get(f"{side}_end_state")
        if not side_state:
            continue
        joint_names = side_state.get("names", [])
        joint_states = side_state.get("end_states", [])
        for name, state in zip(joint_names, joint_states, strict=False):
            if name in GRIPPER_JOINT_NAMES and "position" in state:
                out[name] = float(state["position"])
    return out


def _read_current_joint_positions(
    robot,
    names: list[str],
    fallback_targets: dict[str, float] | None = None,
) -> np.ndarray:
    """Read the live joint state and project onto our joint-name order.

    Falls back to ``get_end_state`` for gripper joints that aren't in
    ``get_joint_states``. If a joint is still missing after that and
    ``fallback_targets`` was supplied, the per-joint target is used as a stand-in
    for current (Δ=0) so calibration won't crash; a clear warning is logged so
    the operator knows the calibration speed for that joint is unknown.
    """
    states = robot.get_joint_states()
    by_name = {s["name"]: float(s["motor_position"]) for s in states["states"]}

    # Augment with gripper readback (omnipicker reports via get_end_state).
    if any(n in GRIPPER_JOINT_NAMES and n not in by_name for n in names):
        gripper_rad = _read_gripper_rad_via_end_state(robot)
        for name, rad in gripper_rad.items():
            if name not in by_name:
                by_name[name] = rad
                logging.info("Gripper %s read via get_end_state(): %.4f rad", name, rad)

    positions: list[float] = []
    proxied: list[str] = []
    for n in names:
        if n in by_name:
            positions.append(by_name[n])
        elif fallback_targets is not None and n in fallback_targets:
            positions.append(fallback_targets[n])
            proxied.append(n)
        else:
            raise RuntimeError(
                f"Joint state unavailable for {n!r} — not in get_joint_states or get_end_state."
            )
    if proxied:
        logging.warning(
            "Current pose unavailable for %s; using dataset target as proxy (Δ=0). "
            "Calibration speed for these joints cannot be checked.", proxied,
        )
    return np.array(positions)


def _read_current_pose_for_dry_run(
    cfg: G2NativeReplayConfig, names: list[str], first_target: np.ndarray,
) -> np.ndarray:
    """Read live joint state via GDK *without* sending any motion command.

    Raises on any failure — dry-run must reflect the robot's real pose so the
    operator can judge how fast calibration would move. The zero-pose fallback
    was removed deliberately because it hides exactly the situation this preview
    is meant to surface. ``first_target`` is only used as a per-joint proxy when
    a specific joint can't be read (e.g. gripper joints on builds that omit
    them from both get_joint_states and get_end_state).
    """
    try:
        import agibot_gdk  # noqa: PLC0415 - intentionally lazy
    except ImportError as e:
        raise RuntimeError(
            "Dry-run needs to read the robot's live joint state via agibot_gdk, "
            "but the package is not importable. Run dry-run on the machine that "
            "has agibot_gdk installed (the robot host)."
        ) from e

    fallback_targets = dict(zip(names, (float(v) for v in first_target), strict=True))
    init_ok = False
    try:
        if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
            raise RuntimeError(
                "agibot_gdk.gdk_init() failed during dry-run. Is the robot powered "
                "on and reachable on the network?"
            )
        init_ok = True
        robot = agibot_gdk.Robot()
        time.sleep(cfg.init_delay_s)
        return _read_current_joint_positions(robot, names, fallback_targets=fallback_targets)
    finally:
        if init_ok and agibot_gdk.gdk_release() != agibot_gdk.GDKRes.kSuccess:
            logging.warning("gdk_release() failed after dry-run readback.")


def _cosine_ease(t: float) -> float:
    return 0.5 * (1.0 - math.cos(math.pi * t))


@dataclass
class _CalibrationPlan:
    current: np.ndarray
    target: np.ndarray
    delta: np.ndarray
    grip_flags: list[bool]
    arm_max_delta: float
    grip_max_delta: float
    required_arm_s: float
    required_grip_s: float
    duration_s: float
    n_steps: int
    over_cap: bool  # True if required duration > calibrate_max_duration_s


def _compute_calibration_plan(
    current: np.ndarray, target: np.ndarray, grip_flags: list[bool], cfg: G2NativeReplayConfig,
) -> _CalibrationPlan:
    delta = target - current
    abs_delta = np.abs(delta)
    arm_mask = np.array([not g for g in grip_flags])
    grip_mask = np.array(grip_flags)

    arm_max = float(abs_delta[arm_mask].max()) if arm_mask.any() else 0.0
    grip_max = float(abs_delta[grip_mask].max()) if grip_mask.any() else 0.0

    required_arm_s = arm_max / cfg.calibrate_max_joint_speed_rad_s if arm_max > 0 else 0.0
    required_grip_s = grip_max / cfg.calibrate_max_gripper_speed_rad_s if grip_max > 0 else 0.0
    required_s = max(required_arm_s, required_grip_s)

    duration_s = max(cfg.calibrate_min_duration_s, required_s)
    over_cap = duration_s > cfg.calibrate_max_duration_s
    n_steps = max(1, int(duration_s * cfg.calibrate_rate_hz))

    return _CalibrationPlan(
        current=current, target=target, delta=delta, grip_flags=grip_flags,
        arm_max_delta=arm_max, grip_max_delta=grip_max,
        required_arm_s=required_arm_s, required_grip_s=required_grip_s,
        duration_s=duration_s, n_steps=n_steps, over_cap=over_cap,
    )


def _format_calibration_plan(plan: _CalibrationPlan, names: list[str], cfg: G2NativeReplayConfig) -> str:
    """Human-readable per-joint breakdown of the calibration motion."""
    lines = [
        "Calibration plan:",
        f"  duration         = {plan.duration_s:.2f} s "
        f"({plan.n_steps} steps @ {cfg.calibrate_rate_hz:.0f} Hz, cosine ease-in/out)",
        f"  arm Δmax         = {plan.arm_max_delta:.4f} rad "
        f"(speed cap {cfg.calibrate_max_joint_speed_rad_s} rad/s → req {plan.required_arm_s:.2f} s)",
        f"  gripper Δmax     = {plan.grip_max_delta:.4f} rad "
        f"(speed cap {cfg.calibrate_max_gripper_speed_rad_s} rad/s → req {plan.required_grip_s:.2f} s)",
    ]
    if plan.over_cap:
        worst_idx = int(np.argmax(np.abs(plan.delta)))
        lines.append(
            f"  !! over cap      : {names[worst_idx]} is {abs(plan.delta[worst_idx]):.3f} rad away — "
            f"calibration would be aborted (cap={cfg.calibrate_max_duration_s}s)."
        )
    lines.append(
        f"  per-joint (peak-speed = |Δ| / duration; cosine peak ≈ π/2 × avg ≈ {math.pi / 2:.2f}×):"
    )
    lines.append(
        f"    {'joint':35s} {'current':>10s} {'target':>10s} {'Δ':>10s} {'avg|v|':>10s} {'peak|v|':>10s}"
    )
    for name, c, t, d in zip(names, plan.current, plan.target, plan.delta, strict=True):
        avg_v = abs(d) / plan.duration_s if plan.duration_s > 0 else 0.0
        peak_v = avg_v * math.pi / 2  # cosine ease peak speed
        lines.append(
            f"    {name:35s} {c:+10.4f} {t:+10.4f} {d:+10.4f} {avg_v:10.4f} {peak_v:10.4f}"
        )
    return "\n".join(lines)


def _calibrate_to_first_frame(
    robot, gdk_module, cfg: G2NativeReplayConfig,
    names: list[str], target: np.ndarray, grip_flags: list[bool],
) -> np.ndarray:
    """Smoothly servo from current pose to `target`. Returns the final commanded pose."""
    fallback_targets = dict(zip(names, (float(v) for v in target), strict=True))
    current = _read_current_joint_positions(robot, names, fallback_targets=fallback_targets)
    plan = _compute_calibration_plan(current, target, grip_flags, cfg)

    if plan.over_cap:
        worst_idx = int(np.argmax(np.abs(plan.delta)))
        raise RuntimeError(
            f"Calibration aborted: implied duration {plan.duration_s:.1f}s exceeds cap "
            f"{cfg.calibrate_max_duration_s:.1f}s — joint {names[worst_idx]} is "
            f"{abs(plan.delta[worst_idx]):.3f} rad away from the dataset start. Hand-jog "
            f"the robot closer to the start pose and retry."
        )

    logging.info(_format_calibration_plan(plan, names, cfg))

    n_steps = plan.n_steps
    delta = plan.delta
    dt = 1.0 / cfg.calibrate_rate_hz
    control_period = max(cfg.control_period, 1.2 * dt)

    t_start = time.perf_counter()
    for step in range(1, n_steps + 1):
        t_norm = step / n_steps
        alpha = _cosine_ease(t_norm)
        pos = current + alpha * delta

        offenders = _validate_targets(names, pos)
        if offenders:
            raise RuntimeError(
                "Calibration interpolation produced out-of-range joints; this should be "
                f"impossible given valid endpoints. Offenders: {offenders}"
            )

        req = gdk_module.JointServoControlReq()
        req.control_period = control_period
        req.joint_names = names
        req.joint_positions = pos.tolist()
        result = robot.joint_servo_control(req, enable_low_latency=cfg.enable_low_latency)
        if result != 0:
            raise RuntimeError(f"joint_servo_control returned {result} during calibration step {step}")

        deadline = t_start + step * dt
        precise_sleep(max(deadline - time.perf_counter(), 0.0))

    # Settle: hold the target for a few ticks so the SDK trajectory smooths out.
    settle_steps = int(cfg.calibrate_settle_s * cfg.calibrate_rate_hz)
    for _ in range(settle_steps):
        req = gdk_module.JointServoControlReq()
        req.control_period = control_period
        req.joint_names = names
        req.joint_positions = target.tolist()
        result = robot.joint_servo_control(req, enable_low_latency=cfg.enable_low_latency)
        if result != 0:
            logging.warning("settle tick: joint_servo_control returned %s", result)
            break
        precise_sleep(dt)

    return target.copy()


def _make_keyboard_listener(stop_event: threading.Event, run_event: threading.Event):
    def on_press(key):
        if key == keyboard.Key.space:
            run_event.set()
        elif key == keyboard.Key.esc:
            print("[ESC] Stopping replay...")
            stop_event.set()
            run_event.set()
            return False

    def on_release(key):
        if key == keyboard.Key.space:
            run_event.clear()

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    return listener


@parser.wrap()
def replay(cfg: G2NativeReplayConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    episode = _load_episode(cfg)
    names = _build_joint_name_list(cfg)
    n_frames = episode.arm_rad.shape[0]
    effective_fps = episode.fps * cfg.speed_factor
    dt_replay = 1.0 / effective_fps
    control_period_replay = max(cfg.control_period, 1.2 * dt_replay)

    task_name = episode.meta.get("taskName", "<unknown>")
    task_desc = episode.meta.get("taskDesc", "")
    logging.info(
        "Episode: task=%s desc=%r robot=%s frames=%d dataset_fps=%.2f effective_fps=%.2f joints=%d",
        task_name, task_desc, episode.meta.get("robotType", "?"),
        n_frames, episode.fps, effective_fps, len(names),
    )

    start_frame = max(0, min(cfg.start_frame, n_frames - 1))

    # Pre-compute every frame's target + sanity-check before we touch the robot.
    pre_targets: list[np.ndarray] = []
    grip_flags: list[bool] | None = None
    bad_frames: list[tuple[int, list[str]]] = []
    for idx in range(n_frames):
        target, flags = _frame_targets(cfg, episode, idx)
        if grip_flags is None:
            grip_flags = flags
        offenders = _validate_targets(names, target)
        if offenders:
            bad_frames.append((idx, offenders))
        pre_targets.append(target)
    if bad_frames:
        logging.warning(
            "Found %d frames with out-of-limit joint values (first 5: %s); they will be skipped at replay time.",
            len(bad_frames), bad_frames[:5],
        )
    assert grip_flags is not None

    first_target = pre_targets[start_frame]

    if cfg.dry_run:
        print("=" * 60)
        print(f"DRY RUN — would replay {n_frames - start_frame} frames at {effective_fps:.1f} Hz")
        print(f"  joint names ({len(names)}):")
        for i, name in enumerate(names):
            lo, hi = JOINT_LIMITS[name]
            print(f"    [{i:2d}] {name:35s}  limits=[{lo:+.3f}, {hi:+.3f}]")
        print(f"  first frame target ({start_frame}):")
        for n, v in zip(names, first_target, strict=True):
            print(f"    {n:35s} = {v:+.4f} rad")
        print(f"  out-of-range frames: {len(bad_frames)}")

        # Calibration preview — read current joint state via GDK without sending
        # any motion command. Raises if the SDK is unavailable: a dry-run that
        # shows fake zeros would defeat the point of previewing motion speed.
        current = _read_current_pose_for_dry_run(cfg, names, first_target)
        plan = _compute_calibration_plan(current, first_target, grip_flags, cfg)
        print()
        print("  Calibration preview (current pose: live readback via agibot_gdk.get_joint_states()):")
        for line in _format_calibration_plan(plan, names, cfg).splitlines():
            print(f"  {line}")
        return

    # GDK lifecycle
    import agibot_gdk

    if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
        raise RuntimeError("agibot_gdk.gdk_init() failed")
    logging.info("GDK initialized.")

    robot = agibot_gdk.Robot()
    time.sleep(cfg.init_delay_s)

    listener = None
    try:
        if cfg.calibrate_start:
            prev_targets = _calibrate_to_first_frame(
                robot=robot, gdk_module=agibot_gdk, cfg=cfg, names=names,
                target=first_target, grip_flags=grip_flags,
            )
            logging.info("Calibration complete.")
        else:
            prev_targets = _read_current_joint_positions(robot, names)

        stop_event = threading.Event()
        run_event = threading.Event()
        listener = _make_keyboard_listener(stop_event, run_event)

        print("=" * 60)
        print(f"Task: {task_name}   frames: {n_frames}   fps: {effective_fps:.1f}")
        print("Hold SPACE to replay, release to pause.  ESC to exit.")
        print("=" * 60)

        loop_dt = 1.0 / effective_fps
        for idx in range(start_frame, n_frames):
            if stop_event.is_set():
                print("Replay stopped by user.")
                break

            while not run_event.is_set() and not stop_event.is_set():
                time.sleep(0.01)
            if stop_event.is_set():
                break

            t_loop = time.perf_counter()

            raw_target = pre_targets[idx]
            offenders = _validate_targets(names, raw_target)
            if offenders:
                logging.warning("Skipping frame %d (out-of-range): %s", idx, offenders)
                precise_sleep(max(loop_dt - (time.perf_counter() - t_loop), 0.0))
                continue

            clipped_target, clipped_log = _clip_step(
                raw_target, prev_targets, grip_flags,
                cfg.max_step_joint_delta_rad, cfg.max_step_gripper_delta_rad,
            )
            if clipped_log:
                for i, requested, allowed in clipped_log[:4]:
                    logging.warning(
                        "frame=%d clip %s requested Δ=%+.4f -> allowed Δ=%+.4f rad",
                        idx, names[i], requested, allowed,
                    )

            req = agibot_gdk.JointServoControlReq()
            req.control_period = control_period_replay
            req.joint_names = names
            req.joint_positions = clipped_target.tolist()
            result = robot.joint_servo_control(req, enable_low_latency=cfg.enable_low_latency)
            if result != 0:
                logging.error("joint_servo_control returned %s at frame %d; aborting replay.", result, idx)
                break

            prev_targets = clipped_target
            precise_sleep(max(loop_dt - (time.perf_counter() - t_loop), 0.0))

        log_say("Replay finished", cfg.play_sounds)
    finally:
        if listener is not None:
            listener.stop()
        if agibot_gdk.gdk_release() != agibot_gdk.GDKRes.kSuccess:
            logging.warning("agibot_gdk.gdk_release() failed")
        else:
            logging.info("GDK released.")


def main():
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()
    replay()


if __name__ == "__main__":
    main()
