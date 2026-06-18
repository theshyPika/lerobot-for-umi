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

"""Sentry rollout with eval-friendly keyboard controls.

Three states:

* RUNNING — frames are recorded and policy actions reach the robot.
* PAUSED — robot released (no actions sent), engine paused, no frames
  recorded. Two flavours, distinguished by ``_new_episode_pending``:
  - normal pause (entered via ``space`` from RUNNING): resume returns
    to the same RUNNING episode.
  - new-episode-pending (entered when RESET completes, or via ``→``/``←``
    when ``reset_time_s == 0``): resume **starts** the next episode.
  Wall time spent in PAUSED is excluded from ``cfg.duration``.
* RESET — between-episode countdown window with a progress bar. Robot
  released so the operator can manually reset the scene. Exits on
  ``reset_time_s`` timeout or by pressing ``n``; either way the next
  state is PAUSED (new-episode-pending) — the operator presses ``n``
  again when ready to start the next episode.

Keys (split to avoid toggle ambiguity):

* ``panic_key`` (default ``space``) — RUNNING → PAUSED only.  No-op in
  PAUSED / RESET.
* ``n`` — PAUSED → RUNNING (resume / start next episode), RESET →
  PAUSED (skip remaining countdown).
* ``→`` (right arrow) — save current episode (if any pending frames)
  and enter RESET. Works in RUNNING **and** PAUSED.
* ``←`` (left arrow) — discard current episode buffer and enter RESET.
  Works in RUNNING **and** PAUSED.
* ``ESC`` — global shutdown.
"""

from __future__ import annotations

import contextlib
import logging
import time
from threading import Event

from lerobot.datasets import VideoEncodingManager
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say

from ..configs import SentryPanicStrategyConfig
from ..context import RolloutContext
from .base_panic import PYNPUT_AVAILABLE, _resolve_special_key, keyboard
from .core import send_next_action
from .sentry import SentryStrategy

logger = logging.getLogger(__name__)


_STATE_RUNNING = "RUNNING"
_STATE_PAUSED = "PAUSED"
_STATE_RESET = "RESET"


class SentryPanicStrategy(SentryStrategy):
    """Sentry recording with eval-oriented keyboard control surface.

    Differences from :class:`SentryStrategy`:

    * Adds RUNNING/PAUSED/RESET state machine driven by keyboard events.
    * In PAUSED and RESET the robot is released (no ``robot.send_action``
      call) so the operator can hand-move the arm. Requires the robot
      servo interface to tolerate brief silence.
    * Per-episode ``_pause_offset_s`` (existing) shields the size-based
      rotation from PAUSED time; new session-wide
      ``_session_pause_offset_s`` shields ``cfg.duration``.
    * ``→`` / ``←`` provide lerobot-record-style manual episode control.
    * Background pushes honour ``cfg.dataset.push_to_hub`` so a value of
      ``false`` keeps eval runs fully local.

    Episode rotation enters RESET when ``reset_time_s > 0``. With
    ``reset_time_s == 0`` the strategy immediately starts the next
    episode (matches the pre-feature behaviour, modulo the new keys).
    """

    config: SentryPanicStrategyConfig

    def __init__(self, config: SentryPanicStrategyConfig) -> None:
        super().__init__(config)
        self._panic: Event = Event()       # space → enter PAUSED
        self._resume: Event = Event()      # n → leave PAUSED / RESET
        self._end_episode: Event = Event()  # → end current episode
        self._rerecord: Event = Event()    # ← discard buffer + RESET
        self._listener = None
        # State machine (written only by the main loop).
        self._state: str = _STATE_RUNNING
        # PAUSED has two flavours: normal pause (resume goes back to the
        # same episode) and "new-episode-pending" (resume starts the next
        # episode). The flag distinguishes them.
        self._new_episode_pending: bool = False
        # Per-episode pause accumulator (shields size-based rotation).
        self._pause_offset_s: float = 0.0
        # Session-wide pause accumulator (shields cfg.duration).
        self._session_pause_offset_s: float = 0.0
        self._pause_started_at: float | None = None
        self._reset_started_at: float | None = None
        # Throttle for the RESET progress bar (only redraw every ~0.25s).
        self._last_reset_bar_emit: float = 0.0
        # Promoted from local var so helper methods can update it.
        self._episodes_since_push: int = 0

    def setup(self, ctx: RolloutContext) -> None:
        super().setup(ctx)
        self._listener = self._spawn_panic_listener(
            self.config.panic_key, ctx.runtime.shutdown_event
        )
        if self._listener is not None:
            logger.info(
                "Keyboard listener armed (pause='%s', resume='n', '→' end-episode, "
                "'←' rerecord, ESC shutdown)",
                self.config.panic_key,
            )
        else:
            logger.warning(
                "Keyboard listener unavailable (headless or pynput missing) — "
                "manual episode control will not be reachable in this session."
            )

    def run(self, ctx: RolloutContext) -> None:
        """Run the recording loop with the three-state keyboard control surface."""
        engine = self._engine
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        dataset = ctx.data.dataset
        interpolator = self._interpolator
        features = ctx.data.dataset_features

        control_interval = interpolator.get_control_interval(cfg.fps)

        engine.resume()
        play_sounds = cfg.play_sounds
        episode_duration_s = self._episode_duration_s
        reset_time_s = self.config.reset_time_s
        push_enabled = cfg.dataset is not None and cfg.dataset.push_to_hub

        start_time = time.perf_counter()
        episode_start = time.perf_counter()
        self._episodes_since_push = 0
        task_str = cfg.dataset.single_task if cfg.dataset else cfg.task
        logger.info(
            "SentryPanic recording started (episode_duration=%.0fs, reset=%.1fs, "
            "panic='%s', push_to_hub=%s)",
            episode_duration_s,
            reset_time_s,
            self.config.panic_key,
            push_enabled,
        )

        with VideoEncodingManager(dataset):
            try:
                while not ctx.runtime.shutdown_event.is_set():
                    loop_start = time.perf_counter()

                    # Global duration check excludes PAUSED time so a long
                    # break doesn't burn the eval budget. RESET time is
                    # included — it's planned reset wall clock.
                    if cfg.duration > 0 and (
                        loop_start - start_time - self._session_pause_offset_s
                    ) >= cfg.duration:
                        logger.info("Duration limit reached (%.0fs)", cfg.duration)
                        break

                    # —— Global priority: →/← in RUNNING or PAUSED ————————
                    # In RESET we drop the events (operator already in the
                    # between-episode window; the next 'n' will advance).
                    if self._state == _STATE_RESET:
                        if self._rerecord.is_set() or self._end_episode.is_set():
                            self._rerecord.clear()
                            self._end_episode.clear()
                    elif self._rerecord.is_set() or self._end_episode.is_set():
                        self._handle_end_episode(
                            loop_start, dataset, cfg, play_sounds,
                            push_enabled, reset_time_s, engine,
                        )
                        self._sleep_remaining(loop_start, control_interval, cfg.fps)
                        continue

                    # —— PAUSED ————————————————————————————————————————
                    if self._state == _STATE_PAUSED:
                        if self._resume.is_set():
                            self._resume.clear()
                            paused_dt = loop_start - (self._pause_started_at or loop_start)
                            self._session_pause_offset_s += paused_dt
                            self._pause_started_at = None
                            # Load-bearing reset on resume in both flavours:
                            # stale chunked actions must never replay against
                            # a freshly hand-placed scene.
                            interpolator.reset()
                            engine.reset()
                            # New-episode flavour only: re-anchor the robot-side
                            # pipelines (FK quaternion sign-continuity anchor + IK
                            # seed) so the next episode starts from a clean per-arm
                            # anchor — mirroring how the training data anchors each
                            # episode independently (frame-0 qw >= 0). A normal
                            # pause/resume keeps the anchor: the pose is physically
                            # continuous across the hold, so re-anchoring there could
                            # flip the hemisphere mid-episode. Done before priming so
                            # the primed obs already carries the re-anchored sign.
                            if self._new_episode_pending:
                                ctx.processors.robot_observation_processor.reset()
                                ctx.processors.robot_action_processor.reset()
                            # Prime the engine with a fresh observation BEFORE
                            # resume so the RTC thread can't race the next main-
                            # loop tick: without this, it picks up the pre-pause
                            # obs still sitting in the holder and emits a chunk
                            # whose absolute targets are anchored to the pose
                            # the robot had before the operator hand-moved it.
                            obs_prime = robot.get_observation()
                            obs_prime_processed = ctx.processors.robot_observation_processor(obs_prime)
                            engine.notify_observation(obs_prime_processed)
                            self._cached_obs_processed = obs_prime_processed
                            engine.resume()
                            if self._new_episode_pending:
                                self._new_episode_pending = False
                                episode_start = time.perf_counter()
                                self._pause_offset_s = 0.0
                                log_say("New episode", play_sounds)
                                self._print_banner(
                                    ">>> NEW EPISODE — policy active"
                                )
                            else:
                                # Normal pause within an in-progress episode.
                                self._pause_offset_s += paused_dt
                                self._print_banner(
                                    ">>> RESUMING — policy back in control"
                                )
                            self._state = _STATE_RUNNING
                        # No robot.send_action — arm is free.
                        self._sleep_remaining(loop_start, control_interval, cfg.fps)
                        continue

                    # —— RESET ——————————————————————————————————————————
                    if self._state == _STATE_RESET:
                        skip = self._resume.is_set()
                        elapsed_reset = loop_start - (self._reset_started_at or loop_start)
                        timeout = elapsed_reset >= reset_time_s
                        if skip or timeout:
                            if skip:
                                self._resume.clear()
                            # Terminate the progress-bar line cleanly before
                            # the next logger.warning prints.
                            self._clear_progress_line()
                            # RESET → PAUSED (new-episode-pending). Operator
                            # presses 'n' again when ready to actually start
                            # the next episode.
                            self._new_episode_pending = True
                            self._pause_started_at = loop_start
                            self._reset_started_at = None
                            self._state = _STATE_PAUSED
                            reason = "skipped" if skip else "timeout"
                            self._print_banner(
                                f">>> RESET complete ({reason}) — press 'n' to start next episode"
                            )
                            log_say("Press n to start", play_sounds)
                        else:
                            # Throttled progress bar refresh.
                            if loop_start - self._last_reset_bar_emit >= 0.25:
                                self._render_reset_progress(elapsed_reset, reset_time_s)
                                self._last_reset_bar_emit = loop_start
                        self._sleep_remaining(loop_start, control_interval, cfg.fps)
                        continue

                    # —— RUNNING ———————————————————————————————————————
                    # Stray 'n' presses in RUNNING are no-ops; clear so they
                    # don't accidentally fire the first time we enter RESET.
                    if self._resume.is_set():
                        self._resume.clear()

                    # Priority 1: panic key -> PAUSED (normal pause)
                    if self._panic.is_set():
                        self._panic.clear()
                        engine.pause()
                        self._pause_started_at = loop_start
                        self._new_episode_pending = False
                        self._state = _STATE_PAUSED
                        self._print_banner(
                            ">>> PAUSED — arm released (press 'n' to resume, "
                            "'→' end ep, '←' rerecord)"
                        )
                        self._sleep_remaining(loop_start, control_interval, cfg.fps)
                        continue

                    # Normal RUNNING frame
                    obs = robot.get_observation()
                    obs_processed = self._process_observation_and_notify(ctx.processors, obs)

                    if self._handle_warmup(cfg.use_torch_compile, loop_start, control_interval):
                        continue

                    action_dict = send_next_action(obs_processed, obs, ctx, interpolator)

                    if action_dict is not None:
                        self._log_telemetry(obs_processed, action_dict, ctx.runtime)
                        obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
                        action_frame = build_dataset_frame(features, action_dict, prefix=ACTION)
                        dataset.add_frame({**obs_frame, **action_frame, "task": task_str})

                    # Size-based auto-rotate
                    elapsed = time.perf_counter() - episode_start - self._pause_offset_s
                    if elapsed >= episode_duration_s:
                        if dataset.has_pending_frames():
                            with self._episode_lock:
                                dataset.save_episode()
                            self._episodes_since_push += 1
                            self._needs_push.set()
                            logger.info(
                                "Episode saved (size-rotated, total: %d, elapsed: %.1fs, paused: %.1fs)",
                                dataset.num_episodes,
                                elapsed,
                                self._pause_offset_s,
                            )
                            log_say(f"Episode {dataset.num_episodes} saved", play_sounds)
                            if (
                                push_enabled
                                and self._episodes_since_push >= self.config.upload_every_n_episodes
                            ):
                                self._background_push(dataset, cfg)
                                self._episodes_since_push = 0
                        self._enter_reset_or_pending(
                            loop_start, reset_time_s, engine, play_sounds,
                        )

                    self._sleep_remaining(loop_start, control_interval, cfg.fps)

            finally:
                self._clear_progress_line()
                logger.info("SentryPanic control loop ended")
                with contextlib.suppress(Exception):
                    if dataset.has_pending_frames():
                        with self._episode_lock:
                            dataset.save_episode()
                        self._needs_push.set()
                        logger.info("Final in-progress episode saved")

    def teardown(self, ctx: RolloutContext) -> None:
        if self._listener is not None:
            with contextlib.suppress(Exception):
                self._listener.stop()
            self._listener = None
        super().teardown(ctx)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _sleep_remaining(loop_start: float, control_interval: float, fps: float) -> None:
        dt = time.perf_counter() - loop_start
        if (sleep_t := control_interval - dt) > 0:
            precise_sleep(sleep_t)
        else:
            logger.warning(
                f"SentryPanic loop is running slower ({1 / dt:.1f} Hz) than the target FPS "
                f"({fps} Hz). Dataset frames might be dropped and robot control might be "
                f"unstable. Common causes are: 1) Camera FPS not keeping up 2) Policy "
                f"inference taking too long 3) CPU starvation"
            )

    # ------------------------------------------------------------------
    # State-transition helpers (factored out so the main loop is short)
    # ------------------------------------------------------------------

    def _handle_end_episode(
        self,
        loop_start: float,
        dataset,
        cfg,
        play_sounds: bool,
        push_enabled: bool,
        reset_time_s: float,
        engine,
    ) -> None:
        """Process a queued ``→`` (end episode) or ``←`` (rerecord) event.

        Called from RUNNING or PAUSED.  Saves / discards the in-progress
        buffer, then transitions into RESET (countdown) when
        ``reset_time_s > 0``, or directly into PAUSED-pending-new-episode
        otherwise.  When called from PAUSED, the accumulated pause time
        is charged to the session offset so ``cfg.duration`` accounting
        stays correct.
        """
        is_rerecord = self._rerecord.is_set()
        self._rerecord.clear()
        self._end_episode.clear()

        # Charge the pre-transition PAUSED window to the session offset.
        # Per-episode offset doesn't matter — the episode is about to be
        # closed and the timer reset.
        if self._state == _STATE_PAUSED and self._pause_started_at is not None:
            self._session_pause_offset_s += loop_start - self._pause_started_at
            self._pause_started_at = None

        if is_rerecord:
            if dataset.has_pending_frames():
                dataset.clear_episode_buffer()
                logger.warning(">>> Discarded episode buffer for re-record")
            log_say("Re-record episode", play_sounds)
        else:
            if dataset.has_pending_frames():
                with self._episode_lock:
                    dataset.save_episode()
                self._episodes_since_push += 1
                self._needs_push.set()
                logger.warning(
                    "Episode saved by user (total: %d)", dataset.num_episodes
                )
                log_say(f"Episode {dataset.num_episodes} saved", play_sounds)
                if (
                    push_enabled
                    and self._episodes_since_push >= self.config.upload_every_n_episodes
                ):
                    self._background_push(dataset, cfg)
                    self._episodes_since_push = 0
            else:
                logger.info("End-episode pressed but buffer empty — entering reset")

        self._enter_reset_or_pending(loop_start, reset_time_s, engine, play_sounds)

    def _enter_reset_or_pending(
        self,
        loop_start: float,
        reset_time_s: float,
        engine,
        play_sounds: bool,
    ) -> None:
        """Transition into RESET (countdown) or PAUSED-pending-new-episode.

        Both end-states pause the inference engine and release the robot
        (no actions sent).  ``reset_time_s == 0`` short-circuits the
        countdown and parks in PAUSED waiting for ``n``.
        """
        engine.pause()
        if reset_time_s > 0:
            self._state = _STATE_RESET
            self._reset_started_at = loop_start
            self._last_reset_bar_emit = 0.0  # force immediate first draw
            log_say("Reset the environment", play_sounds)
            self._print_banner(
                f">>> RESET window started ({reset_time_s:.1f}s) — press 'n' to skip"
            )
        else:
            self._new_episode_pending = True
            self._pause_started_at = loop_start
            self._state = _STATE_PAUSED
            log_say("Press n to start", play_sounds)
            self._print_banner(
                ">>> Episode ended — press 'n' to start next"
            )

    # ------------------------------------------------------------------
    # Terminal output helpers (progress bar + banner)
    # ------------------------------------------------------------------

    @staticmethod
    def _print_banner(text: str) -> None:
        """Print a state-change banner that survives the chatty robot logs.

        Goes through stdout (not the ``logger``) so the message stays
        visually separate from the timestamped log stream and clears any
        leftover progress bar on the same line. ANSI bold makes the
        transition obvious even when surrounded by INFO/WARNING lines.
        """
        # \r + clear-to-eol overwrites any leftover progress bar; bold
        # via \033[1m...\033[0m makes the transition pop.
        print(f"\r\033[K\033[1m{text}\033[0m", flush=True)

    def _render_reset_progress(self, elapsed: float, total: float) -> None:
        """Redraw the RESET countdown progress bar on the current line."""
        pct = min(elapsed / total, 1.0) if total > 0 else 1.0
        bar_width = 30
        filled = int(bar_width * pct)
        bar = "█" * filled + "░" * (bar_width - filled)
        remaining = max(total - elapsed, 0.0)
        line = (
            f"[RESET] [{bar}] {elapsed:5.1f}s / {total:.1f}s "
            f"(remaining {remaining:4.1f}s)  press 'n' to skip"
        )
        # \r returns to col 0; \033[K clears to end-of-line so a shorter
        # repaint cleanly overwrites the previous one.
        print(f"\r\033[K{line}", end="", flush=True)

    @staticmethod
    def _clear_progress_line() -> None:
        """End the in-place progress-bar line with a newline."""
        print("\r\033[K", end="", flush=True)

    def _spawn_panic_listener(self, key_name: str, shutdown_event: Event):
        """Spawn a pynput listener for the eval control surface.

        Split-key semantics avoid the toggle ambiguity of a single key:

        * ``panic_key`` (default space) sets ``_panic`` — handled in
          RUNNING only (main loop transitions to PAUSED).
        * ``n`` sets ``_resume`` — handled in PAUSED (resume or start
          next episode) or RESET (skip countdown).
        * ``→`` / ``←`` set ``_end_episode`` / ``_rerecord``; the main
          loop consumes them in RUNNING **and** PAUSED so the operator
          can decide to end or rerecord without un-pausing first.
        * ESC raises the global shutdown event.
        """
        from lerobot.common.control_utils import is_headless

        if not PYNPUT_AVAILABLE or is_headless():
            return None

        special_target = _resolve_special_key(key_name)

        def on_press(key):
            with contextlib.suppress(Exception):
                if key == keyboard.Key.esc:
                    shutdown_event.set()
                    return
                if key == keyboard.Key.right:
                    self._end_episode.set()
                    return
                if key == keyboard.Key.left:
                    self._rerecord.set()
                    return
                if hasattr(key, "char") and key.char == "n":
                    self._resume.set()
                    return
                if special_target is not None and key == special_target:
                    self._panic.set()
                elif hasattr(key, "char") and key.char == key_name:
                    self._panic.set()

        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        return listener
