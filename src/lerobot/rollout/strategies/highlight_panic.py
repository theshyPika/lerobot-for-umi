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

"""Highlight rollout with a human-triggered keyboard panic-pause.

Designed for multi-trial policy evaluation where the operator wants:

1. Manual control of each evaluation episode's start and end (``save_key``).
2. A pause window between episodes during which the policy is silent
   so the robot can be manually returned to a reset pose (``panic_key``).

PAUSED phase semantics differ from :class:`SentryPanicStrategy`: nothing
is written to the robot, so the arm is free to be moved by hand.  This
requires the robot's servo interface to tolerate a silent control
channel for the pause duration.

Example keypress workflow for two evaluation episodes
-----------------------------------------------------
Defaults: ``save_key='s'``, ``panic_key='space'``, ``push_key='h'``.

::

    [launch script]
        -> policy auto-runs, ring buffer captures last N seconds (BUFFERING)

    press  s        episode 1 start: flush ring buffer + go LIVE
        ... robot performs the task while everything is recorded ...
    press  s        episode 1 end:   save_episode() -> back to BUFFERING

    press  SPACE    enter PAUSED: engine.pause(), no commands to robot,
                                  ring buffer dropped, robot is free
        ... operator hand-moves the arm back to its reset pose ...
    press  SPACE    exit PAUSED:  interpolator.reset() + engine.reset()
                                  + engine.resume() -> back to BUFFERING
                                  (ring buffer starts fresh)

    press  s        episode 2 start: flush ring buffer + go LIVE
        ... robot performs the task again ...
    press  s        episode 2 end:   save_episode() -> back to BUFFERING

    press  h        (optional) trigger background push_to_hub

    press  ESC      stop the script (teardown saves any in-progress
                                     episode and pushes to Hub if configured)

Edge case: pressing SPACE while a live recording is in progress
auto-saves the in-progress episode for a clean boundary before
entering PAUSED -- so SPACE doubles as a "save + reset" shortcut.
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

from ..configs import HighlightPanicStrategyConfig
from ..context import RolloutContext
from .base_panic import PYNPUT_AVAILABLE, _resolve_special_key, keyboard
from .core import send_next_action
from .highlight import HighlightStrategy

logger = logging.getLogger(__name__)


class HighlightPanicStrategy(HighlightStrategy):
    """Highlight recording with a one-key human panic-pause.

    Adds a third keyboard event on top of :class:`HighlightStrategy`:

    * ``save_key`` (default ``s``) — toggle live recording (unchanged).
    * ``push_key`` (default ``h``) — push dataset to Hub (unchanged).
    * ``panic_key`` (default ``space``) — toggle RUNNING ↔ PAUSED.

    In PAUSED:

    * ``engine.pause()`` is called once on the falling edge.
    * **No action** is sent to the robot — the arm is released so the
      operator can hand-move it to a reset pose.
    * No frames are added to the dataset or the ring buffer.

    If the user enters PAUSED while a live recording is in progress, the
    in-progress episode is saved first (clean boundary) so the recorded
    file never contains a temporal gap.

    On resume the interpolator and engine are reset so stale chunked
    actions computed against the pre-pause pose never replay against the
    operator's freshly-reset starting position.
    """

    config: HighlightPanicStrategyConfig

    def __init__(self, config: HighlightPanicStrategyConfig) -> None:
        super().__init__(config)
        self._panic: Event = Event()
        self._panic_listener = None
        self._was_paused: bool = False

    def setup(self, ctx: RolloutContext) -> None:
        super().setup(ctx)
        self._panic_listener = self._spawn_panic_listener(
            self.config.panic_key, ctx.runtime.shutdown_event
        )
        if self._panic_listener is not None:
            logger.info(
                "Panic-pause listener armed (key='%s'); press to toggle PAUSED/RUNNING",
                self.config.panic_key,
            )
        else:
            logger.warning(
                "Panic-pause listener unavailable (headless or pynput missing) — "
                "soft-stop will not be reachable in this session."
            )

    def run(self, ctx: RolloutContext) -> None:
        """Run the autonomous loop with on-demand recording AND panic-pause."""
        engine = self._engine
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        dataset = ctx.data.dataset
        ring = self._ring
        interpolator = self._interpolator
        features = ctx.data.dataset_features

        control_interval = interpolator.get_control_interval(cfg.fps)

        engine.resume()
        play_sounds = cfg.play_sounds

        start_time = time.perf_counter()
        task_str = cfg.dataset.single_task if cfg.dataset else cfg.task
        logger.info(
            "HighlightPanic recording started (save='%s', panic='%s', push='%s')",
            self.config.save_key,
            self.config.panic_key,
            self.config.push_key,
        )

        with VideoEncodingManager(dataset):
            try:
                while not ctx.runtime.shutdown_event.is_set():
                    loop_start = time.perf_counter()

                    if cfg.duration > 0 and (time.perf_counter() - start_time) >= cfg.duration:
                        logger.info("Duration limit reached (%.0fs)", cfg.duration)
                        break

                    if self._panic.is_set():
                        # —— PAUSED phase ————————————————————————————————
                        if not self._was_paused:
                            # On entering pause, close any in-progress live
                            # episode so the saved file never contains the
                            # gap-during-pause.  Subsequent frames after
                            # resume start a fresh ring-buffer cycle.
                            if self._recording_live.is_set():
                                logger.warning(
                                    ">>> PANIC PAUSED during live recording — saving in-progress episode"
                                )
                                with contextlib.suppress(Exception):
                                    with self._episode_lock:
                                        dataset.save_episode()
                                    log_say(
                                        f"Episode {dataset.num_episodes} saved (panic)",
                                        play_sounds,
                                    )
                                self._recording_live.clear()
                            engine.pause()
                            # Drop ring-buffer contents — operator is about to
                            # hand-move the robot, so the pre-pause frames are
                            # not part of any next evaluation episode.
                            ring.drain()
                            self._was_paused = True
                            logger.warning(
                                ">>> PANIC PAUSED — robot released, no commands sent "
                                "(press '%s' to resume)",
                                self.config.panic_key,
                            )
                        # Quiet wait: NO robot.send_action — arm is free.
                        self._sleep_remaining(loop_start, control_interval, cfg.fps)
                        continue

                    # —— RUNNING phase ————————————————————————————————
                    if self._was_paused:
                        logger.warning(
                            ">>> RESUMING — clearing stale inference state; "
                            "press '%s' to start the next evaluation episode",
                            self.config.save_key,
                        )
                        interpolator.reset()
                        engine.reset()
                        engine.resume()
                        self._was_paused = False

                    obs = robot.get_observation()
                    obs_processed = self._process_observation_and_notify(ctx.processors, obs)

                    if self._handle_warmup(cfg.use_torch_compile, loop_start, control_interval):
                        continue

                    action_dict = send_next_action(obs_processed, obs, ctx, interpolator)

                    if action_dict is not None:
                        self._log_telemetry(obs_processed, action_dict, ctx.runtime)
                        obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
                        action_frame = build_dataset_frame(features, action_dict, prefix=ACTION)
                        frame = {**obs_frame, **action_frame, "task": task_str}

                        # ``is_set()`` then ``clear()`` is non-atomic against the
                        # keyboard thread but benign — lose at most one toggle,
                        # processed on the next iteration.  (Same property as
                        # HighlightStrategy.)
                        if self._save_requested.is_set():
                            self._save_requested.clear()
                            if not self._recording_live.is_set():
                                logger.info(
                                    "Flushing ring buffer (%d frames) + starting live recording",
                                    len(ring),
                                )
                                for buffered_frame in ring.drain():
                                    dataset.add_frame(buffered_frame)
                                self._recording_live.set()
                            else:
                                dataset.add_frame(frame)
                                with self._episode_lock:
                                    dataset.save_episode()
                                logger.info("Episode saved (total: %d)", dataset.num_episodes)
                                log_say(
                                    f"Episode {dataset.num_episodes} saved",
                                    play_sounds,
                                )
                                self._recording_live.clear()
                                self._sleep_remaining(loop_start, control_interval, cfg.fps)
                                continue  # frame already consumed — skip ring.append

                        if self._push_requested.is_set():
                            self._push_requested.clear()
                            logger.info("Push requested by user")
                            self._background_push(dataset, cfg)

                        if self._recording_live.is_set():
                            dataset.add_frame(frame)
                        else:
                            ring.append(frame)

                    self._sleep_remaining(loop_start, control_interval, cfg.fps)

            finally:
                logger.info("HighlightPanic control loop ended")
                if self._recording_live.is_set():
                    logger.info("Saving in-progress live episode")
                    with contextlib.suppress(Exception), self._episode_lock:
                        dataset.save_episode()

    def teardown(self, ctx: RolloutContext) -> None:
        if self._panic_listener is not None:
            with contextlib.suppress(Exception):
                self._panic_listener.stop()
            self._panic_listener = None
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
                f"HighlightPanic loop is running slower ({1 / dt:.1f} Hz) than the target FPS "
                f"({fps} Hz). Dataset frames might be dropped and robot control might be "
                f"unstable. Common causes are: 1) Camera FPS not keeping up 2) Policy "
                f"inference taking too long 3) CPU starvation"
            )

    def _spawn_panic_listener(self, key_name: str, shutdown_event: Event):
        """Start a separate pynput listener for the panic key.

        Runs alongside HighlightStrategy's own listener (which handles
        ``save_key`` / ``push_key`` / ESC), so the panic toggle is
        independent of the save/push events.
        """
        from lerobot.common.control_utils import is_headless

        if not PYNPUT_AVAILABLE or is_headless():
            return None

        special_target = _resolve_special_key(key_name)

        def on_press(key):
            with contextlib.suppress(Exception):
                if special_target is not None and key == special_target:
                    self._toggle_panic()
                elif hasattr(key, "char") and key.char == key_name:
                    self._toggle_panic()

        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        return listener

    def _toggle_panic(self) -> None:
        if self._panic.is_set():
            self._panic.clear()
        else:
            self._panic.set()
