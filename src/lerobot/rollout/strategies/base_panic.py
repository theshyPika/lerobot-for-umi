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

"""Base rollout strategy with split-key human-triggered panic-pause.

Two distinct keys avoid the toggle ambiguity that made it easy to
accidentally resume by tapping the pause key twice:

* ``panic_key`` (default space) — enter PAUSED.  Pressing while already
  PAUSED is a no-op.
* ``n`` — leave PAUSED.  Pressing while already RUNNING is a no-op.

In PAUSED the inference engine is paused and the strategy keeps
re-sending the last commanded robot action so the robot holds its pose
without entering a servo-timeout.  On resume the inference queue and
policy state are cleared so chunked actions based on stale observations
never replay after a near-collision pause.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import time
from threading import Event

from lerobot.common.control_utils import is_headless
from lerobot.utils.import_utils import _pynput_available
from lerobot.utils.robot_utils import precise_sleep

from ..configs import BasePanicStrategyConfig
from ..context import RolloutContext
from .base import BaseStrategy
from .core import send_next_action

PYNPUT_AVAILABLE = _pynput_available
keyboard = None
if PYNPUT_AVAILABLE:
    try:
        if ("DISPLAY" not in os.environ) and ("linux" in sys.platform):
            logging.info("No DISPLAY set. Skipping pynput import.")
            PYNPUT_AVAILABLE = False
        else:
            from pynput import keyboard
    except Exception as e:
        PYNPUT_AVAILABLE = False
        logging.info(f"Could not import pynput: {e}")

logger = logging.getLogger(__name__)


class BasePanicStrategy(BaseStrategy):
    """Autonomous rollout with a one-key human panic-pause.

    Same control loop as :class:`BaseStrategy` plus a pynput listener that
    toggles a ``threading.Event``.  When the event is set, the strategy:

    1. Calls ``engine.pause()`` once on the falling edge to stop RTC
       inference (no-op for sync inference).
    2. Re-sends ``self._last_safe_action`` every tick so the robot holds
       its current pose at the same servo period as during normal control.

    On resume:

    1. Calls ``interpolator.reset()`` and ``engine.reset()`` to drop any
       cached actions computed from pre-pause observations.
    2. Calls ``engine.resume()`` to restart RTC inference.

    The reset-on-resume is the load-bearing safety property — without it,
    the RTC queue's stale chunk would replay the very action the operator
    paused to avoid.
    """

    config: BasePanicStrategyConfig

    def __init__(self, config: BasePanicStrategyConfig) -> None:
        super().__init__(config)
        self._panic: Event = Event()
        self._listener = None
        self._last_safe_action: dict | None = None
        self._was_paused: bool = False

    def setup(self, ctx: RolloutContext) -> None:
        super().setup(ctx)
        self._listener = self._spawn_panic_listener(
            self.config.panic_key, ctx.runtime.shutdown_event
        )
        if self._listener is not None:
            logger.info(
                "Panic-pause listener armed (pause='%s', resume='n', shutdown=ESC)",
                self.config.panic_key,
            )
        else:
            logger.warning(
                "Panic-pause listener unavailable (headless or pynput missing) — "
                "soft-stop will not be reachable in this session."
            )

    def run(self, ctx: RolloutContext) -> None:
        """Run the autonomous control loop with soft-stop semantics."""
        engine = self._engine
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        interpolator = self._interpolator

        control_interval = interpolator.get_control_interval(cfg.fps)

        start_time = time.perf_counter()
        engine.resume()
        logger.info("BasePanic strategy control loop started")

        while not ctx.runtime.shutdown_event.is_set():
            loop_start = time.perf_counter()

            if cfg.duration > 0 and (time.perf_counter() - start_time) >= cfg.duration:
                logger.info("Duration limit reached (%.0fs)", cfg.duration)
                break

            # Observation is harvested every tick regardless of phase, so
            # post-resume the engine sees fresh state instead of one based
            # on whatever was on screen when the operator hit the key.
            obs = robot.get_observation()

            if self._panic.is_set():
                # —— PAUSED phase ————————————————————————————————
                if not self._was_paused:
                    engine.pause()
                    self._was_paused = True
                    logger.warning(">>> PANIC PAUSED — robot holding pose (press '%s' to resume)", self.config.panic_key)
                if self._last_safe_action is not None:
                    # Repeating the last commanded action satisfies servo
                    # liveness on robots whose interface requires periodic
                    # writes (e.g. G2 joint_servo_control).
                    robot.send_action(self._last_safe_action)
                self._sleep_remaining(loop_start, control_interval)
                continue

            # —— RUNNING phase ————————————————————————————————
            if self._was_paused:
                logger.warning(">>> RESUMING — clearing stale inference state")
                interpolator.reset()
                engine.reset()
                engine.resume()
                self._was_paused = False

            obs_processed = self._process_observation_and_notify(ctx.processors, obs)

            if self._handle_warmup(cfg.use_torch_compile, loop_start, control_interval):
                continue

            action_dict = send_next_action(obs_processed, obs, ctx, interpolator)

            if action_dict is not None:
                # Stash the IK-processed joint-space action so the next
                # PAUSED tick has something concrete to resend.
                # TODO(g2-frame-refactor): pass obs_processed so the action processor
                # sees EE poses in arm_base_link frame (matches policy action frame).
                # See plan hashed-roaming-riddle.md.
                self._last_safe_action = ctx.processors.robot_action_processor(
                    (action_dict, obs_processed)
                )
                self._log_telemetry(obs_processed, action_dict, ctx.runtime)

            self._sleep_remaining(loop_start, control_interval)

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
    def _sleep_remaining(loop_start: float, control_interval: float) -> None:
        dt = time.perf_counter() - loop_start
        if (sleep_t := control_interval - dt) > 0:
            precise_sleep(sleep_t)
        else:
            logger.warning(
                f"BasePanic loop is running slower ({1 / dt:.1f} Hz) than the configured "
                f"control rate. Pause reaction latency may exceed one control tick."
            )

    def _spawn_panic_listener(self, key_name: str, shutdown_event: Event):
        """Start a pynput keyboard listener with split pause/resume semantics.

        ``panic_key`` (default space) only **enters** PAUSED — pressing
        it while already PAUSED is a no-op so a repeated tap can't
        accidentally resume. ``n`` only **leaves** PAUSED. ESC raises
        the global shutdown event.
        """
        if not PYNPUT_AVAILABLE or is_headless():
            return None

        special_target = _resolve_special_key(key_name)

        def on_press(key):
            with contextlib.suppress(Exception):
                if key == keyboard.Key.esc:
                    shutdown_event.set()
                    return
                if hasattr(key, "char") and key.char == "n":
                    self._panic.clear()
                    return
                if special_target is not None and key == special_target:
                    self._panic.set()
                elif hasattr(key, "char") and key.char == key_name:
                    self._panic.set()

        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        return listener


def _resolve_special_key(key_name: str):
    """Resolve a config key string to a pynput special key, if applicable."""
    if not PYNPUT_AVAILABLE:
        return None
    if key_name == "space":
        return keyboard.Key.space
    if key_name == "tab":
        return keyboard.Key.tab
    if key_name == "enter":
        return keyboard.Key.enter
    return None
