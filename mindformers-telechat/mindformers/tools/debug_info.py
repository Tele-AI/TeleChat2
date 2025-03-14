# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Show debug information."""
from importlib import import_module
import os
import time

import numpy as np
from mindspore import Profiler
from mindspore.profiler import ProfilerLevel
from mindformers.tools.logger import logger
from mindformers.tools.utils import get_predict_run_mode
from mindformers.version_control import synchronize


def get_profile_settings():
    """Get profile settings for context."""
    context_module = import_module("mindformers.core.context.build_context")
    context_instance = context_module.Context()
    profile = False
    profile_start_step = 0
    profile_stop_step = 0
    profile_level = 1
    if context_instance is not None and context_instance.is_exists():
        profile = context_module.get_context("profile")
        profile_start_step = context_module.get_context("profile_start_step")
        profile_stop_step = context_module.get_context("profile_stop_step")
        profile_level = context_module.get_context("profile_level") or profile_level
        max_level = len(ProfilerLevel.__members__) - 1
        if profile_level < 0 or profile_level > max_level:
            logger.warning("Invalid profile level: %s, return level1.", profile_level)
            profile_level = 1
    profile_level = getattr(ProfilerLevel, f"Level{profile_level}")
    return profile, profile_start_step, profile_stop_step, profile_level


class BaseDebugInfo:
    """Base class for debug information."""


class DetailedLatency(BaseDebugInfo):
    """Show latency of preprocess, predict and postprocess."""

    def __init__(self):
        super().__init__()
        self.predict_run_mode = get_predict_run_mode()
        self.preprocess_start_time = 0
        self.predict_start_time = 0
        self.postprocess_start_time = 0
        self.preprocess_time_list = []
        self.predict_time_list = []
        self.postprocess_time_list = []

    def start_preprocess_timer(self):
        """PreProcess starts."""
        if self.predict_run_mode:
            self.preprocess_start_time = time.time()

    def start_predict_timer(self):
        """PreProcess ends and Predict starts."""
        if self.predict_run_mode:
            self.predict_start_time = time.time()
            self.preprocess_time_list.append(self.predict_start_time - self.preprocess_start_time)

    def start_postprocess_timer(self):
        """Predict ends and PostProcess starts."""
        if self.predict_run_mode:
            synchronize()
            self.postprocess_start_time = time.time()
            self.predict_time_list.append(self.postprocess_start_time - self.predict_start_time)

    def end_postprocess_timer(self):
        """PostProcess ends."""
        if self.predict_run_mode:
            self.postprocess_time_list.append(time.time() - self.postprocess_start_time)

    def print_info(self):
        """Print timer result."""
        if self.predict_run_mode and all(
                len(x) > 2 for x in [self.preprocess_time_list, self.predict_time_list, self.postprocess_time_list]):
            logger.info(
                "prefill prepare time: %s s; "
                "prefill predict time: %s s; "
                "prefill post time: %s s; "
                "decode prepare time: %s s; "
                "decode predict time: %s s; "
                "decode post time: %s s",
                self.preprocess_time_list[0],
                self.predict_time_list[0],
                self.postprocess_time_list[0],
                np.mean(self.preprocess_time_list[1:]),
                np.mean(self.predict_time_list[2:]),
                np.mean(self.postprocess_time_list[1:]))

    def clear(self):
        """Clear time list."""
        if self.predict_run_mode:
            self.preprocess_time_list.clear()
            self.predict_time_list.clear()
            self.postprocess_time_list.clear()


class Profiling(BaseDebugInfo):
    """Get profiling info while inferring."""

    def __init__(self):
        super().__init__()
        self.profiler = None
        self.profile, self.profile_start_step, self.profile_stop_step, self.profile_level = get_profile_settings()
        self.output_path = ""

    def start_profiling(self, cur_step):
        """Start profiling."""
        if self.profile:
            if self.profile_start_step == 0:
                self.output_path = "./profile_prefill"
                self.profile_stop_step = 0  # only get prefill profile
            else:
                self.output_path = "./profile_decode"
            if self.profiler is None:
                self.profiler = Profiler(start_profile=False, profiler_level=self.profile_level,
                                         output_path=self.output_path)
            if cur_step == self.profile_start_step:
                self.profiler.start()

    def stop_profiling(self, cur_step):
        """Stop profiling."""
        if self.profile:
            if self.profiler is None:
                raise ValueError(
                    "profiler is not inited, please check `profile_start_step` is less than `profile_stop_step`"
                )
            if cur_step == self.profile_stop_step:
                self.profiler.stop()
                logger.info("Saved profiling data to %s", os.path.realpath(self.output_path))
