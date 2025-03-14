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
"""pynative profile module"""
import os

from mindspore.profiler import DynamicProfilerMonitor, Profiler, ProfilerLevel
from mindspore.train import RunContext
from mindspore.communication import get_rank
from mindformers.tools import logger


def get_real_rank():
    """get rank id"""
    try:
        return get_rank()
    except RuntimeError:
        return int(os.getenv("RANK_ID", "0"))


class ProfilerCallbackDict(dict):
    """Profiler Callback Dict, whether cur_step_num in self"""
    @property
    def cur_step_num(self):
        res = -1
        if "cur_step_num" in self:
            res = self["cur_step_num"]
        return res


class PynativeProfiler:
    r"""
    Pynative profiling class
    Args:
        training_config (TransformerConfig): training_config;
    """

    def __init__(self, training_config):
        self.training_config = training_config
        self.is_dynamic = False
        if self.training_config.profile:
            if not self.training_config.profile_save_path:
                logger.warning(f"profile_save_path is not specified, using './profile' instead.")
            logger.info(f"profile will be saving to {self.training_config.profile_save_path}")
            if self.training_config.profile_dynamic_profiler_config_path:
                self.dynamic_profiler = DynamicProfilerMonitor(
                    cfg_path=self.training_config.profile_dynamic_profiler_config_path,
                    output_path=self.training_config.profile_save_path)
                self.is_dynamic = True
            else:
                profiler_level = None
                if self.training_config.profile_level == "level0":
                    profiler_level = ProfilerLevel.Level0
                elif self.training_config.profile_level == "level1":
                    profiler_level = ProfilerLevel.Level1
                elif self.training_config.profile_level == "level2":
                    profiler_level = ProfilerLevel.Level2
                logger.debug(f"profiler level {profiler_level}")

                profile_framework = None
                if self.training_config.profile_framework in ['all', 'time']:
                    profile_framework = self.training_config.profile_framework
                logger.debug(f"profile_framework {profile_framework}")

                # 按照rank_id设置性能数据落盘路径
                rank_id = get_real_rank()
                output_path = os.path.join(self.training_config.profile_save_path, f"rank_{rank_id}")

                self.profiler = Profiler(start_profile=False,
                                         output_path=output_path,
                                         profiler_level=profiler_level,
                                         with_stack=self.training_config.profile_with_stack,
                                         profile_memory=self.training_config.profile_memory,
                                         profile_framework=profile_framework,
                                         profile_communication=self.training_config.profile_communication,
                                         parallel_strategy=self.training_config.profile_parallel_strategy,
                                         aicore_metrics=self.training_config.profile_aicore_metrics,
                                         l2_cache=self.training_config.profile_l2_cache,
                                         hbm_ddr=self.training_config.profile_hbm_ddr,
                                         pcie=self.training_config.profile_pcie,
                                         data_process=self.training_config.profile_data_process,
                                         data_simplification=self.training_config.profile_data_simplification,
                                         op_time=self.training_config.profile_op_time)

    def step_begin(self, current_step):
        '''
        profiler step begin function
        Args:
            current_step (int): which step in training loop
        '''
        if not self.training_config.profile:
            return
        if self.is_dynamic:
            logger.info(f"start profiling in step {current_step}")
            cb_params = ProfilerCallbackDict({"cur_step_num": current_step})
            run_context = RunContext(cb_params)
            self.dynamic_profiler.step_begin(run_context)
        else:
            if current_step == self.training_config.profile_step_start:
                logger.info(f"start profiling in step {current_step}")
                self.profiler.start()

    def step_end(self, current_step):
        '''
        profiler step end function
        Args:
            current_step (int): which step in training loop
        '''
        if not self.training_config.profile:
            return
        if self.is_dynamic:
            cb_params = ProfilerCallbackDict({"cur_step_num": current_step})
            run_context = RunContext(cb_params)
            logger.info(f"end profiling in step {current_step}")
            self.dynamic_profiler.step_end(run_context)
        else:
            if current_step == self.training_config.profile_step_end:
                logger.info(f"stop profiling in step {current_step}")
                if self.training_config.profile_offline_analyse:
                    self.profiler.stop()
                else:
                    logger.info(f"analyzing profile")
                    self.profiler.analyse()
