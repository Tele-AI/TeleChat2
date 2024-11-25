# Copyright 2023 Huawei Technologies Co., Ltd
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
"""
Parallel Config for the Parallel Training
This is an experimental interface that is subject to change and/or deletion.
"""
from __future__ import absolute_import
from enum import Enum
# MindSpore 2.0 has changed the APIs of _checkparam, the following try except is for compatibility
try:
    from mindspore._checkparam import Validator
except ImportError:
    import mindspore._checkparam as Validator
from mindspore import context
import mindspore.communication.management as D
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode
from mindspore import log as logger

__all__ = [
    "OpParallelConfig"
]


class _Config:
    r""" A basic class of the configure"""

    def __str__(self):
        info = "[ParallelConfig]" + '\n'
        for k, v in self.__dict__.items():
            var_info = "{}:{}\n".format(k, v)
            info += var_info
        return info


class MoEParallelConfig(_Config):
    r"""
        MoEParallelConfig for MoE structure, which includes setting data parallel, model parallel and expert parallel.

        Args:
            data_parallel (int): The data parallel way. Default: 1
            model_parallel (int): The model parallel way. Default: 1
            expert_parallel (int): The expert parallel way. Default: 1
        Supported Platforms:
            ``Ascend``
    """

    def __init__(self, data_parallel=1, model_parallel=1, expert_parallel=1, context_parallel=1,
                 use_seq_parallel=False, select_recompute=False):
        Validator.check_positive_int(data_parallel, "data_parallel")
        Validator.check_positive_int(model_parallel, "model_parallel")
        Validator.check_positive_int(context_parallel, "context_parallel")
        Validator.check_positive_int(expert_parallel, "expert_parallel")
        Validator.check_bool(use_seq_parallel, "use_seq_parallel")
        self._dpmp = OpParallelConfig(data_parallel=data_parallel,
                                      model_parallel=model_parallel,
                                      context_parallel=context_parallel,
                                      use_seq_parallel=use_seq_parallel,
                                      select_recompute=select_recompute)
        self.expert_parallel = expert_parallel
        self.use_seq_parallel = use_seq_parallel
        self.select_recompute = select_recompute

    @property
    def data_parallel(self):
        return self._dpmp.data_parallel

    @data_parallel.setter
    def data_parallel(self, value):
        Validator.check_positive_int(value, "data_parallel")
        self._dpmp.data_parallel = value

    @property
    def model_parallel(self):
        return self._dpmp.model_parallel

    @model_parallel.setter
    def model_parallel(self, value):
        Validator.check_positive_int(value, "model_parallel")
        self._dpmp.model_parallel = value

    @property
    def context_parallel(self):
        return self._dpmp.context_parallel

    @context_parallel.setter
    def context_parallel(self, value):
        Validator.check_positive_int(value, "context_parallel")
        self._dpmp.model_parallel = value

    @property
    def expert_parallel(self):
        return self._expert_parallel

    @expert_parallel.setter
    def expert_parallel(self, value):
        Validator.check_positive_int(value, "expert_parallel")
        self._expert_parallel = value

    @property
    def dpmp(self):
        """ Get the configuration for dpmp """
        return self._dpmp


class ContextParallelAlgo(Enum):
    """context parallel algorithm type.

    Args:
        Enum (str): chosses context parallel type
    """
    colossalai_cp = "colossalai_cp"
    ulysses_cp = "ulysses_cp"


class OpParallelConfig(_Config):
    r"""
        Configuration for operator parallelism, used to set the method of operator-level parallelism.

        Args:
            data_parallel (int, optional): The number of data parallel. Default: ``1`` .
            model_parallel (int, optional): The number of model parallel. Default: ``1`` .
            use_seq_parallel (bool, optional): Whether to use sequence parallelism. Default: ``False`` .
            context_parallel (int, optional): The number of context parallelism. Default: ``1`` .
            select_recompute (bool, optional): Whether to select recomputation. Default: ``False`` .
            context_parallel_algo (str, optional):  The context parallelism algorithm,
                with options ``"colossalai_cp"`` and ``"ulysses_cp"`` . Default: ``"colossalai_cp"`` .

        Returns:
            Instance of OpParallelConfig.

        Examples:
            >>> from mindformers.modules import OpParallelConfig
            >>> config=OpParallelConfig(data_parallel=1, model_parallel=1)
            >>> print(config)
            [ParallelConfig]
            _data_parallel:1
            _model_parallel:1
            _context_parallel:1
            use_seq_parallel:False
            select_recompute:False
            context_parallel_algo:ContextParallelAlgo.colossalai_cp
    """

    def __init__(self, data_parallel=1, model_parallel=1, use_seq_parallel=False, context_parallel=1,
                 select_recompute=False, context_parallel_algo: str = "colossalai_cp"):
        Validator.check_positive_int(data_parallel, "data_parallel")
        Validator.check_positive_int(model_parallel, "model_parallel")
        Validator.check_positive_int(context_parallel, "context_parallel")
        Validator.check_bool(use_seq_parallel, "use_seq_parallel")
        self.data_parallel = data_parallel
        self.model_parallel = model_parallel
        self.context_parallel = context_parallel
        self.use_seq_parallel = use_seq_parallel
        self.select_recompute = select_recompute
        self.context_parallel_algo = ContextParallelAlgo(context_parallel_algo)

    def _check_context_parallel(self):
        """check whether context parallel config is valid.
        """
        if self.context_parallel == 1 and self.context_parallel_algo != ContextParallelAlgo.colossalai_cp:
            logger.warning(f"context_parallel_algo {self.context_parallel_algo.value} will not take effect "
                           "when context_parallel == 1.")

    def get_ulysses_cp_num(self):
        """get ulysses context parallel num under this config.

        Returns:
            int, ulysses degrees.
        """
        if self.context_parallel == 1:
            return 1
        if self.context_parallel_algo == ContextParallelAlgo.colossalai_cp:
            return 1
        # ulysses_cp
        return self.context_parallel

    @property
    def data_parallel(self):
        return self._data_parallel

    @data_parallel.setter
    def data_parallel(self, value):
        Validator.check_positive_int(value, "data_parallel")
        self._data_parallel = value

    @property
    def model_parallel(self):
        return self._model_parallel

    @model_parallel.setter
    def model_parallel(self, value):
        Validator.check_positive_int(value, "model_parallel")
        self._model_parallel = value

    @property
    def context_parallel(self):
        return self._context_parallel

    @context_parallel.setter
    def context_parallel(self, value):
        Validator.check_positive_int(value, "context_parallel")
        self._context_parallel = value

    def __eq__(self, other) -> bool:
        return isinstance(other, OpParallelConfig) and (self.to_dict() == other.to_dict())

    def to_dict(self):
        """to dict"""
        config_dict = {
            'data_parallel': self.data_parallel,
            'model_parallel': self.model_parallel,
            'context_parallel': self.context_parallel,
            'use_seq_parallel': self.use_seq_parallel,
            'select_recompute': self.select_recompute,
            'context_parallel_algo': self.context_parallel_algo.value,
        }
        return config_dict

    def to_diff_dict(self):
        """
        Compare the configuration dictionary of the current object with the default configuration dictionary,
        identify the differences between the two, and store these differences in a new dictionary called res-dict
        """
        config_dict = self.to_dict()
        default_dict = OpParallelConfig().to_dict()
        res_dict = {}
        for k, v in config_dict.items():
            if v != default_dict.get(k):
                res_dict[k] = v
        return res_dict


class _PipeLineConfig(_Config):
    r"""
        PPConfig for the setting data parallel, model parallel

        Args:
            pipeline_stage (int): The number of the pipeline stages. Default: 1
            micro_batch_num (int): The model parallel way. Default: 1
        Supported Platforms:
            ``Ascend`` ``GPU``

        Examples:
            >>> config=_PipeLineConfig(pipeline_stage=1, micro_batch_num=1)
    """

    def __init__(self, pipeline_stage=1, micro_batch_num=1):
        Validator.check_positive_int(pipeline_stage, "pipeline_stage")
        Validator.check_positive_int(micro_batch_num, "micro_batch_num")
        self.pipeline_stage = pipeline_stage
        self.micro_batch_num = micro_batch_num

    @property
    def pipeline_stage(self):
        return self._pipeline_stage

    @pipeline_stage.setter
    def pipeline_stage(self, value):
        Validator.check_positive_int(value, "pipeline_stage")
        self._pipeline_stage = value
        context.set_auto_parallel_context(pipeline_stages=value)

    @property
    def micro_batch_num(self):
        return self._micro_batch_num

    @micro_batch_num.setter
    def micro_batch_num(self, value):
        Validator.check_positive_int(value, "micro_batch_num")
        self._micro_batch_num = value


# In case the user doesn't pass a config as args.
default_dpmp_config = OpParallelConfig()
default_moeparallel_config = MoEParallelConfig()


def _check_config(config):
    """
       Check if micro_batch_num >= pipeline_stage
    """
    # the config pipeline_stage is same with context.pipeline_stage
    pipeline_stage = context.get_auto_parallel_context("pipeline_stages")
    if hasattr(config, 'pipeline_stage') and pipeline_stage != config.pipeline_stage:
        raise ValueError(
            f"The pipeline stage {pipeline_stage} in auto_parallel_context is not equal to the pipeline_stage "
            f"{config.pipeline_stage}"
            f" in the config.")

    # make sure the following is in auto parallel mode
    is_auto_parallel = _get_parallel_mode() in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL)
    if not is_auto_parallel:
        return

    device_num = D.get_group_size()
    optimizer_shard = context.get_auto_parallel_context("enable_parallel_optimizer")

    if config.data_parallel * config.model_parallel * pipeline_stage > device_num:
        raise ValueError(f"The product of the data parallel {config.data_parallel}, "
                         f"model parallel {config.model_parallel} "
                         f"pipeline stages {pipeline_stage} "
                         f"should be less than device_num {device_num}.")

    # the config optimizer_shard is same with context.optimizer_shard
    if hasattr(config, "optimizer_shard") and optimizer_shard and optimizer_shard != config.optimizer_shard:
        logger.warning(f"The optimizer shard {optimizer_shard} in auto_parallel_context is not equal to the"
                       f" optimizer_shard {config.optimizer_shard} in the OpParallelConfig. Please check the "
                       f"optimizer_shard to make them consistent.")
