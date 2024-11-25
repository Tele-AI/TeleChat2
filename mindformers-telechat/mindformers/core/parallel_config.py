# Copyright 2022 Huawei Technologies Co., Ltd
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
"""Parallel Config Init."""
from mindspore.parallel._cost_model_context import _set_rp_matmul_mem_coef

from mindformers.modules.transformer.moe import default_moe_config, MoEConfig
from mindformers.modules.transformer import TransformerOpParallelConfig, TransformerRecomputeConfig
from mindformers.tools.logger import logger

default_recompute_config = TransformerRecomputeConfig()
default_parallel_config = TransformerOpParallelConfig(recompute=default_recompute_config)


def build_parallel_config(config):
    """
    Build context config.

    Args:
            - config (Union[MindFormerConfig, Dict]) - Input config.

    Returns:
            - config (Union[MindFormerConfig, Dict]) - Output config,
                with its moe_config, recompute_config and parallel_config
                are initialized from dict or assigned with a default one.
    """
    if config.moe_config:
        if not isinstance(config.moe_config, MoEConfig):
            logger.info("initial moe_config from dict: %s", config.moe_config)
            config.moe_config = MoEConfig(**config.moe_config)
    else:
        config.moe_config = default_moe_config
    if config.recompute_config:
        if not isinstance(config.recompute_config, TransformerRecomputeConfig):
            logger.info("initial recompute_config from dict: %s", config.recompute_config)
            config.recompute_config = TransformerRecomputeConfig(**config.recompute_config)
    else:
        config.recompute_config = default_recompute_config
    if config.parallel_config:
        if not isinstance(config.parallel_config, TransformerOpParallelConfig):
            logger.info("initial parallel_config from dict: %s", config.parallel_config)
            if config.parallel_config.auto_parallel or config.parallel_config.pipeline_stage > 1:
                logger.info("pipeline_stage = %s > 1, vocab_emd_dp will be reset to False.",
                            config.parallel_config.pipeline_stage)
                config.parallel_config.vocab_emb_dp = False
            _set_rp_matmul_mem_coef(config.parallel_config.pop('mem_coeff', 0.1))
            config.parallel_config = TransformerOpParallelConfig(recompute=config.recompute_config,
                                                                 **config.parallel_config)
    else:
        config.parallel_config = default_parallel_config


def reset_parallel_config(config, reset_diff_config=True):
    """
    Reset parallel config.

    Args:
        config (Union[MindFormerConfig, Dict]) - Input config.
        reset_diff_config (bool, optional): Only reset diff config. Defaults to True.
    """
    if config.moe_config and isinstance(config.moe_config, MoEConfig):
        logger.info("reset moe_config to dict: %s", config.moe_config)
        if reset_diff_config:
            config.moe_config = config.moe_config.to_diff_dict()
        else:
            config.moe_config = config.moe_config.to_dict()
    if config.recompute_config and isinstance(config.recompute_config, TransformerRecomputeConfig):
        logger.info("reset recompute_config to dict: %s", config.recompute_config)
        if reset_diff_config:
            config.recompute_config = config.recompute_config.to_diff_dict()
        else:
            config.recompute_config = config.recompute_config.to_dict()
    if config.parallel_config and isinstance(config.parallel_config, TransformerOpParallelConfig):
        logger.info("reset parallel_config to dict: %s", config.parallel_config)
        if reset_diff_config:
            config.parallel_config = config.parallel_config.to_diff_dict()
        else:
            config.parallel_config = config.parallel_config.to_dict()
