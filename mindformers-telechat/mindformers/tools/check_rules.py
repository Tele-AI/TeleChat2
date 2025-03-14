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
"""Functions to check rules"""
import os
import json
import yaml
from yaml.nodes import MappingNode
import mindspore as ms
from .utils import get_real_group_size
from .logger import logger


YAML_MAX_NESTING_DEPTH = 10


def get_parallel_strategy(config):
    dp = config.parallel_config.data_parallel
    mp = config.parallel_config.model_parallel
    cp = config.parallel_config.context_parallel
    pp = config.parallel_config.pipeline_stage
    return dp, mp, cp, pp


def get_device_num():
    return get_real_group_size()


def get_server_num():
    path = os.getenv('RANK_TABLE_FILE', None)
    if path is None:
        return 1
    with open(path, 'r') as json_file:
        data = json.load(json_file)
    return int(data['server_count'])


def _check_mode(config, mode, **kwargs):
    """rules with different mode"""
    if mode == 'train':
        if config.model.model_config.use_past:
            config.model.model_config.use_past = False
            logger.warning("use_past could not be used in train mode, "
                           "it has been forced to False")
        if config.metric:
            _check_keyword_gen_dataset(config, mode, **kwargs)
    elif mode == 'predict':
        _restore_net_type(config)
        _rule_bs_divisible_by_dp(config, **kwargs)
    elif mode == 'eval':
        _restore_net_type(config)
        _rule_fa_only_for_train(config, mode)
        _rule_pp_only_for_train(config, mode)

        if config.metric:
            _check_keyword_gen_dataset(config, mode, **kwargs)

        _rule_bs_divisible_by_dp(config, **kwargs)
    else:
        raise ValueError(f"mode should be in ['train', 'predict', 'eval'], but get {mode}")


def _restore_net_type(config):
    """net data type with different mode for llama2 7b"""
    if config.model.model_config.compute_dtype == 'bfloat16' and \
        config.model.model_config.param_init_type == 'float32':
        config.model.model_config.compute_dtype = 'float16'
        config.model.model_config.param_init_type = 'float16'
        logger.warning("cast compute_dtype and param_init_type to float16 for predict/eval performance")


def _rule_bs_divisible_by_dp(config, **kwargs):
    """check bs % dp == 0 when task is text_generation"""
    network = kwargs.get("network", None)
    task = kwargs.get("task", None)
    dataset = kwargs.get("dataset", None)
    if task != "text_generation":
        return
    if network is not None:
        bs = network.config.batch_size
        dp = network.config.parallel_config.data_parallel
    else:
        bs = config.model.model_config.batch_size
        dp = config.parallel_config.data_parallel
    if dataset is not None:
        bs = dataset.get_batch_size()
    bs = 1 if bs is None else bs
    if bs % dp != 0:
        raise ValueError(f"batch_size should be divisible by dp. "
                         f"But batch_size % dp = {bs} % {dp} = {bs % dp}")


def _rule_fa_only_for_train(config, mode):
    """flash attention only support training for now."""
    if config.model.model_config.use_flash_attention:
        config.model.model_config.use_flash_attention = False
        logger.warning("Flash attention only support training process for now, "
                       f"disable use_flash_attention in {mode} mode.")


def _rule_pp_only_for_train(config, mode):
    """pp only support training for now"""
    _, _, _, pp = get_parallel_strategy(config)
    if pp > 1:
        raise ValueError(f"pipeline stage only support training process for now, set pipeline stage=1 to {mode} ")


def _check_full_batch():
    """check full_batch"""
    parallel_mode = ms.get_auto_parallel_context("parallel_mode")
    full_batch = ms.get_auto_parallel_context("full_batch")
    if parallel_mode not in ["semi_auto_parallel", "auto_parallel"] and full_batch:
        ms.set_auto_parallel_context(full_batch=False)
        logger.warning(f"full_batch could only be used under semi_auto_parallel or auto_parallel, "
                       f"but get {parallel_mode}, full_batch has been forced to False")


def _check_pipeline_interleave(config, pp):
    """check vpp config"""
    pipeline_interleave_enabled = getattr(config.parallel.pipeline_config, 'pipeline_interleave', False)
    if not pipeline_interleave_enabled:
        return False
    # Set pp_interleave_num to 0 if there is no pp_interleave_num in model_config
    # or if pp_interleave_num is set to None.
    pp_interleave_num = getattr(config.model.model_config, 'pp_interleave_num', 0) or 0
    return pp_interleave_num * pp > config.model.model_config.num_layers


def _check_context_parallel_algo_valid(config, cp, mp):
    """check cp config"""
    n_kv_heads = getattr(config.model.model_config, 'n_kv_heads', None)
    multi_query_num = getattr(config.model.model_config, 'multi_query_group_num', None)
    num_heads = getattr(config.model.model_config, 'num_heads', None)
    num_attention_heads = getattr(config.model.model_config, 'num_attention_heads', None)
    n_q_heads = num_heads or num_attention_heads
    num_kv_heads = n_kv_heads or multi_query_num
    if num_kv_heads:
        n_heads = num_kv_heads
    else:
        n_heads = n_q_heads
    algo_is_ulysses_cp = config.parallel_config.context_parallel_algo.value == "ulysses_cp"
    if algo_is_ulysses_cp and n_heads is not None:
        if cp * mp > n_heads:
            raise ValueError(f"ulysses_cp and mp is shard attention head, it need cp * mp <= attention head, "
                             f"but got attention head which is {n_heads}, and cp * mp = {cp * mp},"
                             f"please check num_heads and n_kv_heads.")


def _check_parallel(config):
    """check parallel config"""
    parallel_mode = ms.get_auto_parallel_context("parallel_mode")
    dp, mp, cp, pp = get_parallel_strategy(config)
    device_num = get_device_num()
    server_num = get_server_num()
    if parallel_mode in ["semi_auto_parallel"]:
        if dp * mp * cp * pp != device_num:
            raise ValueError(f"The parallel config data_parallel * model_parallel "
                             f"* context_parallel * pipeline_stage should "
                             f"be equal to device_num, but get dp*mp*sp*pp = {dp}*{mp}*{cp}*{pp} = {dp * mp * cp * pp} "
                             f"!= device_num({device_num})")

        if config.model.model_config.num_layers and config.model.model_config.num_layers < pp:
            raise ValueError(f"num_layers of model should be greater than or equal to pipeline_stage, but get "
                             f"num_layers ({config.model.model_config.num_layers}) < pp({pp})")

        if server_num > 1:
            if server_num % pp != 0:
                logger.warning(f"server_num % pipeline_stage = {server_num} % {pp} = {server_num % pp} != 0, "
                               f"which may cause parallel error when using multiple servers")

        if config.parallel.enable_parallel_optimizer:
            if config.model.model_config.vocab_size and config.model.model_config.vocab_size % device_num != 0:
                logger.warning(f"vocab_size({config.model.model_config.vocab_size}) % device_num({device_num})"
                               f" = {config.model.model_config.vocab_size % device_num} != 0, which "
                               f"may cause the optimizer parallel of the relevant parameters to fail")
            if config.model.model_config.hidden_size and config.model.model_config.hidden_size % device_num != 0:
                logger.warning(f"hidden_size({config.model.model_config.hidden_size}) % device_num({device_num})"
                               f" = {config.model.model_config.hidden_size % device_num} != 0, which "
                               f"may cause the optimizer parallel of the relevant parameters to fail")

        if cp > 1 and not config.model.model_config.use_flash_attention:
            raise ValueError(f"context_parallel is only available for flash attention for now, but got "
                             f"use_flash_attention {config.model.model_config.use_flash_attention}, please "
                             f"set use_flash_attention=True")

        if _check_pipeline_interleave(config, pp):
            raise ValueError(f"num_layers should be greater than `pp * pp_interleave_num`, "
                             f"but got num_layers : {config.model.model_config.num_layers} "
                             f"and pp * pp_interleave_num = {pp * config.model.model_config.pp_interleave_num}.")
        if cp > 1:
            _check_context_parallel_algo_valid(config, cp, mp)


def _check_keyword_gen_dataset(config, mode, **kwargs):
    """
    check seq_len keyword_gen_dataset satisfy
    `seq_length = max_source_length + max_target_length + 1` in train_dataset or
    `seq_length = max_source_length` in eval_dataset
    """
    dataset = kwargs.get('dataset', None)
    model = kwargs.get('network', None)
    # if model or dataset was not generate from config,
    # skip this verification
    if model or dataset:
        return

    train_dataset = config.train_dataset
    eval_dataset = config.eval_dataset
    seq_length = config.model.model_config.seq_length

    def raise_error_msg(phase, max_source_length, max_target_length, dataset_phase):
        """generate error message"""
        if phase == "train" and max_source_length + max_target_length + 1 != seq_length:
            raise ValueError(f"make sure `seq_length = max_source_length + max_target_length + 1`, "
                             f"but got seq_length={seq_length}, "
                             f"max_source_length={max_source_length}, "
                             f"max_target_length={max_target_length} in {dataset_phase}.")
        if phase == "eval" and max_source_length != seq_length:
            raise ValueError(f"make sure `seq_length = max_source_length`, "
                             f"but got seq_length={seq_length}, "
                             f"max_source_length={max_source_length} in {dataset_phase}.")

    for metric_config in config.metric:
        if mode == 'train' and train_dataset and train_dataset.data_loader.type == "ADGenDataLoader":
            # verify train_dataset
            raise_error_msg("train", train_dataset.max_source_length,
                            train_dataset.max_target_length, "train_dataset")

            # when do_eval == True, using ADGENMetric, verify eval_dataset
            if config.do_eval and metric_config['type'] == "ADGENMetric":
                raise_error_msg("eval", eval_dataset.max_source_length,
                                eval_dataset.max_target_length, "eval_dataset")

            # when do_eval == True, using PerplexityMetric, verify eval_dataset
            if config.do_eval and metric_config['type'] == "PerplexityMetric":
                raise_error_msg("train", eval_dataset.max_source_length,
                                eval_dataset.max_target_length, "eval_dataset")

        if mode == 'eval' and eval_dataset and eval_dataset.data_loader.type == "ADGenDataLoader":
            # verify eval_dataset
            if metric_config['type'] == "ADGENMetric":
                raise_error_msg("eval", eval_dataset.max_source_length,
                                eval_dataset.max_target_length, "eval_dataset")

            if metric_config['type'] == "PerplexityMetric":
                raise_error_msg("train", eval_dataset.max_source_length,
                                eval_dataset.max_target_length, "eval_dataset")

        # when do_eval == True, eval_dataset should be in train mode
        if metric_config['type'] == "PerplexityMetric" and \
            eval_dataset and eval_dataset.data_loader.phase != 'train':
            logger.warning("when using 'PerplexityMetric', eval_dataset.data_loader.phase would be set to 'train'.")
            eval_dataset.data_loader.phase = 'train'
            config.eval_dataset_task.dataset_config.data_loader.phase = eval_dataset.data_loader.phase


def _check_env(config):
    """check environment"""
    fine_grain_interleave = config.model.model_config.fine_grain_interleave
    if fine_grain_interleave and fine_grain_interleave > 1:
        if os.getenv("ENABLE_LAZY_INLINE", "1") == '0':
            os.environ["ENABLE_LAZY_INLINE"] = '1'
            logger.warning(f"ENABLE_LAZY_INLINE must be set in environment when use fine_grain_interleave"
                           f" (export ENABLE_LAZY_INLINE=1)")


def _rule_recompute(pp, recompute, key):
    if isinstance(recompute, list) and len(recompute) > pp:
        if all(isinstance(n, int) for n in recompute):
            raise ValueError(f"length of {key} should be equal or less than pipeline_stage number, but get "
                             f"length of {key} ({recompute}) > pp({pp})")


def _check_recompute(config):
    if config.swap and not config.swap.swap:
        pp = config.parallel_config.pipeline_stage
        _rule_recompute(pp, config.recompute_config.recompute, "recompute")
        _rule_recompute(pp, config.recompute_config.select_recompute, "select_recompute")
        _rule_recompute(pp, config.recompute_config.select_comm_recompute, "select_comm_recompute")


def _check_config_campacity(config):
    fine_grain_interleave = config.model.model_config.fine_grain_interleave
    use_seq_parallel = config.parallel_config.use_seq_parallel
    use_context_parallel = config.parallel_config.context_parallel and config.parallel_config.context_parallel > 1
    if fine_grain_interleave and fine_grain_interleave > 1 and not use_seq_parallel and not use_context_parallel:
        raise ValueError(f"When use fine_grain_interleave without context_parallel, "
                         f"use_seq_parallel must be set to 'True'.")


def check_rules(config, mode='train', **kwargs):
    """check rules"""
    _check_mode(config, mode, **kwargs)
    _check_full_batch()
    _check_parallel(config)
    _check_env(config)
    _check_recompute(config)
    _check_config_campacity(config)


def get_yaml_ast_depth(node, depth=0):
    """Recursively calculate the maximum nesting depth of yaml ast structures."""
    if isinstance(node, MappingNode):  # process dict
        return max((get_yaml_ast_depth(v, depth + 1) for _, v in node.value), default=depth)
    return depth


def check_yaml_depth_before_loading(yaml_str, max_depth=YAML_MAX_NESTING_DEPTH):
    """Check yaml depth before loading"""
    try:
        node = yaml.compose(yaml_str)  # parse yaml to ast
        if node is None:
            return  # null file has no question
        depth = get_yaml_ast_depth(node)
        if depth > max_depth:
            raise ValueError(f"YAML nesting depth {depth} exceeds the maximum allowed value of {max_depth}")
    except yaml.YAMLError as e:
        raise ValueError(f"YAML parse error: {e}") from e
