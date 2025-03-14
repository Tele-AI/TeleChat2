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
"""Llm boost utils."""
import subprocess
from mindspore.communication import get_group_size
from mindformers import logger
from mindformers.version_control import need_nz

EMBEDDING_PARALLEL_THRESHOLD = 128256


# pylint: disable=C0330, C0111
def execute_command(cmd_list):
    with subprocess.Popen(
        cmd_list, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as p:
        out, _ = p.communicate(timeout=1000)
    res = out.decode()
    return res


# pylint: disable=C0111
def is_support_lccl():
    npu_smi_info = execute_command(["npu-smi", "info", "-t", "topo"])
    legend_index = npu_smi_info.find("Legend")
    return not need_nz() and "hccs" in npu_smi_info[:legend_index].lower()


def _generate_model_sharded_state_dict(config):
    """generate harded state dict for model"""
    model_sharded_state_dict = {}
    embedding_weight_name = "model.tok_embeddings.embedding_weight"
    device_num = get_group_size()
    lm_head_name = "lm_head.weight"
    hidden_size = config.model.model_config.hidden_size
    vocab_size = config.model.model_config.vocab_size
    num_layers = config.model.model_config.num_layers
    if vocab_size >= EMBEDDING_PARALLEL_THRESHOLD:
        embedding_weight = {
            embedding_weight_name: {
                "shape": (vocab_size, hidden_size // device_num),
                "shard": (1, device_num),
            }
        }
        model_sharded_state_dict.update(embedding_weight)

    for i in range(num_layers):
        model_sharded_state_dict.update(_generate_layer_shard_state_dict(i, config))
    if not config.parallel_config.vocab_emb_dp:
        lm_head_state = {
            lm_head_name: {
                "shape": (vocab_size // device_num, hidden_size),
                "shard": (device_num, 1),
            }
        }
        model_sharded_state_dict.update(lm_head_state)
    return model_sharded_state_dict


def _generate_layer_shard_state_dict(layer_id, config):
    """generate harded state dict for each layer"""
    layer_shard_state_dict = {}
    device_num = get_group_size()
    wq_name = "attention.wq"
    wk_name = "attention.wk"
    wv_name = "attention.wv"
    wo_name = "attention.wo"
    mlp_w1_name = "feed_forward.w1"
    mlp_w2_name = "feed_forward.w2"
    mlp_w3_name = "feed_forward.w3"

    hidden_size = config.model.model_config.hidden_size
    num_heads = config.model.model_config.num_heads
    n_kv_head = (
        config.model.model_config.n_kv_head
        if config.model.model_config.n_kv_head
        else num_heads
    )
    head_dim = hidden_size // num_heads
    kv_dim = n_kv_head * head_dim
    wq_state = {
        f"model.layers.{layer_id}.{wq_name}.weight": {
            "shape": (hidden_size // device_num, hidden_size),
            "shard": (device_num, 1),
        },
        f"model.layers.{layer_id}.{wq_name}.bias": {
            "shape": (hidden_size // device_num,),
            "shard": (device_num,),
        },
    }
    wk_state = {
        f"model.layers.{layer_id}.{wk_name}.weight": {
            "shape": (kv_dim // device_num, hidden_size),
            "shard": (device_num, 1),
        },
        f"model.layers.{layer_id}.{wk_name}.bias": {
            "shape": (kv_dim // device_num,),
            "shard": (device_num,),
        },
    }
    wv_state = {
        f"model.layers.{layer_id}.{wv_name}.weight": {
            "shape": (kv_dim // device_num, hidden_size),
            "shard": (device_num, 1),
        },
        f"model.layers.{layer_id}.{wv_name}.bias": {
            "shape": (kv_dim // device_num,),
            "shard": (device_num,),
        },
    }

    wo_state = {
        f"model.layers.{layer_id}.{wo_name}.weight": {
            "shape": (hidden_size, hidden_size // device_num),
            "shard": (1, device_num),
        }
    }
    mlp_w1_state = {
        f"model.layers.{layer_id}.{mlp_w1_name}.weight": {
            "shape": (hidden_size // device_num, hidden_size),
            "shard": (device_num, 1),
        }
    }
    mlp_w2_state = {
        f"model.layers.{layer_id}.{mlp_w2_name}.weight": {
            "shape": (hidden_size, hidden_size // device_num),
            "shard": (1, device_num),
        }
    }
    mlp_w3_state = {
        f"model.layers.{layer_id}.{mlp_w3_name}.weight": {
            "shape": (hidden_size // device_num, hidden_size),
            "shard": (device_num, 1),
        }
    }

    layer_shard_state_dict.update(wq_state)
    layer_shard_state_dict.update(wk_state)
    layer_shard_state_dict.update(wv_state)
    layer_shard_state_dict.update(wo_state)
    layer_shard_state_dict.update(mlp_w1_state)
    layer_shard_state_dict.update(mlp_w2_state)
    layer_shard_state_dict.update(mlp_w3_state)
    return layer_shard_state_dict


def generate_state_dict(config):
    """Generate the sharded state dict for network"""
    state_dict = {
        "total_rank": get_group_size(),
        "stage_rank_size": get_group_size(),
        "stage": 0,
    }
    model_state_dict = {}
    model_state_dict.update(_generate_model_sharded_state_dict(config))
    state_dict["model"] = model_state_dict
    state_dict["optimizer"] = {}
    return state_dict


def save_strategy_file(state_dict, strategy_file_name):
    r"""
    Save the strategy file according to the state_dict and strategy_file_name

    Args:
        state_dict (Dict): dict with sharding metainfo
        strategy_file_name (String): the name of the target saving file

    Supported Platforms:
        ``Ascend``
    """
    import os
    import stat
    from mindspore.train.node_strategy_pb2 import ParallelStrategyMap as ckpt_strategy

    stra = ckpt_strategy()

    # pylint: disable=W0612
    total_rank = state_dict["total_rank"]
    stage_rank_size = state_dict["stage_rank_size"]
    stage = state_dict["stage"]
    model_param = state_dict["model"]
    optimizer_param = state_dict["optimizer"]
    stra.current_stage = 0
    model_param.update(optimizer_param)
    for name, item in model_param.items():
        if "shard" not in item or "shape" not in item:
            continue
        opt_weight_shard_step = (
            item["opt_weight_shard_step"]
            if "opt_weight_shard_step" in item.keys()
            else 0
        )
        opt_weight_shard_size = (
            item["opt_weight_shard_size"]
            if "opt_weight_shard_size" in item.keys()
            else 0
        )
        strategy_item = stra.parallel_strategy_item.add()
        strategy_item.node_name = name
        parallel_strategys = strategy_item.parallel_strategys
        parallel_strategys.stage = stage
        shard = item["shard"]
        shape = item["shape"]
        parallel_strategy = parallel_strategys.parallel_strategy.add()
        shard_mul = 1
        for ele in shard:
            parallel_strategy.dim.append(ele)
            shard_mul = shard_mul * ele
        layout_item = stra.parallel_layout_item.add()
        layout_item.param_name = name
        parallel_layouts = layout_item.parallel_layouts
        parallel_layouts.field = 0
        parallel_layouts.opt_weight_shard_step = opt_weight_shard_step
        parallel_layouts.opt_weight_shard_size = opt_weight_shard_size
        dev_matrix = parallel_layouts.dev_matrix.add()
        repeat_calc_num = 1
        if stage_rank_size == shard_mul:
            repeat_calc_num = 1
        elif stage_rank_size % shard_mul == 0:
            repeat_calc_num = stage_rank_size // shard_mul
        else:
            raise ValueError(
                f"For {name}, the shard{shard} requires {shard_mul} devices, "
                f"but the device number of this stage is {stage_rank_size}, "
                f"it can not be divisible by {shard_mul}"
            )
        if repeat_calc_num != 1:
            dev_matrix.dim.append(repeat_calc_num)
        for ele in shard:
            dev_matrix.dim.append(ele)
        tensor_map = parallel_layouts.tensor_map.add()
        shape_len = len(shape)
        index = shape_len - 1
        for _ in range(shape_len):
            tensor_map.dim.append(index)
            index = index - 1
        param_split_shape = parallel_layouts.param_split_shape.add()
        for ele in shape:
            param_split_shape.dim.append(ele)

    try:
        if os.path.exists(strategy_file_name):
            os.chmod(strategy_file_name, stat.S_IWUSR)
        if "/" in strategy_file_name:
            real_path = os.path.abspath(
                strategy_file_name[: strategy_file_name.rfind("/")]
            )
            os.makedirs(real_path, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT
        modes = stat.S_IWUSR | stat.S_IRUSR
        with os.fdopen(os.open(strategy_file_name, flags, modes), "wb") as f:
            f.write(stra.SerializeToString())
            os.chmod(strategy_file_name, stat.S_IRUSR)

    except BaseException as e:
        logger.critical(
            f"Failed to save the checkpoint file {strategy_file_name}. Maybe don't have "
            "the permission to write files, or the disk space is insufficient and so on."
        )
        raise e
