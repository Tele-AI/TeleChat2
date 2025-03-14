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
"""
Test module for testing the paralleled llama interface used for mindformers.
"""
import argparse
import os
from types import SimpleNamespace
from functools import partial
import numpy as np

import mindspore as ms
from mindspore.dataset import GeneratorDataset
from mindformers import build_context
from mindformers.experimental.parallel_core.pynative.parallel_state import initialize_model_parallel

from tests.utils.model_tester import ModelTester
from base_model import get_config, get_model


def generate_data(seq_len, vocab_size, batch_size=4, step_num=20, use_attn_mask=False):
    """generate data for testing model."""
    input_ids = np.random.randint(
        low=0, high=vocab_size, size=(step_num * batch_size, seq_len,)).astype(np.int32)

    attention_mask = np.tril(np.ones((seq_len - 1, seq_len - 1), dtype=np.uint8))

    for input_data in input_ids:
        if use_attn_mask:
            yield input_data, np.empty(1), attention_mask
        else:
            yield input_data


def get_dataset(vocab_size, seq_length, batch_size, step_num, use_attn_mask):
    """build dataset for model training."""
    seq_length = seq_length + 1
    prepare_data = partial(generate_data,
                           seq_len=seq_length,
                           vocab_size=vocab_size,
                           batch_size=batch_size,
                           step_num=step_num,
                           use_attn_mask=use_attn_mask)

    if use_attn_mask:
        dataset = GeneratorDataset(prepare_data, column_names=['input_ids', 'position_ids', 'attention_mask'])
    else:
        dataset = GeneratorDataset(prepare_data, column_names=['input_ids'])
    dataset = dataset.batch(batch_size=batch_size)
    return dataset


def add_attr_for_pynative(config):
    """add attributes for pynative."""
    config.parallel_config.standalone_embedding_stage = True
    config.lora_config = SimpleNamespace(use_lora=False)
    config.dataset_config = SimpleNamespace(data_layout='BSH')
    config.apply_residual_connection_post_norm = False
    config.seq_len_interpolation_factor = 1.0
    config.residual_connection_dtype = None
    config.kv_channels = config.hidden_size // config.num_heads
    config.param_init_dtype = config.params_dtype
    config.use_gqa = False
    config.parallel_config.use_sequence_parallel = False
    config.parallel_config.sequence_parallel = False
    config.parallel_config.deterministic_mode = False
    config.parallel_config.use_cpu_initialization = False
    config.parallel_config.recompute_config = None
    config.parallel_config.expert_model_parallel_size = 1
    config.use_flash_attention = True
    config.qkv_has_bias = False
    config.bias_init = 'zeros'
    config.parallel_config.zero_level = None
    config.init_method = 'normal'
    config.parallel_config.gradient_accumulation_fusion = False
    config.attention_dropout_rate = config.hidden_dropout
    config.hidden_dropout_rate = config.hidden_dropout
    config.out_proj_has_bias = False
    config.mlp_has_bias = False
    config.recompute_granularity = None
    config.recompute_method = None
    config.recompute_num_layers = None
    config.moe_config.num_experts = 1
    config.out_hidden_size = None
    config.use_retriever = False
    config.use_final_norm = True
    config.vocab_size = 32000
    config.embedding_init_dtype = config.embedding_init_type
    config.parallel_position_embedding = False
    config.rotary_interleaved = False
    config.rotary_base = 10000
    config.use_sequence_parallel = False
    config.retro_add_retriever = False
    config.gradient_accumulation_fusion = False
    config.fp32_residual_connection = False
    config.clone_scatter_output_in_embedding = False
    config.transformer_impl = None
    config.distribute_saved_activations = False
    config.fp8 = None
    config.encoder_num_layers = None
    config.decoder_num_layers = None
    config.norm_epsilon = config.rms_norm_eps
    config.enable_flash_sp = False
    config.expert_model_parallel_size = 1
    config.select_comm_recompute = False
    config.masked_softmax_fusion = False
    config.bias_dropout_fusion = False
    config.select_recompute = False
    config.apply_rope_fusion = False
    config.use_sandwich_norm = False
    fa_config = {
        'pre_tokens': 65536,
        'next_tokens': 0,
        'sparse_mode': 0,
        'input_layout': 'BNSD',
    }
    config.fa_config = fa_config


def _save_or_load_ckpt(model, save_ckpt: bool, load_ckpt: bool, ckpt_rel_path: str = 'ckpt/llama_single_train.ckpt'):
    """save or load checkpoint"""
    if save_ckpt and load_ckpt:
        raise ValueError("Unexpected argument value: save_ckpt and load_ckpt are both True.")

    if save_ckpt or load_ckpt:
        ckpt_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), ckpt_rel_path)
        os.makedirs(os.path.dirname(ckpt_file_path), exist_ok=True)
    else:
        ckpt_file_path = None

    if save_ckpt:
        ms.save_checkpoint(model, ckpt_file_path)
    if load_ckpt:
        ms.load_checkpoint(ckpt_file_path, model)


def _base_train(args, loss_std, base_parallel_config, new_parallel_config=None):
    """base llama train"""
    if args.experiment_mode is None or args.save_ckpt is None or args.load_ckpt is None:
        raise ValueError("Unexpected argument value: any of experiment_mode or save_ckpt or load_ckpt is None.")

    runner = ModelTester(run_mode='train', batch_size=2, experiment_mode=True, **base_parallel_config)
    runner.args.mode = args.ms_run_mode
    build_context(runner.args)

    model_config = get_config()
    model_config.parallel_config = runner.args.get_parallel_config()

    if runner.args.mode == 1:
        tensor_parallel = new_parallel_config.tensor_parallel \
            if new_parallel_config is not None and hasattr(new_parallel_config, 'tensor_parallel') else 1
        initialize_model_parallel(tensor_model_parallel_size=tensor_parallel)
        add_attr_for_pynative(model_config)
        use_attn_mask = True
    else:
        use_attn_mask = False

    dataset = get_dataset(model_config.vocab_size, model_config.seq_length, runner.batch_size,
                          runner.step_num, use_attn_mask)
    if new_parallel_config is not None:
        new_parallel_config.update(**base_parallel_config)
        for key, value in new_parallel_config.items():
            model_config.__setattr__(key, value)
    model = get_model(model_config)

    _save_or_load_ckpt(model, args.save_ckpt, args.load_ckpt)
    runner.set_train(model, model_config, loss_std=loss_std, dataset=dataset)


def single_train(args):
    """test llama train in single"""
    base_parallel_config = {
        'use_parallel': False,
        'data_parallel': 1,
        'use_seq_parallel': True,
        'enable_parallel_optimizer': True
    }

    if args.ms_run_mode == 1:
        base_parallel_config['use_parallel'] = True
        base_parallel_config['parallel_mode'] = 2
        new_loss_std = [10.643806, 10.634241, 10.627177, 10.639130, 10.627960,
                        10.623453, 10.620171, 10.632998, 10.622291, 10.624990,
                        10.623909, 10.634945, 10.619917, 10.616154, 10.634602,
                        10.640321, 10.632576, 10.645661, 10.613758, 10.614880]
    else:
        new_loss_std = [10.626314, 10.624199, 10.631677, 10.629707, 10.635541,
                        10.630877, 10.616421, 10.628949, 10.628935, 10.622899,
                        10.635184, 10.622277, 10.640517, 10.625608, 10.629306,
                        10.631349, 10.633284, 10.635038, 10.631706, 10.633798]

    old_loss_std = [10.644210, 10.639138, 10.638718, 10.633285, 10.619514,
                    10.636455, 10.618894, 10.630442, 10.612017, 10.634727,
                    10.624849, 10.622943, 10.633147, 10.631532, 10.634797,
                    10.627925, 10.634336, 10.638637, 10.636755, 10.631924]

    # If this code passes the tests and is merged into `dev`,
    # the `new_loss` should be used as the benchmark for test cases.
    if args.use_new_loss:
        loss_std = new_loss_std
    else:
        loss_std = old_loss_std

    _base_train(args, loss_std, base_parallel_config)


def parallel_train_dp2(args):
    """test llama train in dp2"""
    base_parallel_config = {
        'use_parallel': True,
        'data_parallel': 2,
        'use_seq_parallel': True,
        'enable_parallel_optimizer': True
    }

    if args.ms_run_mode == 1:
        base_parallel_config['parallel_mode'] = 2
        loss_std = [10.643806, 10.634241, 10.627187, 10.639118, 10.627972,
                    10.623462, 10.620193, 10.633015, 10.622290, 10.624989,
                    10.623926, 10.634968, 10.619913, 10.616154, 10.634594,
                    10.640324, 10.632589, 10.645652, 10.613745, 10.614884]
    else:
        loss_std = [10.624823, 10.626235, 10.635187, 10.623976, 10.627350,
                    10.622116, 10.623417, 10.628310, 10.627144, 10.624737,
                    10.622215, 10.610016, 10.618230, 10.605141, 10.615884,
                    10.599818, 10.616759, 10.620478, 10.610780, 10.614862]

    new_parallel_config = {
        'data_parallel': 2
    }

    _base_train(args, loss_std, base_parallel_config, new_parallel_config)


def parallel_train_mp2(args):
    """test llama train in mp2"""
    base_parallel_config = {
        'use_parallel': True,
        'data_parallel': 1,
        'model_parallel': 2,
        'use_seq_parallel': True,
        'enable_parallel_optimizer': True
    }

    if args.ms_run_mode == 1:
        base_parallel_config['parallel_mode'] = 2
        loss_std = [10.643806, 10.634238, 10.627186, 10.639116, 10.627967,
                    10.623456, 10.620173, 10.633016, 10.622282, 10.624996,
                    10.623921, 10.634953, 10.619897, 10.616146, 10.634610,
                    10.640322, 10.632586, 10.645667, 10.613750, 10.614882]
    else:
        loss_std = [10.624356, 10.634629, 10.623917, 10.621689, 10.623317,
                    10.607912, 10.608279, 10.621058, 10.618675, 10.612767,
                    10.610808, 10.607621, 10.610729, 10.600761, 10.600204,
                    10.586555, 10.591313, 10.595157, 10.594136, 10.600330]

    new_parallel_config = {
        'tensor_parallel': 2
    }

    _base_train(args, loss_std, base_parallel_config, new_parallel_config)


def parallel_train_dp2_mp2(args):
    """test llama train in dp2mp2"""
    base_parallel_config = {
        'use_parallel': True,
        'data_parallel': 2,
        'model_parallel': 2,
        'use_seq_parallel': True,
        'enable_parallel_optimizer': True
    }

    if args.ms_run_mode == 1:
        base_parallel_config['parallel_mode'] = 2
        loss_std = [10.643806, 10.634238, 10.627181, 10.639125, 10.627966,
                    10.623468, 10.620173, 10.633016, 10.622279, 10.624969,
                    10.623915, 10.634941, 10.619912, 10.616146, 10.634592,
                    10.640333, 10.632586, 10.645665, 10.613761, 10.614880]
    else:
        loss_std = [10.626953, 10.626993, 10.619582, 10.626608, 10.636086,
                    10.626451, 10.625716, 10.638491, 10.616043, 10.610565,
                    10.617392, 10.608209, 10.630942, 10.613029, 10.611053,
                    10.607447, 10.618746, 10.607036, 10.602592, 10.620325]

    new_parallel_config = {
        'tensor_parallel': 2
    }

    _base_train(args, loss_std, base_parallel_config, new_parallel_config)


TEST_MAP = {
    'single_train': single_train,
    'parallel_train_dp2': parallel_train_dp2,
    'parallel_train_mp2': parallel_train_mp2,
    'parallel_train_dp2_mp2': parallel_train_dp2_mp2
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='test mode of llama model.')
    parser.add_argument('--experiment-mode', action='store_true', help='whether to use experiment mode.')
    parser.add_argument('--save-ckpt', action='store_true', help='whether to save checkpoint.')
    parser.add_argument('--load-ckpt', action='store_true', help='whether to load checkpoint.')
    parser.add_argument('--use-new-loss', action='store_true', help='whether to use new loss.')
    parser.add_argument('--ms-run-mode', type=int, default=0, required=False,
                        help='MindSpore run mode. 0 for Graph mode, 1 for Pynative mode.')

    args_ = parser.parse_args()
    if args_.mode not in TEST_MAP:
        raise ValueError(f"Unsupported test mode: {args_.mode}")
    TEST_MAP[args_.mode](args_)
