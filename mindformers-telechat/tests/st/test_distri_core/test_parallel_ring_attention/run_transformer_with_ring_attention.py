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
""" Test ParallelTransformer. """
import os
import argparse
from typing import Optional
import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds
from mindspore import Tensor
import mindspore.common.dtype as mstype
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.communication.management import init
from mindspore.communication.comm_func import all_gather_into_tensor

from mindformers.core import optim
from mindformers.modules import FeedForward
from mindformers.models.llama.llama_transformer import LLamaAttention
from mindformers.models.llama.llama_layer import LlamaRMSNorm
from mindformers.modules.layers import FreqsMgr
from mindformers.modules.transformer.op_parallel_config import default_dpmp_config
from mindformers.modules.transformer.transformer import (
    default_transformer_recompute_config,
)
from mindformers.experimental.parallel_core.pynative.config import (
    ModelParallelConfig,
    TransformerConfig,
)
from mindformers.experimental.parallel_core.pynative.parallel_state import initialize_model_parallel
from mindformers.experimental.parallel_core.pynative.transformer import ParallelTransformer
from mindformers.experimental.parallel_core.pynative.transformer.rotary_pos_embedding import (
    RotaryEmbedding,
)
from mindformers.experimental.parallel_core.pynative.parallel_state import (
    get_data_parallel_world_size,
    get_data_parallel_rank,
    get_context_parallel_world_size,
    get_context_parallel_group,
    get_data_parallel_group,
)

from mindformers.experimental.parallel_core.pynative.context_parallel.utils import (
    get_batch_on_this_cp_rank_with_ringattention,
    get_batch_on_this_cp_rank_with_flashsp,
)

from mindformers.experimental.parallel_core.pynative.optimizer.zero.adamw_zero import AdamW
from tests.st.test_distri_core.utils import (
    TestData,
    train,
    transform_transformerlayer_params,
    generate_ckpt,
)


class LLamaDecodeLayer(nn.Cell):
    """LLamaDecodeLayer."""

    def __init__(self,
                 seq_length,
                 layer_id,
                 dim: int = 512,
                 ffn_dim: int = 2048,
                 hidden_act: str = "gelu",
                 n_heads: int = 8,
                 n_kv_heads: Optional[int] = None,
                 norm_eps: float = 1e-5,
                 qkv_concat=False,
                 compute_dtype=mstype.float32,
                 layernorm_compute_dtype=mstype.float32,
                 softmax_compute_dtype=mstype.float32,
                 rotary_dtype=mstype.float32,
                 param_init_type=mstype.float32,
                 use_past=False,
                 parallel_config=default_dpmp_config,
                 ):
        super().__init__()
        self.seq_length = seq_length
        self.layer_id = layer_id
        self.hidden_size = dim
        self.n_head = n_heads
        self.head_dim = self.hidden_size // self.n_head
        self.n_kv_head = n_heads if n_kv_heads is None else n_kv_heads
        self.dtype = compute_dtype
        self.is_first_iteration = True
        self.use_past = use_past

        self.shape = ops.Shape()
        self.reshape = ops.Reshape().add_prim_attr("skip_redistribution", True)
        self.add = ops.Add()
        self.ffn_norm = LlamaRMSNorm(
            self.hidden_size, norm_eps, compute_type=layernorm_compute_dtype
        )
        self.attention_norm = LlamaRMSNorm(
            self.hidden_size, norm_eps, compute_type=layernorm_compute_dtype
        )
        self.attention = LLamaAttention(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            qkv_concat=qkv_concat,
            compute_dtype=compute_dtype,
            softmax_compute_dtype=softmax_compute_dtype,
            rotary_dtype=rotary_dtype,
            param_init_type=param_init_type,
            qkv_has_bias=True,
            use_past=use_past,
            use_flash_attention=True,
            parallel_config=parallel_config,
        )

        self.mlp = FeedForward(
            hidden_size=dim,
            ffn_hidden_size=ffn_dim,
            dropout_rate=0.0,
            hidden_act=hidden_act,
            param_init_type=param_init_type,
            parallel_config=parallel_config,
            compute_dtype=compute_dtype,
        )

    def construct(self, x, freqs_cis, attention_mask):
        """Forward of transformer block."""
        # [bs, seq/1, hidden_dim]
        input_x = self.attention_norm(x)
        # [bs, seq/1, hidden_dim]
        h = self.attention(input_x, freqs_cis, attention_mask)
        h = self.add(x, h)
        ffn_norm = self.ffn_norm(h)
        # [bs, seq/1, hidden_dim]
        ffn_out = self.mlp(ffn_norm)
        # [bs, seq/1, hidden_dim] or [bs * seq/1, hidden_dim]
        out = self.add(h, ffn_out)

        return out


class LlamaTransformerNet(nn.Cell):
    """LlamaTransformerNet."""

    def __init__(self,
                 num_layers,
                 seq_length,
                 hidden_size,
                 ffn_hidden_size,
                 num_attention_heads,
                 qkv_concat=True,
                 parallel_config=default_dpmp_config,
                 ):
        super(LlamaTransformerNet, self).__init__()
        self.freqs_mgr = FreqsMgr(
            head_dim=hidden_size // num_attention_heads,
            seq_length=seq_length,
            rotary_dtype=mstype.float32,
        )
        self.transformer = nn.CellList()
        self.num_layers = num_layers
        for i in range(self.num_layers):
            layer = LLamaDecodeLayer(
                seq_length=seq_length,
                layer_id=i,
                dim=hidden_size,
                ffn_dim=ffn_hidden_size,
                n_heads=num_attention_heads,
                qkv_concat=qkv_concat,
                parallel_config=parallel_config,
            )
            self.transformer.append(layer)
        self.loss = SoftmaxCrossEntropyWithLogits()

    def construct(self, x, attention_mask, label):
        """construct."""
        freqs_cis = self.freqs_mgr(x.shape[1])
        h = x
        for i in range(self.num_layers):
            h = self.transformer[i](h, freqs_cis, attention_mask)
        output = ops.sum(h, dim=-1, keepdim=False)
        loss = self.loss(output, label)
        return loss


class ParallelTransformerNet(nn.Cell):
    """ParallelTransformerNet."""

    def __init__(self, config, with_rope=False):
        super(ParallelTransformerNet, self).__init__()
        self.with_rope = with_rope
        if with_rope:
            self.rope = RotaryEmbedding(
                kv_channels=config.hidden_size // config.num_attention_heads, rotary_percent=1.0
            )
        self.transformer = ParallelTransformer(config=config, post_norm=False)
        self.loss = SoftmaxCrossEntropyWithLogits(reduction="none")
        if get_context_parallel_world_size() > 1:
            self.cp_group = get_context_parallel_group()

    def construct(self, x, attention_mask, labels):
        """construct."""
        if self.with_rope:
            emb = self.rope(max_seq_len=x.shape[1])
            output = self.transformer(x, attention_mask, rotary_pos_emb=emb)
        else:
            output = self.transformer(x, attention_mask)
        output = ops.sum(output, dim=-1, keepdim=False)

        seq_dim = 1
        if get_context_parallel_world_size() > 1:
            output = self.allgather(output)
            split_outputs = ops.split(
                Tensor(output), output.shape[0] // get_context_parallel_world_size(), axis=0
            )
            output = ops.cat(split_outputs, axis=seq_dim)

            labels = all_gather_into_tensor(labels, group=self.cp_group)[0]
            split_labels = ops.split(
                Tensor(labels), labels.shape[0] // get_context_parallel_world_size(), axis=0
            )
            labels = ops.cat(split_labels, axis=seq_dim)

        loss = self.loss(output, labels)
        return loss


def _count_unequal_element(data_expected, data_me, rtol, atol):
    """Statistics error location and ratio"""
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_me) * rtol)
    nan_diff = np.not_equal(np.isnan(data_expected), np.isnan(data_me))
    inf_diff = np.not_equal(np.isinf(data_expected), np.isinf(data_me))
    neginf_diff = np.not_equal(np.isneginf(data_expected), np.isneginf(data_me))
    greater = greater + nan_diff + inf_diff + neginf_diff
    loss_count = np.count_nonzero(greater)
    print(
        "data_expected_std:{0}\ndata_me_error:{1}\nloss:{2}\nerror_percent_num:{3}\nerror_percent_max:{4}".format(
            data_expected[greater],
            data_me[greater],
            error[greater],
            str(loss_count / total_count),
            np.max(np.abs(error[greater] / data_me[greater])),
        )
    )
    assert (
        loss_count / total_count
    ) < rtol, "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}\nerror_percent:{3}".format(
        data_expected[greater],
        data_me[greater],
        error[greater],
        str(loss_count / total_count),
    )


def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    if not np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan):
        _count_unequal_element(data_expected, data_me, rtol, atol)
    else:
        assert np.array(data_expected).shape == np.array(data_me).shape


def generate_golden_with_flashattention():
    """Generate golden."""
    os.environ["GRAPH_OP_RUN"] = "1"
    batch_size = 8
    dataset_size = 128  # 16 steps
    num_layers = 1
    seq_length = 4096
    num_attention_heads = 4
    hidden_size = 128

    ms.set_context(
        device_target="Ascend",
        mode=ms.GRAPH_MODE,
        deterministic="ON",
        pynative_synchronize=True,
    )
    init()

    ms.reset_auto_parallel_context()
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.STAND_ALONE)

    ms.set_seed(2024)
    input_data = np.random.random((dataset_size, seq_length, hidden_size)).astype(
        np.float32
    )
    label_data = np.zeros((dataset_size, seq_length)).astype(np.float32)
    dataset = TestData(
        input_data=input_data, label_data=label_data, with_attn_mask=True
    )
    dataset = ds.GeneratorDataset(
        dataset, column_names=["input_ids", "labels", "attention_mask"], shuffle=False
    )
    dataset = dataset.batch(batch_size)

    default_dpmp_config.recompute = default_transformer_recompute_config
    network = LlamaTransformerNet(
        num_layers=num_layers,
        seq_length=seq_length,
        hidden_size=hidden_size,
        ffn_hidden_size=4 * hidden_size,
        num_attention_heads=num_attention_heads,
        qkv_concat=True,
        parallel_config=default_dpmp_config,
    )
    param_dict = generate_ckpt(
        hidden_size=hidden_size, module_type="transformer", num_layers=num_layers
    )
    ms.load_param_into_net(network, param_dict)

    save_golden = True
    if save_golden:
        ms.save_checkpoint(network, "transformer_golden.ckpt")

    optimizer = optim.AdamW(params=network.get_parameters())
    losses = train(1, dataset, network, optimizer, None, with_attn_input=True)

    print("losses:", losses)
    np.save("golden_losses", losses)


def run_parallel_transformer_with_tp(tp=2):
    """Test ParallelTransformer."""
    batch_size = 8
    dataset_size = 128
    num_layers = 1
    seq_length = 4096
    num_attention_heads = 4
    hidden_size = 128
    tensor_parallel = tp

    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic="ON")

    init()
    initialize_model_parallel(tensor_model_parallel_size=tensor_parallel)

    ms.set_seed(2024)
    input_data = np.random.random((dataset_size, seq_length, hidden_size)).astype(
        np.float32
    )
    label_data = np.zeros((dataset_size, seq_length)).astype(np.float32)
    dataset = TestData(
        input_data=input_data, label_data=label_data, with_attn_mask=True
    )
    dataset = ds.GeneratorDataset(
        dataset, column_names=["input_ids", "labels", "attention_mask"], shuffle=False
    )
    dataset = dataset.batch(batch_size)

    parallel_config = ModelParallelConfig(tensor_model_parallel_size=tensor_parallel)
    config = TransformerConfig(
        seq_length=seq_length,
        vocab_size=1,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        hidden_size=hidden_size,
        attention_type="self_attn",
        qkv_has_bias=True,
        out_proj_has_bias=False,
        parallel_config=parallel_config,
        params_dtype="float32",
        compute_dtype="float32",
        softmax_compute_dtype="float32",
        hidden_dropout=0.0,
        attention_dropout=0.0,
        mask_func_type="attn_mask_add",
        mlp_has_bias=True,
        ffn_hidden_size=4 * hidden_size,
        hidden_act="gelu",
        apply_residual_connection_post_norm=False,
        normalization="FusedRMSNorm",
        norm_epsilon=1.0e-5,
        use_flash_attention=True,
        fa_config={
            "pre_tokens": 65536,
            "next_tokens": 0,
            "sparse_mode": 0,
        },
    )
    network = ParallelTransformerNet(config=config, with_rope=True)
    save_golden = True
    if save_golden:
        golden_ckpt_path = f"transformer_golden.ckpt"
        assert os.path.exists(golden_ckpt_path), (
            "'golden.ckpt' did not exits, please run generate_golden() to generate one by running below command: \n"
            + "`pytest -sv test_transformer.py::TestParallelTransformer::generate_golden`"
        )
        golden_params = ms.load_checkpoint(golden_ckpt_path)
        pynative_params = transform_transformerlayer_params(
            golden_params,
            hidden_size=hidden_size,
            kv_hidden_size=None,
            prefix="transformer.layers.",
        )
        param_not_load, _ = ms.load_param_into_net(network, pynative_params)
        assert (
            not param_not_load
        ), f"{param_not_load} was not loaded in this net, test failed."
    else:
        param_dict = generate_ckpt(
            hidden_size=hidden_size, module_type="transformer", num_layers=num_layers
        )
        pynative_params = transform_transformerlayer_params(
            param_dict,
            hidden_size=hidden_size,
            kv_hidden_size=None,
            prefix="transformer.layers.",
        )
        ms.load_param_into_net(network, pynative_params)

    input_ids = Tensor(shape=(None, None, None), dtype=mstype.float32)
    labels = Tensor(shape=(None, None), dtype=mstype.float32)
    attn_mask = Tensor(shape=(None, None), dtype=mstype.float32)
    network.set_inputs(input_ids, labels, attn_mask)

    optimizer = optim.AdamW(params=network.get_parameters())

    losses = train(1, dataset, network, optimizer, None, with_attn_input=True)
    print("losses:", losses)
    golden_losses = np.load("golden_losses.npy")
    print("golden_losses:", golden_losses)
    losses = np.array(losses)

    tols = dict(atol=1.0e-3, rtol=1.0e-3)
    allclose_nparray(golden_losses, losses, **tols)

    print("test passed!")


def run_parallel_transformer_with_cp(sp=8, enable_flash_sp=False):
    """Test ParallelTransformer."""

    golden_batch_size = 8
    batch_size = golden_batch_size

    dataset_size = 128
    num_layers = 1
    seq_length = 4096
    num_attention_heads = 4
    hidden_size = 128

    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic="ON")

    init()
    initialize_model_parallel(context_parallel_size=sp)

    ms.set_seed(2024)

    input_data = np.random.random((dataset_size, seq_length, hidden_size)).astype(
        np.float32
    )
    label_data = np.zeros((dataset_size, seq_length)).astype(np.float32)
    dataset = TestData(
        input_data=input_data, label_data=label_data, with_attn_mask=True
    )
    dataset = ds.GeneratorDataset(
        dataset,
        column_names=["input_ids", "labels", "attention_mask"],
        num_shards=get_data_parallel_world_size(),
        shard_id=get_data_parallel_rank(),
        shuffle=False,
    )
    dataset = dataset.batch(batch_size)

    if not enable_flash_sp:
        dataset = dataset.map(
            operations=get_batch_on_this_cp_rank_with_ringattention,
            input_columns=["input_ids", "labels", "attention_mask"],
            output_columns=["input_ids", "labels", "attention_mask"],
        )
    else:
        dataset = dataset.map(
            operations=get_batch_on_this_cp_rank_with_flashsp,
            input_columns=["input_ids", "labels", "attention_mask"],
            output_columns=["input_ids", "labels", "attention_mask"],
        )

    parallel_config = ModelParallelConfig()
    config = TransformerConfig(
        seq_length=seq_length,
        vocab_size=1,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        hidden_size=hidden_size,
        attention_type="self_attn",
        qkv_has_bias=True,
        out_proj_has_bias=False,
        parallel_config=parallel_config,
        params_dtype="float32",
        compute_dtype="float32",
        softmax_compute_dtype="float32",
        hidden_dropout=0.0,
        attention_dropout=0.0,
        mask_func_type="attn_mask_add",
        mlp_has_bias=True,
        ffn_hidden_size=4 * hidden_size,
        hidden_act="gelu",
        apply_residual_connection_post_norm=False,
        normalization="FusedRMSNorm",
        norm_epsilon=1.0e-5,
        use_flash_attention=True,
        fa_config={
            "pre_tokens": 65536,
            "next_tokens": 0,
            "sparse_mode": 0,
        },
        enable_flash_sp=enable_flash_sp,
    )

    network = ParallelTransformerNet(config=config, with_rope=True)
    save_golden = True
    if save_golden:
        golden_ckpt_path = f"transformer_golden.ckpt"
        assert os.path.exists(golden_ckpt_path), (
            "'golden.ckpt' did not exits, please run generate_golden() to generate one by running below command: \n"
            + "`pytest -sv test_transformer.py::TestParallelTransformer::generate_golden`"
        )
        golden_params = ms.load_checkpoint(golden_ckpt_path)
        pynative_params = transform_transformerlayer_params(
            golden_params,
            hidden_size=hidden_size,
            kv_hidden_size=None,
            prefix="transformer.layers.",
        )
        param_not_load, _ = ms.load_param_into_net(network, pynative_params)
        assert (
            not param_not_load
        ), f"{param_not_load} was not loaded in this net, test failed."
    else:
        param_dict = generate_ckpt(
            hidden_size=hidden_size, module_type="transformer", num_layers=num_layers
        )
        pynative_params = transform_transformerlayer_params(
            param_dict,
            hidden_size=hidden_size,
            kv_hidden_size=None,
            prefix="transformer.layers.",
        )
        ms.load_param_into_net(network, pynative_params)

    input_ids = Tensor(shape=(None, None, None), dtype=mstype.float32)
    labels = Tensor(shape=(None, None), dtype=mstype.float32)
    attn_mask = Tensor(shape=(None, None), dtype=mstype.float32)
    network.set_inputs(input_ids, labels, attn_mask)

    optimizer = optim.AdamW(params=network.get_parameters())
    losses = train(1, dataset, network, optimizer, None, with_attn_input=True)
    print("losses:", losses)
    golden_losses = np.load("golden_losses.npy")
    print("golden_losses:", golden_losses)
    losses = np.array(losses)

    tols = dict(atol=5.0e-3, rtol=5.0e-3)
    allclose_nparray(golden_losses, losses, **tols)
    print("test passed!")


def run_parallel_transformer_with_cp_and_dp(sp=8, enable_flash_sp=False):
    """Test ParallelTransformer."""

    golden_batch_size = 8
    total_cards = 8
    dp = total_cards // sp
    batch_size = golden_batch_size // dp

    dataset_size = 128
    num_layers = 1
    seq_length = 4096
    num_attention_heads = 4
    hidden_size = 128

    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic="ON")

    init()
    initialize_model_parallel(context_parallel_size=sp)

    ms.set_seed(2024)

    input_data = np.random.random((dataset_size, seq_length, hidden_size)).astype(
        np.float32
    )
    label_data = np.zeros((dataset_size, seq_length)).astype(np.float32)
    dataset = TestData(
        input_data=input_data, label_data=label_data, with_attn_mask=True
    )
    dataset = ds.GeneratorDataset(
        dataset,
        column_names=["input_ids", "labels", "attention_mask"],
        num_shards=get_data_parallel_world_size(),
        shard_id=get_data_parallel_rank(),
        shuffle=False,
    )
    dataset = dataset.batch(batch_size)

    if not enable_flash_sp:
        dataset = dataset.map(
            operations=get_batch_on_this_cp_rank_with_ringattention,
            input_columns=["input_ids", "labels", "attention_mask"],
            output_columns=["input_ids", "labels", "attention_mask"],
        )
    else:
        dataset = dataset.map(
            operations=get_batch_on_this_cp_rank_with_flashsp,
            input_columns=["input_ids", "labels", "attention_mask"],
            output_columns=["input_ids", "labels", "attention_mask"],
        )

    parallel_config = ModelParallelConfig()
    config = TransformerConfig(
        seq_length=seq_length,
        vocab_size=1,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        hidden_size=hidden_size,
        attention_type="self_attn",
        qkv_has_bias=True,
        out_proj_has_bias=False,
        parallel_config=parallel_config,
        params_dtype="float32",
        compute_dtype="float32",
        softmax_compute_dtype="float32",
        hidden_dropout=0.0,
        attention_dropout=0.0,
        mask_func_type="attn_mask_add",
        mlp_has_bias=True,
        ffn_hidden_size=4 * hidden_size,
        hidden_act="gelu",
        apply_residual_connection_post_norm=False,
        normalization="FusedRMSNorm",
        norm_epsilon=1.0e-5,
        use_flash_attention=True,
        fa_config={
            "pre_tokens": 65536,
            "next_tokens": 0,
            "sparse_mode": 0,
        },
        enable_flash_sp=enable_flash_sp,
    )

    network = ParallelTransformerNet(config=config, with_rope=True)
    save_golden = True
    if save_golden:
        golden_ckpt_path = f"transformer_golden.ckpt"
        assert os.path.exists(golden_ckpt_path), (
            "'golden.ckpt' did not exits, please run generate_golden() to generate one by running below command: \n"
            + "`pytest -sv test_transformer.py::TestParallelTransformer::generate_golden`"
        )
        golden_params = ms.load_checkpoint(golden_ckpt_path)
        pynative_params = transform_transformerlayer_params(
            golden_params,
            hidden_size=hidden_size,
            kv_hidden_size=None,
            prefix="transformer.layers.",
        )
        param_not_load, _ = ms.load_param_into_net(network, pynative_params)
        assert (
            not param_not_load
        ), f"{param_not_load} was not loaded in this net, test failed."
    else:
        param_dict = generate_ckpt(
            hidden_size=hidden_size, module_type="transformer", num_layers=num_layers
        )
        pynative_params = transform_transformerlayer_params(
            param_dict,
            hidden_size=hidden_size,
            kv_hidden_size=None,
            prefix="transformer.layers.",
        )
        ms.load_param_into_net(network, pynative_params)

    input_ids = Tensor(shape=(None, None, None), dtype=mstype.float32)
    labels = Tensor(shape=(None, None), dtype=mstype.float32)
    attn_mask = Tensor(shape=(None, None), dtype=mstype.float32)
    network.set_inputs(input_ids, labels, attn_mask)

    optimizer = optim.AdamW(params=network.get_parameters())

    losses = train(1, dataset, network, optimizer, None, with_attn_input=True)
    print("losses:", losses)
    golden_losses = np.load("golden_losses.npy")
    golden_losses = golden_losses[:, get_data_parallel_rank(): golden_losses.shape[1]: dp]
    print("golden_losses:", golden_losses)
    losses = np.array(losses)

    tols = dict(atol=2.0e-3, rtol=2.0e-3)
    allclose_nparray(golden_losses, losses, **tols)
    print("test passed!")


def run_parallel_transformer_with_cp_and_zero1(sp=8, enable_flash_sp=False):
    """Test ParallelTransformer."""
    golden_batch_size = 8
    total_cards = 8
    dp = total_cards // sp
    batch_size = golden_batch_size // dp

    dataset_size = 128
    num_layers = 1
    seq_length = 4096
    num_attention_heads = 4
    hidden_size = 128

    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic="ON")

    init()
    initialize_model_parallel(context_parallel_size=sp)

    ms.set_seed(2024)

    input_data = np.random.random((dataset_size, seq_length, hidden_size)).astype(
        np.float32
    )
    label_data = np.zeros((dataset_size, seq_length)).astype(np.float32)
    dataset = TestData(
        input_data=input_data, label_data=label_data, with_attn_mask=True
    )
    dataset = ds.GeneratorDataset(
        dataset,
        column_names=["input_ids", "labels", "attention_mask"],
        num_shards=get_data_parallel_world_size(),
        shard_id=get_data_parallel_rank(),
        shuffle=False,
    )
    dataset = dataset.batch(batch_size)

    if not enable_flash_sp:
        dataset = dataset.map(
            operations=get_batch_on_this_cp_rank_with_ringattention,
            input_columns=["input_ids", "labels", "attention_mask"],
            output_columns=["input_ids", "labels", "attention_mask"],
        )
    else:
        dataset = dataset.map(
            operations=get_batch_on_this_cp_rank_with_flashsp,
            input_columns=["input_ids", "labels", "attention_mask"],
            output_columns=["input_ids", "labels", "attention_mask"],
        )

    parallel_config = ModelParallelConfig()
    config = TransformerConfig(
        seq_length=seq_length,
        vocab_size=1,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        hidden_size=hidden_size,
        attention_type="self_attn",
        qkv_has_bias=True,
        out_proj_has_bias=False,
        parallel_config=parallel_config,
        params_dtype="float32",
        compute_dtype="float32",
        softmax_compute_dtype="float32",
        hidden_dropout=0.0,
        attention_dropout=0.0,
        mask_func_type="attn_mask_add",
        mlp_has_bias=True,
        ffn_hidden_size=4 * hidden_size,
        hidden_act="gelu",
        apply_residual_connection_post_norm=False,
        normalization="FusedRMSNorm",
        norm_epsilon=1.0e-5,
        use_flash_attention=True,
        fa_config={
            "pre_tokens": 65536,
            "next_tokens": 0,
            "sparse_mode": 0,
        },
        enable_flash_sp=enable_flash_sp,
    )

    network = ParallelTransformerNet(config=config, with_rope=True)
    save_golden = True
    if save_golden:
        golden_ckpt_path = f"transformer_golden.ckpt"
        assert os.path.exists(golden_ckpt_path), (
            "'golden.ckpt' did not exits, please run generate_golden() to generate one by running below command: \n"
            + "`pytest -sv test_transformer.py::TestParallelTransformer::generate_golden`"
        )
        golden_params = ms.load_checkpoint(golden_ckpt_path)
        pynative_params = transform_transformerlayer_params(
            golden_params,
            hidden_size=hidden_size,
            kv_hidden_size=None,
            prefix="transformer.layers.",
        )
        param_not_load, _ = ms.load_param_into_net(network, pynative_params)
        assert (
            not param_not_load
        ), f"{param_not_load} was not loaded in this net, test failed."
    else:
        param_dict = generate_ckpt(
            hidden_size=hidden_size, module_type="transformer", num_layers=num_layers
        )
        pynative_params = transform_transformerlayer_params(
            param_dict,
            hidden_size=hidden_size,
            kv_hidden_size=None,
            prefix="transformer.layers.",
        )
        ms.load_param_into_net(network, pynative_params)

    input_ids = Tensor(shape=(None, None, None), dtype=mstype.float32)
    labels = Tensor(shape=(None, None), dtype=mstype.float32)
    attn_mask = Tensor(shape=(None, None), dtype=mstype.float32)
    network.set_inputs(input_ids, labels, attn_mask)

    optimizer = AdamW(
        network=network,
        params=network.get_parameters(),
        zero_level="z1",
        opt_parallel_group=get_data_parallel_group(with_context_parallel=True),
        cpu_offload=False,
        with_context_parallel=True
    )

    losses = train(
        1, dataset, network, optimizer, None, with_attn_input=True, zero_level=1
    )
    print("losses:", losses)
    golden_losses = np.load("golden_losses.npy")
    golden_losses = golden_losses[:, get_data_parallel_rank(): golden_losses.shape[1]: dp]
    print("golden_losses:", golden_losses)
    losses = np.array(losses)

    tols = dict(atol=1.0e-3, rtol=1.0e-3)
    allclose_nparray(golden_losses, losses, **tols)
    print("test passed!")


def run_parallel_transformer_with_cp_and_zero2(sp=8, enable_flash_sp=False):
    """Test ParallelTransformer."""
    golden_batch_size = 8
    total_cards = 8
    dp = total_cards // sp
    batch_size = golden_batch_size // dp

    dataset_size = 128
    num_layers = 1
    seq_length = 4096
    num_attention_heads = 4
    hidden_size = 128

    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic="ON")

    init()
    initialize_model_parallel(context_parallel_size=sp)

    ms.set_seed(2024)

    input_data = np.random.random((dataset_size, seq_length, hidden_size)).astype(
        np.float32
    )
    label_data = np.zeros((dataset_size, seq_length)).astype(np.float32)
    dataset = TestData(
        input_data=input_data, label_data=label_data, with_attn_mask=True
    )
    dataset = ds.GeneratorDataset(
        dataset,
        column_names=["input_ids", "labels", "attention_mask"],
        num_shards=get_data_parallel_world_size(),
        shard_id=get_data_parallel_rank(),
        shuffle=False,
    )
    dataset = dataset.batch(batch_size)

    if not enable_flash_sp:
        dataset = dataset.map(
            operations=get_batch_on_this_cp_rank_with_ringattention,
            input_columns=["input_ids", "labels", "attention_mask"],
            output_columns=["input_ids", "labels", "attention_mask"],
        )
    else:
        dataset = dataset.map(
            operations=get_batch_on_this_cp_rank_with_flashsp,
            input_columns=["input_ids", "labels", "attention_mask"],
            output_columns=["input_ids", "labels", "attention_mask"],
        )

    parallel_config = ModelParallelConfig()
    config = TransformerConfig(
        seq_length=seq_length,
        vocab_size=1,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        hidden_size=hidden_size,
        attention_type="self_attn",
        qkv_has_bias=True,
        out_proj_has_bias=False,
        parallel_config=parallel_config,
        params_dtype="float32",
        compute_dtype="float32",
        softmax_compute_dtype="float32",
        hidden_dropout=0.0,
        attention_dropout=0.0,
        mask_func_type="attn_mask_add",
        mlp_has_bias=True,
        ffn_hidden_size=4 * hidden_size,
        hidden_act="gelu",
        apply_residual_connection_post_norm=False,
        normalization="FusedRMSNorm",
        norm_epsilon=1.0e-5,
        use_flash_attention=True,
        fa_config={
            "pre_tokens": 65536,
            "next_tokens": 0,
            "sparse_mode": 0,
        },
        enable_flash_sp=enable_flash_sp,
    )

    network = ParallelTransformerNet(config=config, with_rope=True)
    save_golden = True
    if save_golden:
        golden_ckpt_path = f"transformer_golden.ckpt"
        assert os.path.exists(golden_ckpt_path), (
            "'golden.ckpt' did not exits, please run generate_golden() to generate one by running below command: \n"
            + "`pytest -sv test_transformer.py::TestParallelTransformer::generate_golden`"
        )
        golden_params = ms.load_checkpoint(golden_ckpt_path)
        pynative_params = transform_transformerlayer_params(
            golden_params,
            hidden_size=hidden_size,
            kv_hidden_size=None,
            prefix="transformer.layers.",
        )
        param_not_load, _ = ms.load_param_into_net(network, pynative_params)
        assert (
            not param_not_load
        ), f"{param_not_load} was not loaded in this net, test failed."
    else:
        param_dict = generate_ckpt(
            hidden_size=hidden_size, module_type="transformer", num_layers=num_layers
        )
        pynative_params = transform_transformerlayer_params(
            param_dict,
            hidden_size=hidden_size,
            kv_hidden_size=None,
            prefix="transformer.layers.",
        )
        ms.load_param_into_net(network, pynative_params)

    input_ids = Tensor(shape=(None, None, None), dtype=mstype.float32)
    labels = Tensor(shape=(None, None), dtype=mstype.float32)
    attn_mask = Tensor(shape=(None, None), dtype=mstype.float32)
    network.set_inputs(input_ids, labels, attn_mask)

    optimizer = AdamW(
        network=network,
        params=network.get_parameters(),
        zero_level="z2",
        opt_parallel_group=get_data_parallel_group(with_context_parallel=True),
        cpu_offload=False,
        with_context_parallel=True
    )

    losses = train(
        1, dataset, network, optimizer, None, with_attn_input=True, zero_level=2
    )
    print("losses:", losses)
    golden_losses = np.load("golden_losses.npy")
    golden_losses = golden_losses[:, get_data_parallel_rank(): golden_losses.shape[1]: dp]
    print("golden_losses:", golden_losses)
    losses = np.array(losses)

    tols = dict(atol=3.0e-3, rtol=3.0e-3)
    allclose_nparray(golden_losses, losses, **tols)
    print("test passed!")


def run_parallel_transformer_with_cp_and_zero3(sp=8, enable_flash_sp=False):
    """Test ParallelTransformer."""
    golden_batch_size = 8
    total_cards = 8
    dp = total_cards // sp
    batch_size = golden_batch_size // dp

    dataset_size = 128
    num_layers = 1
    seq_length = 4096
    num_attention_heads = 4
    hidden_size = 128

    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic="ON")

    init()
    initialize_model_parallel(context_parallel_size=sp)

    ms.set_seed(2024)

    input_data = np.random.random((dataset_size, seq_length, hidden_size)).astype(
        np.float32
    )
    label_data = np.zeros((dataset_size, seq_length)).astype(np.float32)
    dataset = TestData(
        input_data=input_data, label_data=label_data, with_attn_mask=True
    )
    dataset = ds.GeneratorDataset(
        dataset,
        column_names=["input_ids", "labels", "attention_mask"],
        num_shards=get_data_parallel_world_size(),
        shard_id=get_data_parallel_rank(),
        shuffle=False,
    )
    dataset = dataset.batch(batch_size)

    if not enable_flash_sp:
        dataset = dataset.map(
            operations=get_batch_on_this_cp_rank_with_ringattention,
            input_columns=["input_ids", "labels", "attention_mask"],
            output_columns=["input_ids", "labels", "attention_mask"],
        )
    else:
        dataset = dataset.map(
            operations=get_batch_on_this_cp_rank_with_flashsp,
            input_columns=["input_ids", "labels", "attention_mask"],
            output_columns=["input_ids", "labels", "attention_mask"],
        )

    parallel_config = ModelParallelConfig()
    config = TransformerConfig(
        seq_length=seq_length,
        vocab_size=1,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        hidden_size=hidden_size,
        attention_type="self_attn",
        qkv_has_bias=True,
        out_proj_has_bias=False,
        parallel_config=parallel_config,
        params_dtype="float32",
        compute_dtype="float32",
        softmax_compute_dtype="float32",
        hidden_dropout=0.0,
        attention_dropout=0.0,
        mask_func_type="attn_mask_add",
        mlp_has_bias=True,
        ffn_hidden_size=4 * hidden_size,
        hidden_act="gelu",
        apply_residual_connection_post_norm=False,
        normalization="FusedRMSNorm",
        norm_epsilon=1.0e-5,
        use_flash_attention=True,
        fa_config={
            "pre_tokens": 65536,
            "next_tokens": 0,
            "sparse_mode": 0,
        },
        enable_flash_sp=enable_flash_sp,
    )

    network = ParallelTransformerNet(config=config, with_rope=True)
    save_golden = True
    if save_golden:
        golden_ckpt_path = f"transformer_golden.ckpt"
        assert os.path.exists(golden_ckpt_path), (
            "'golden.ckpt' did not exits, please run generate_golden() to generate one by running below command: \n"
            + "`pytest -sv test_transformer.py::TestParallelTransformer::generate_golden`"
        )
        golden_params = ms.load_checkpoint(golden_ckpt_path)
        pynative_params = transform_transformerlayer_params(
            golden_params,
            hidden_size=hidden_size,
            kv_hidden_size=None,
            prefix="transformer.layers.",
        )
        param_not_load, _ = ms.load_param_into_net(network, pynative_params)
        assert (
            not param_not_load
        ), f"{param_not_load} was not loaded in this net, test failed."
    else:
        param_dict = generate_ckpt(
            hidden_size=hidden_size, module_type="transformer", num_layers=num_layers
        )
        pynative_params = transform_transformerlayer_params(
            param_dict,
            hidden_size=hidden_size,
            kv_hidden_size=None,
            prefix="transformer.layers.",
        )
        ms.load_param_into_net(network, pynative_params)

    input_ids = Tensor(shape=(None, None, None), dtype=mstype.float32)
    labels = Tensor(shape=(None, None), dtype=mstype.float32)
    attn_mask = Tensor(shape=(None, None), dtype=mstype.float32)
    network.set_inputs(input_ids, labels, attn_mask)

    optimizer = AdamW(
        network=network,
        params=network.get_parameters(),
        zero_level="z3",
        opt_parallel_group=get_data_parallel_group(with_context_parallel=True),
        cpu_offload=False,
        with_context_parallel=True
    )

    losses = train(
        1, dataset, network, optimizer, None, with_attn_input=True, zero_level=3
    )
    print("losses:", losses)
    golden_losses = np.load("golden_losses.npy")
    golden_losses = golden_losses[:, get_data_parallel_rank(): golden_losses.shape[1]: dp]
    print("golden_losses:", golden_losses)
    losses = np.array(losses)

    tols = dict(atol=3.0e-3, rtol=3.0e-3)
    allclose_nparray(golden_losses, losses, **tols)
    print("test passed!")


def run_parallel_transformer_with_dp(dp=8):
    """Test ParallelTransformer."""
    golden_batch_size = 8
    batch_size = golden_batch_size // dp
    dataset_size = 128
    num_layers = 1
    seq_length = 4096
    num_attention_heads = 4
    hidden_size = 128

    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic="ON")

    init()
    initialize_model_parallel()

    ms.set_seed(2024)
    input_data = np.random.random((dataset_size, seq_length, hidden_size)).astype(
        np.float32
    )
    label_data = np.zeros((dataset_size, seq_length)).astype(np.float32)
    dataset = TestData(
        input_data=input_data, label_data=label_data, with_attn_mask=True
    )
    dataset = ds.GeneratorDataset(
        dataset,
        column_names=["input_ids", "labels", "attention_mask"],
        num_shards=get_data_parallel_world_size(),
        shard_id=get_data_parallel_rank(),
        shuffle=False,
    )

    dataset = dataset.batch(batch_size)

    parallel_config = ModelParallelConfig()
    config = TransformerConfig(
        seq_length=seq_length,
        vocab_size=1,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        hidden_size=hidden_size,
        attention_type="self_attn",
        qkv_has_bias=True,
        out_proj_has_bias=False,
        parallel_config=parallel_config,
        params_dtype="float32",
        compute_dtype="float32",
        softmax_compute_dtype="float32",
        hidden_dropout=0.0,
        attention_dropout=0.0,
        mask_func_type="attn_mask_add",
        mlp_has_bias=True,
        ffn_hidden_size=4 * hidden_size,
        hidden_act="gelu",
        apply_residual_connection_post_norm=False,
        normalization="FusedRMSNorm",
        norm_epsilon=1.0e-5,
        use_flash_attention=True,
        fa_config={
            "pre_tokens": 65536,
            "next_tokens": 0,
            "sparse_mode": 0,
        },
    )

    network = ParallelTransformerNet(config=config, with_rope=True)
    save_golden = True
    if save_golden:
        golden_ckpt_path = f"transformer_golden.ckpt"
        assert os.path.exists(golden_ckpt_path), (
            "'golden.ckpt' did not exits, please run generate_golden() to generate one by running below command: \n"
            + "`pytest -sv test_transformer.py::TestParallelTransformer::generate_golden`"
        )
        golden_params = ms.load_checkpoint(golden_ckpt_path)
        pynative_params = transform_transformerlayer_params(
            golden_params,
            hidden_size=hidden_size,
            kv_hidden_size=None,
            prefix="transformer.layers.",
        )
        param_not_load, _ = ms.load_param_into_net(network, pynative_params)
        assert (
            not param_not_load
        ), f"{param_not_load} was not loaded in this net, test failed."
    else:
        param_dict = generate_ckpt(
            hidden_size=hidden_size, module_type="transformer", num_layers=num_layers
        )
        pynative_params = transform_transformerlayer_params(
            param_dict,
            hidden_size=hidden_size,
            kv_hidden_size=None,
            prefix="transformer.layers.",
        )
        ms.load_param_into_net(network, pynative_params)

    input_ids = Tensor(shape=(None, None, None), dtype=mstype.float32)
    labels = Tensor(shape=(None, None), dtype=mstype.float32)
    attn_mask = Tensor(shape=(None, None), dtype=mstype.float32)
    network.set_inputs(input_ids, labels, attn_mask)

    optimizer = optim.AdamW(params=network.get_parameters())

    losses = train(1, dataset, network, optimizer, None, with_attn_input=True)
    print("losses:", losses)
    golden_losses = np.load("golden_losses.npy")
    golden_losses = golden_losses[
        :,
        batch_size
        * get_data_parallel_rank(): min(batch_size * (get_data_parallel_rank() + 1), golden_losses.shape[1]),
    ]
    print("golden_losses:", golden_losses)
    losses = np.array(losses)
    tols = dict(atol=2.0e-3, rtol=2.0e-3)
    allclose_nparray(golden_losses, losses, **tols)
    print("test passed!")


def run_parallel_transformer_with_dp_and_zero1(dp=8):
    """Test ParallelTransformer."""
    golden_batch_size = 8
    batch_size = golden_batch_size // dp
    dataset_size = 128
    num_layers = 1
    seq_length = 4096
    num_attention_heads = 4
    hidden_size = 128

    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic="ON")

    init()
    initialize_model_parallel()

    ms.set_seed(2024)
    input_data = np.random.random((dataset_size, seq_length, hidden_size)).astype(
        np.float32
    )
    label_data = np.zeros((dataset_size, seq_length)).astype(np.float32)
    dataset = TestData(
        input_data=input_data, label_data=label_data, with_attn_mask=True
    )
    dataset = ds.GeneratorDataset(
        dataset,
        column_names=["input_ids", "labels", "attention_mask"],
        num_shards=get_data_parallel_world_size(),
        shard_id=get_data_parallel_rank(),
        shuffle=False,
    )

    dataset = dataset.batch(batch_size)

    parallel_config = ModelParallelConfig()

    config = TransformerConfig(
        seq_length=seq_length,
        vocab_size=1,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        hidden_size=hidden_size,
        attention_type="self_attn",
        qkv_has_bias=True,
        out_proj_has_bias=False,
        parallel_config=parallel_config,
        params_dtype="float32",
        compute_dtype="float32",
        softmax_compute_dtype="float32",
        hidden_dropout=0.0,
        attention_dropout=0.0,
        mask_func_type="attn_mask_add",
        mlp_has_bias=True,
        ffn_hidden_size=4 * hidden_size,
        hidden_act="gelu",
        apply_residual_connection_post_norm=False,
        normalization="FusedRMSNorm",
        norm_epsilon=1.0e-5,
        use_flash_attention=True,
        fa_config={
            "pre_tokens": 65536,
            "next_tokens": 0,
            "sparse_mode": 0,
        },
    )

    network = ParallelTransformerNet(config=config, with_rope=True)
    save_golden = True
    if save_golden:
        golden_ckpt_path = f"transformer_golden.ckpt"
        assert os.path.exists(golden_ckpt_path), (
            "'golden.ckpt' did not exits, please run generate_golden() to generate one by running below command: \n"
            + "`pytest -sv test_transformer.py::TestParallelTransformer::generate_golden`"
        )
        golden_params = ms.load_checkpoint(golden_ckpt_path)
        pynative_params = transform_transformerlayer_params(
            golden_params,
            hidden_size=hidden_size,
            kv_hidden_size=None,
            prefix="transformer.layers.",
        )
        param_not_load, _ = ms.load_param_into_net(network, pynative_params)
        assert (
            not param_not_load
        ), f"{param_not_load} was not loaded in this net, test failed."

    else:
        param_dict = generate_ckpt(
            hidden_size=hidden_size, module_type="transformer", num_layers=num_layers
        )
        pynative_params = transform_transformerlayer_params(
            param_dict,
            hidden_size=hidden_size,
            kv_hidden_size=None,
            prefix="transformer.layers.",
        )
        ms.load_param_into_net(network, pynative_params)

    input_ids = Tensor(shape=(None, None, None), dtype=mstype.float32)
    labels = Tensor(shape=(None, None), dtype=mstype.float32)
    attn_mask = Tensor(shape=(None, None), dtype=mstype.float32)
    network.set_inputs(input_ids, labels, attn_mask)

    optimizer = AdamW(
        network=network,
        params=network.get_parameters(),
        zero_level="z1",
        grad_allreduce_op="mean",
        opt_parallel_group=get_data_parallel_group(with_context_parallel=True),
        cpu_offload=False,
    )

    losses = train(
        1, dataset, network, optimizer, None, with_attn_input=True, zero_level=1
    )

    print("losses:", losses)
    golden_losses = np.load("golden_losses.npy")
    golden_losses = golden_losses[
        :,
        batch_size
        * get_data_parallel_rank(): min(batch_size * (get_data_parallel_rank() + 1), golden_losses.shape[1]),
    ]
    print("golden_losses:", golden_losses)
    losses = np.array(losses)

    tols = dict(atol=3.0e-3, rtol=3.0e-3)
    allclose_nparray(golden_losses, losses, **tols)

    print("test passed!")


def run_parallel_transformer_with_dp_and_zero2(dp=8):
    """Test ParallelTransformer."""
    golden_batch_size = 8
    batch_size = golden_batch_size // dp
    dataset_size = 128
    num_layers = 1
    seq_length = 4096
    num_attention_heads = 4
    hidden_size = 128

    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic="ON")

    init()
    initialize_model_parallel()

    ms.set_seed(2024)
    input_data = np.random.random((dataset_size, seq_length, hidden_size)).astype(
        np.float32
    )
    label_data = np.zeros((dataset_size, seq_length)).astype(np.float32)
    dataset = TestData(
        input_data=input_data, label_data=label_data, with_attn_mask=True
    )
    dataset = ds.GeneratorDataset(
        dataset,
        column_names=["input_ids", "labels", "attention_mask"],
        num_shards=get_data_parallel_world_size(),
        shard_id=get_data_parallel_rank(),
        shuffle=False,
    )

    dataset = dataset.batch(batch_size)

    parallel_config = ModelParallelConfig()

    config = TransformerConfig(
        seq_length=seq_length,
        vocab_size=1,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        hidden_size=hidden_size,
        attention_type="self_attn",
        qkv_has_bias=True,
        out_proj_has_bias=False,
        parallel_config=parallel_config,
        params_dtype="float32",
        compute_dtype="float32",
        softmax_compute_dtype="float32",
        hidden_dropout=0.0,
        attention_dropout=0.0,
        mask_func_type="attn_mask_add",
        mlp_has_bias=True,
        ffn_hidden_size=4 * hidden_size,
        hidden_act="gelu",
        apply_residual_connection_post_norm=False,
        normalization="FusedRMSNorm",
        norm_epsilon=1.0e-5,
        use_flash_attention=True,
        fa_config={
            "pre_tokens": 65536,
            "next_tokens": 0,
            "sparse_mode": 0,
        },
    )

    network = ParallelTransformerNet(config=config, with_rope=True)
    save_golden = True
    if save_golden:
        golden_ckpt_path = f"transformer_golden.ckpt"
        assert os.path.exists(golden_ckpt_path), (
            "'golden.ckpt' did not exits, please run generate_golden() to generate one by running below command: \n"
            + "`pytest -sv test_transformer.py::TestParallelTransformer::generate_golden`"
        )
        golden_params = ms.load_checkpoint(golden_ckpt_path)
        pynative_params = transform_transformerlayer_params(
            golden_params,
            hidden_size=hidden_size,
            kv_hidden_size=None,
            prefix="transformer.layers.",
        )
        param_not_load, _ = ms.load_param_into_net(network, pynative_params)
        assert (
            not param_not_load
        ), f"{param_not_load} was not loaded in this net, test failed."

    else:
        param_dict = generate_ckpt(
            hidden_size=hidden_size, module_type="transformer", num_layers=num_layers
        )
        pynative_params = transform_transformerlayer_params(
            param_dict,
            hidden_size=hidden_size,
            kv_hidden_size=None,
            prefix="transformer.layers.",
        )
        ms.load_param_into_net(network, pynative_params)

    input_ids = Tensor(shape=(None, None, None), dtype=mstype.float32)
    labels = Tensor(shape=(None, None), dtype=mstype.float32)
    attn_mask = Tensor(shape=(None, None), dtype=mstype.float32)
    network.set_inputs(input_ids, labels, attn_mask)

    optimizer = AdamW(
        network=network,
        params=network.get_parameters(),
        zero_level="z2",
        grad_allreduce_op="mean",
        opt_parallel_group=get_data_parallel_group(with_context_parallel=True),
        cpu_offload=False,
    )

    losses = train(
        1, dataset, network, optimizer, None, with_attn_input=True, zero_level=2
    )

    print("losses:", losses)
    golden_losses = np.load("golden_losses.npy")
    golden_losses = golden_losses[
        :,
        batch_size
        * get_data_parallel_rank(): min(batch_size * (get_data_parallel_rank() + 1), golden_losses.shape[1]),
    ]
    print("golden_losses:", golden_losses)
    losses = np.array(losses)

    tols = dict(atol=3.0e-3, rtol=3.0e-3)
    allclose_nparray(golden_losses, losses, **tols)

    print("test passed!")


def run_parallel_transformer_with_dp_and_zero3(dp=8):
    """Test ParallelTransformer."""
    golden_batch_size = 8
    batch_size = golden_batch_size // dp
    dataset_size = 128
    num_layers = 1
    seq_length = 4096
    num_attention_heads = 4
    hidden_size = 128

    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic="ON")

    init()
    initialize_model_parallel()

    ms.set_seed(2024)
    input_data = np.random.random((dataset_size, seq_length, hidden_size)).astype(
        np.float32
    )
    label_data = np.zeros((dataset_size, seq_length)).astype(np.float32)
    dataset = TestData(
        input_data=input_data, label_data=label_data, with_attn_mask=True
    )
    dataset = ds.GeneratorDataset(
        dataset,
        column_names=["input_ids", "labels", "attention_mask"],
        num_shards=get_data_parallel_world_size(),
        shard_id=get_data_parallel_rank(),
        shuffle=False,
    )

    dataset = dataset.batch(batch_size)

    parallel_config = ModelParallelConfig()

    config = TransformerConfig(
        seq_length=seq_length,
        vocab_size=1,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        hidden_size=hidden_size,
        attention_type="self_attn",
        qkv_has_bias=True,
        out_proj_has_bias=False,
        parallel_config=parallel_config,
        params_dtype="float32",
        compute_dtype="float32",
        softmax_compute_dtype="float32",
        hidden_dropout=0.0,
        attention_dropout=0.0,
        mask_func_type="attn_mask_add",
        mlp_has_bias=True,
        ffn_hidden_size=4 * hidden_size,
        hidden_act="gelu",
        apply_residual_connection_post_norm=False,
        normalization="FusedRMSNorm",
        norm_epsilon=1.0e-5,
        use_flash_attention=True,
        fa_config={
            "pre_tokens": 65536,
            "next_tokens": 0,
            "sparse_mode": 0,
        },
    )

    network = ParallelTransformerNet(config=config, with_rope=True)
    save_golden = True
    if save_golden:
        golden_ckpt_path = f"transformer_golden.ckpt"
        assert os.path.exists(golden_ckpt_path), (
            "'golden.ckpt' did not exits, please run generate_golden() to generate one by running below command: \n"
            + "`pytest -sv test_transformer.py::TestParallelTransformer::generate_golden`"
        )
        golden_params = ms.load_checkpoint(golden_ckpt_path)
        pynative_params = transform_transformerlayer_params(
            golden_params,
            hidden_size=hidden_size,
            kv_hidden_size=None,
            prefix="transformer.layers.",
        )
        param_not_load, _ = ms.load_param_into_net(network, pynative_params)
        assert (
            not param_not_load
        ), f"{param_not_load} was not loaded in this net, test failed."

    else:
        param_dict = generate_ckpt(
            hidden_size=hidden_size, module_type="transformer", num_layers=num_layers
        )
        pynative_params = transform_transformerlayer_params(
            param_dict,
            hidden_size=hidden_size,
            kv_hidden_size=None,
            prefix="transformer.layers.",
        )
        ms.load_param_into_net(network, pynative_params)

    input_ids = Tensor(shape=(None, None, None), dtype=mstype.float32)
    labels = Tensor(shape=(None, None), dtype=mstype.float32)
    attn_mask = Tensor(shape=(None, None), dtype=mstype.float32)
    network.set_inputs(input_ids, labels, attn_mask)

    optimizer = AdamW(
        network=network,
        params=network.get_parameters(),
        zero_level="z3",
        grad_allreduce_op="mean",
        opt_parallel_group=get_data_parallel_group(with_context_parallel=True),
        cpu_offload=False,
    )

    losses = train(
        1, dataset, network, optimizer, None, with_attn_input=True, zero_level=3
    )

    print("losses:", losses)
    golden_losses = np.load("golden_losses.npy")
    golden_losses = golden_losses[
        :,
        batch_size
        * get_data_parallel_rank(): min(batch_size * (get_data_parallel_rank() + 1), golden_losses.shape[1]),
    ]
    print("golden_losses:", golden_losses)
    losses = np.array(losses)

    tols = dict(atol=3.0e-3, rtol=3.0e-3)
    allclose_nparray(golden_losses, losses, **tols)

    print("test passed!")


def run_parallel_transformer_with_cp_and_tp(sp=4, tp=2, enable_flash_sp=False):
    """Test ParallelTransformer."""

    batch_size = 8
    dataset_size = 128
    num_layers = 1
    seq_length = 4096
    num_attention_heads = 4
    hidden_size = 128
    sequence_parallel = sp
    tensor_parallel = tp

    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic="ON")

    init()
    initialize_model_parallel(
        context_parallel_size=sequence_parallel,
        tensor_model_parallel_size=tensor_parallel,
    )

    ms.set_seed(2024)
    input_data = np.random.random((dataset_size, seq_length, hidden_size)).astype(
        np.float32
    )
    label_data = np.zeros((dataset_size, seq_length)).astype(np.float32)
    dataset = TestData(
        input_data=input_data, label_data=label_data, with_attn_mask=True
    )
    dataset = ds.GeneratorDataset(
        dataset,
        column_names=["input_ids", "labels", "attention_mask"],
        num_shards=get_data_parallel_world_size(),
        shard_id=get_data_parallel_rank(),
        shuffle=False,
    )
    dataset = dataset.batch(batch_size)

    if not enable_flash_sp:
        dataset = dataset.map(
            operations=get_batch_on_this_cp_rank_with_ringattention,
            input_columns=["input_ids", "labels", "attention_mask"],
            output_columns=["input_ids", "labels", "attention_mask"],
        )
    else:
        dataset = dataset.map(
            operations=get_batch_on_this_cp_rank_with_flashsp,
            input_columns=["input_ids", "labels", "attention_mask"],
            output_columns=["input_ids", "labels", "attention_mask"],
        )

    parallel_config = ModelParallelConfig(tensor_model_parallel_size=tensor_parallel)
    config = TransformerConfig(
        seq_length=seq_length,
        vocab_size=1,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        hidden_size=hidden_size,
        attention_type="self_attn",
        qkv_has_bias=True,
        out_proj_has_bias=False,
        parallel_config=parallel_config,
        params_dtype="float32",
        compute_dtype="float32",
        softmax_compute_dtype="float32",
        hidden_dropout=0.0,
        attention_dropout=0.0,
        mask_func_type="attn_mask_add",
        mlp_has_bias=True,
        ffn_hidden_size=4 * hidden_size,
        hidden_act="gelu",
        apply_residual_connection_post_norm=False,
        normalization="FusedRMSNorm",
        norm_epsilon=1.0e-5,
        use_flash_attention=True,
        fa_config={
            "pre_tokens": 65536,
            "next_tokens": 0,
            "sparse_mode": 0,
        },
        enable_flash_sp=enable_flash_sp,
    )

    network = ParallelTransformerNet(config=config, with_rope=True)
    save_golden = True
    if save_golden:
        golden_ckpt_path = f"transformer_golden.ckpt"
        assert os.path.exists(golden_ckpt_path), (
            "'golden.ckpt' did not exits, please run generate_golden() to generate one by running below command: \n"
            + "`pytest -sv test_transformer.py::TestParallelTransformer::generate_golden`"
        )
        golden_params = ms.load_checkpoint(golden_ckpt_path)
        pynative_params = transform_transformerlayer_params(
            golden_params,
            hidden_size=hidden_size,
            kv_hidden_size=None,
            prefix="transformer.layers.",
        )
        param_not_load, _ = ms.load_param_into_net(network, pynative_params)
        assert (
            not param_not_load
        ), f"{param_not_load} was not loaded in this net, test failed."
    else:
        param_dict = generate_ckpt(
            hidden_size=hidden_size, module_type="transformer", num_layers=num_layers
        )
        pynative_params = transform_transformerlayer_params(
            param_dict,
            hidden_size=hidden_size,
            kv_hidden_size=None,
            prefix="transformer.layers.",
        )
        ms.load_param_into_net(network, pynative_params)

    input_ids = Tensor(shape=(None, None, None), dtype=mstype.float32)
    labels = Tensor(shape=(None, None), dtype=mstype.float32)
    attn_mask = Tensor(shape=(None, None), dtype=mstype.float32)
    network.set_inputs(input_ids, labels, attn_mask)

    optimizer = optim.AdamW(params=network.get_parameters())

    losses = train(1, dataset, network, optimizer, None, with_attn_input=True)
    print("losses:", losses)
    golden_losses = np.load("golden_losses.npy")
    print("golden_losses:", golden_losses)

    tols = dict(atol=3.0e-3, rtol=3.0e-3)
    allclose_nparray(golden_losses, losses, **tols)

    print("test passed!")


def run_parallel_transformer_with_cp_and_tp_and_dp(sp=2, tp=2, dp=2, enable_flash_sp=False
                                                   ):
    """Test ParallelTransformer."""

    golden_batch_size = 8
    batch_size = golden_batch_size // dp
    dataset_size = 128
    num_layers = 1
    seq_length = 4096
    num_attention_heads = 4
    hidden_size = 128
    sequence_parallel = sp
    tensor_parallel = tp

    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic="ON")

    init()
    initialize_model_parallel(
        context_parallel_size=sequence_parallel,
        tensor_model_parallel_size=tensor_parallel,
    )

    ms.set_seed(2024)
    input_data = np.random.random((dataset_size, seq_length, hidden_size)).astype(
        np.float32
    )
    label_data = np.zeros((dataset_size, seq_length)).astype(np.float32)
    dataset = TestData(
        input_data=input_data, label_data=label_data, with_attn_mask=True
    )
    dataset = ds.GeneratorDataset(
        dataset,
        column_names=["input_ids", "labels", "attention_mask"],
        num_shards=get_data_parallel_world_size(),
        shard_id=get_data_parallel_rank(),
        shuffle=False,
    )
    dataset = dataset.batch(batch_size)

    if not enable_flash_sp:
        dataset = dataset.map(
            operations=get_batch_on_this_cp_rank_with_ringattention,
            input_columns=["input_ids", "labels", "attention_mask"],
            output_columns=["input_ids", "labels", "attention_mask"],
        )
    else:
        dataset = dataset.map(
            operations=get_batch_on_this_cp_rank_with_flashsp,
            input_columns=["input_ids", "labels", "attention_mask"],
            output_columns=["input_ids", "labels", "attention_mask"],
        )

    parallel_config = ModelParallelConfig(tensor_model_parallel_size=tensor_parallel)
    config = TransformerConfig(
        seq_length=seq_length,
        vocab_size=1,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        hidden_size=hidden_size,
        attention_type="self_attn",
        qkv_has_bias=True,
        out_proj_has_bias=False,
        parallel_config=parallel_config,
        params_dtype="float32",
        compute_dtype="float32",
        softmax_compute_dtype="float32",
        hidden_dropout=0.0,
        attention_dropout=0.0,
        mask_func_type="attn_mask_add",
        mlp_has_bias=True,
        ffn_hidden_size=4 * hidden_size,
        hidden_act="gelu",
        apply_residual_connection_post_norm=False,
        normalization="FusedRMSNorm",
        norm_epsilon=1.0e-5,
        use_flash_attention=True,
        fa_config={
            "pre_tokens": 65536,
            "next_tokens": 0,
            "sparse_mode": 0,
        },
        enable_flash_sp=enable_flash_sp,
    )

    network = ParallelTransformerNet(config=config, with_rope=True)
    save_golden = True
    if save_golden:
        golden_ckpt_path = f"transformer_golden.ckpt"
        assert os.path.exists(golden_ckpt_path), (
            "'golden.ckpt' did not exits, please run generate_golden() to generate one by running below command: \n"
            + "`pytest -sv test_transformer.py::TestParallelTransformer::generate_golden`"
        )
        golden_params = ms.load_checkpoint(golden_ckpt_path)
        pynative_params = transform_transformerlayer_params(
            golden_params,
            hidden_size=hidden_size,
            kv_hidden_size=None,
            prefix="transformer.layers.",
        )
        param_not_load, _ = ms.load_param_into_net(network, pynative_params)
        assert (
            not param_not_load
        ), f"{param_not_load} was not loaded in this net, test failed."
    else:
        param_dict = generate_ckpt(
            hidden_size=hidden_size, module_type="transformer", num_layers=num_layers
        )
        pynative_params = transform_transformerlayer_params(
            param_dict,
            hidden_size=hidden_size,
            kv_hidden_size=None,
            prefix="transformer.layers.",
        )
        ms.load_param_into_net(network, pynative_params)

    input_ids = Tensor(shape=(None, None, None), dtype=mstype.float32)
    labels = Tensor(shape=(None, None), dtype=mstype.float32)
    attn_mask = Tensor(shape=(None, None), dtype=mstype.float32)
    network.set_inputs(input_ids, labels, attn_mask)

    optimizer = optim.AdamW(params=network.get_parameters())

    losses = train(1, dataset, network, optimizer, None, with_attn_input=True)
    print("losses:", losses)

    golden_losses = np.load("golden_losses.npy")
    golden_losses = golden_losses[:, get_data_parallel_rank(): golden_losses.shape[1]: dp]
    print("golden_losses:", golden_losses)
    losses = np.array(losses)
    tols = dict(atol=3.0e-3, rtol=3.0e-3)
    allclose_nparray(golden_losses, losses, **tols)

    print("test passed!")


def run_parallel_transformer_with_cp_and_tp_and_zero1(sp=2, tp=2, dp=2, enable_flash_sp=False
                                                      ):
    """Test ParallelTransformer."""

    golden_batch_size = 8
    batch_size = golden_batch_size // dp
    dataset_size = 128
    num_layers = 1
    seq_length = 4096
    num_attention_heads = 4
    hidden_size = 128
    sequence_parallel = sp
    tensor_parallel = tp

    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic="ON")

    init()
    initialize_model_parallel(
        context_parallel_size=sequence_parallel,
        tensor_model_parallel_size=tensor_parallel,
    )

    ms.set_seed(2024)
    input_data = np.random.random((dataset_size, seq_length, hidden_size)).astype(
        np.float32
    )
    label_data = np.zeros((dataset_size, seq_length)).astype(np.float32)
    dataset = TestData(
        input_data=input_data, label_data=label_data, with_attn_mask=True
    )
    dataset = ds.GeneratorDataset(
        dataset,
        column_names=["input_ids", "labels", "attention_mask"],
        num_shards=get_data_parallel_world_size(),
        shard_id=get_data_parallel_rank(),
        shuffle=False,
    )

    dataset = dataset.batch(batch_size)

    if not enable_flash_sp:
        dataset = dataset.map(
            operations=get_batch_on_this_cp_rank_with_ringattention,
            input_columns=["input_ids", "labels", "attention_mask"],
            output_columns=["input_ids", "labels", "attention_mask"],
        )
    else:
        dataset = dataset.map(
            operations=get_batch_on_this_cp_rank_with_flashsp,
            input_columns=["input_ids", "labels", "attention_mask"],
            output_columns=["input_ids", "labels", "attention_mask"],
        )

    parallel_config = ModelParallelConfig(tensor_model_parallel_size=tensor_parallel)
    config = TransformerConfig(
        seq_length=seq_length,
        vocab_size=1,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        hidden_size=hidden_size,
        attention_type="self_attn",
        qkv_has_bias=True,
        out_proj_has_bias=False,
        parallel_config=parallel_config,
        params_dtype="float32",
        compute_dtype="float32",
        softmax_compute_dtype="float32",
        hidden_dropout=0.0,
        attention_dropout=0.0,
        mask_func_type="attn_mask_add",
        mlp_has_bias=True,
        ffn_hidden_size=4 * hidden_size,
        hidden_act="gelu",
        apply_residual_connection_post_norm=False,
        normalization="FusedRMSNorm",
        norm_epsilon=1.0e-5,
        use_flash_attention=True,
        fa_config={
            "pre_tokens": 65536,
            "next_tokens": 0,
            "sparse_mode": 0,
        },
        enable_flash_sp=enable_flash_sp,
    )

    network = ParallelTransformerNet(config=config, with_rope=True)
    save_golden = True
    if save_golden:
        golden_ckpt_path = f"transformer_golden.ckpt"
        assert os.path.exists(golden_ckpt_path), (
            "'golden.ckpt' did not exits, please run generate_golden() to generate one by running below command: \n"
            + "`pytest -sv test_transformer.py::TestParallelTransformer::generate_golden`"
        )
        golden_params = ms.load_checkpoint(golden_ckpt_path)
        pynative_params = transform_transformerlayer_params(
            golden_params,
            hidden_size=hidden_size,
            kv_hidden_size=None,
            prefix="transformer.layers.",
        )
        param_not_load, _ = ms.load_param_into_net(network, pynative_params)
        assert (
            not param_not_load
        ), f"{param_not_load} was not loaded in this net, test failed."
    else:
        param_dict = generate_ckpt(
            hidden_size=hidden_size, module_type="transformer", num_layers=num_layers
        )
        pynative_params = transform_transformerlayer_params(
            param_dict,
            hidden_size=hidden_size,
            kv_hidden_size=None,
            prefix="transformer.layers.",
        )
        ms.load_param_into_net(network, pynative_params)

    input_ids = Tensor(shape=(None, None, None), dtype=mstype.float32)
    labels = Tensor(shape=(None, None), dtype=mstype.float32)
    attn_mask = Tensor(shape=(None, None), dtype=mstype.float32)
    network.set_inputs(input_ids, labels, attn_mask)

    optimizer = AdamW(
        network=network,
        params=network.get_parameters(),
        zero_level="z1",
        grad_allreduce_op="mean",
        opt_parallel_group=get_data_parallel_group(with_context_parallel=True),
        cpu_offload=False,
        with_context_parallel=True
    )

    losses = train(
        1, dataset, network, optimizer, None, with_attn_input=True, zero_level=1
    )
    print("losses:", losses)

    golden_losses = np.load("golden_losses.npy")
    golden_losses = golden_losses[:, get_data_parallel_rank(): golden_losses.shape[1]: dp]
    print("golden_losses:", golden_losses)
    losses = np.array(losses)
    tols = dict(atol=3.0e-3, rtol=3.0e-3)
    allclose_nparray(golden_losses, losses, **tols)

    print("test passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generate_golden_with_fa",
        action="store_true",
        help="Generate golden data with fa for test.",
    )
    parser.add_argument("--use_dp", action="store_true", help="Run dp for test.")
    parser.add_argument("--use_zero1", action="store_true", help="Run zero1 for test.")
    parser.add_argument("--use_zero2", action="store_true", help="Run zero2 for test.")
    parser.add_argument("--use_zero3", action="store_true", help="Run zero3 for test.")
    parser.add_argument("--use_tp", action="store_true", help="Run tp for test.")
    parser.add_argument(
        "--use_ringattention", action="store_true", help="Run ringattention for test."
    )
    parser.add_argument(
        "--use_flashsp", action="store_true", help="Enable flashsp for test."
    )
    parser.add_argument(
        "--use_cp_and_dp",
        action="store_true",
        help="Run ringattention and dp for test.",
    )
    parser.add_argument(
        "--use_cp_and_zero1",
        action="store_true",
        help="Run ringattention and zero1 for test.",
    )
    parser.add_argument(
        "--use_cp_and_zero2",
        action="store_true",
        help="Run ringattention and zero2 for test.",
    )
    parser.add_argument(
        "--use_cp_and_zero3",
        action="store_true",
        help="Run ringattention and zero3 for test.",
    )
    parser.add_argument(
        "--use_cp_and_tp",
        action="store_true",
        help="Run ringattention and tp for test.",
    )
    parser.add_argument(
        "--use_cp_and_tp_and_dp",
        action="store_true",
        help="Run ringattention and tp and dp for test.",
    )
    parser.add_argument(
        "--use_cp_and_tp_and_zero1",
        action="store_true",
        help="Run ringattention and tp and zero1 for test.",
    )
    args, rest_args = parser.parse_known_args()
    if args.generate_golden_with_fa:
        generate_golden_with_flashattention()
    elif args.use_dp:
        run_parallel_transformer_with_dp(dp=8)
    elif args.use_zero1:
        run_parallel_transformer_with_dp_and_zero1(dp=8)
    elif args.use_zero2:
        run_parallel_transformer_with_dp_and_zero2(dp=8)
    elif args.use_zero3:
        run_parallel_transformer_with_dp_and_zero3(dp=8)
    elif args.use_tp:
        run_parallel_transformer_with_tp(tp=2)
    elif args.use_flashsp:
        run_parallel_transformer_with_cp(sp=8, enable_flash_sp=True)
    elif args.use_ringattention:
        run_parallel_transformer_with_cp(sp=2, enable_flash_sp=False)
    elif args.use_cp_and_tp:
        run_parallel_transformer_with_cp_and_tp(sp=2, tp=4)
    elif args.use_cp_and_dp:
        run_parallel_transformer_with_cp_and_dp(sp=2, enable_flash_sp=False)
    elif args.use_cp_and_zero1:
        run_parallel_transformer_with_cp_and_zero1(sp=4, enable_flash_sp=False)
    elif args.use_cp_and_zero2:
        run_parallel_transformer_with_cp_and_zero1(sp=4, enable_flash_sp=False)
    elif args.use_cp_and_zero3:
        run_parallel_transformer_with_cp_and_zero1(sp=4, enable_flash_sp=False)
    elif args.use_cp_and_tp_and_dp:
        run_parallel_transformer_with_cp_and_tp_and_dp(sp=2, tp=2, dp=2)
    elif args.use_cp_and_tp_and_zero1:
        run_parallel_transformer_with_cp_and_tp_and_zero1(sp=2, tp=2, dp=2)
    else:
        pass
    print("End test.")
