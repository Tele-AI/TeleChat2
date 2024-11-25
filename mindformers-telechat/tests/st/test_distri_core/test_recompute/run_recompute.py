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
"""Test selective recompute and gradient checkpointed recompute"""

import argparse
import numpy as np
import mindspore as ms
import mindspore.common.dtype as mstype

from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore import Tensor, nn, ops, grad
from mindspore.communication.management import init


from mindformers import MindFormerConfig
from mindformers.experimental.parallel_core.pynative.parallel_state import initialize_model_parallel
from mindformers.experimental.parallel_core.pynative.transformer import ParallelTransformerLayer
from mindformers.experimental.parallel_core.pynative.transformer import ParallelTransformer


class ParallelTransformerLayerNet(nn.Cell):
    """ParallelTransformerLayerNet."""

    def __init__(self, config):
        super(ParallelTransformerLayerNet, self).__init__()
        self.layer = ParallelTransformerLayer(layer_number=1, config=config)
        self.loss = SoftmaxCrossEntropyWithLogits()
        self.outputs = []

    def construct(self, x, attention_mask, labels):
        """Construct"""
        output = self.layer(x, attention_mask)
        self.outputs.append(output.asnumpy())
        output = ops.sum(output, dim=-1, keepdim=False)
        loss = self.loss(output, labels)
        return loss


class ParallelTransformerNet(nn.Cell):
    """ParallelTransformerNet."""

    def __init__(self, config):
        super(ParallelTransformerNet, self).__init__()
        self.transformer = ParallelTransformer(config=config, post_norm=False)
        self.loss = SoftmaxCrossEntropyWithLogits()

    def construct(self, x, attention_mask, labels):
        """Construct"""
        output = self.transformer(x, attention_mask)
        output = ops.sum(output, dim=-1, keepdim=False)
        loss = self.loss(output, labels)
        return loss


def run_selective_recompute():
    """test selective recompute"""
    seed = 42
    batch_size = 2
    seq_length = 16
    hidden_size = 32
    num_attention_heads = 8
    tensor_parallel = 1

    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic="ON")

    init()
    initialize_model_parallel(tensor_model_parallel_size=tensor_parallel)
    ms.set_seed(seed)
    ms.manual_seed(seed)

    parallel_config = MindFormerConfig(expert_model_parallel_size=1, use_sequence_parallel=False)
    lora_config = MindFormerConfig(use_lora=False)
    config = MindFormerConfig(
        num_attention_heads=num_attention_heads,
        hidden_size=hidden_size,
        attn_type="self_attn",
        qkv_has_bias=True,
        out_proj_has_bias=False,
        params_dtype=mstype.float32,
        compute_dtype=mstype.float32,
        softmax_compute_dtype=mstype.float32,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        parallel_config=parallel_config,
        lora_config=lora_config,
        mask_func_type="attn_mask_add",
        mlp_has_bias=True,
        ffn_hidden_size=4 * hidden_size,
        hidden_act="gelu",
        apply_residual_connection_post_norm=False,
        normalization="FusedRMSNorm",
        norm_epsilon=1.0e-5,
    )

    network = ParallelTransformerLayerNet(config=config)
    network.set_train()
    # set input
    inputs = Tensor(shape=(None, None, None), dtype=mstype.float32)
    attn_mask = Tensor(shape=(None, None), dtype=mstype.float32)
    labels = Tensor(shape=(None, None), dtype=mstype.float32)
    network.set_inputs(inputs, attn_mask, labels)

    grad_fn = grad(network.construct, grad_position=(0), weights=None)

    inputs = Tensor(np.random.random((batch_size, seq_length, hidden_size)).astype(np.float32), ms.float32)
    attn_mask = Tensor(np.tril(np.ones(shape=(1, 1, seq_length, seq_length))).astype(np.uint8), ms.float32)
    labels = Tensor(np.zeros((batch_size, seq_length)).astype(np.float32), ms.float32)

    # without recompute
    ms.set_seed(seed)
    ms.manual_seed(seed)
    grad_without_recompute = grad_fn(inputs, attn_mask, labels)
    # with recompute
    # pylint: disable=W0212
    network.layer._set_selective_recompute()
    ms.set_seed(seed)
    ms.manual_seed(seed)
    grad_with_recompute = grad_fn(inputs, attn_mask, labels)

    print("grad_without_recompute: ", grad_without_recompute)
    print("grad_with_recompute: ", grad_with_recompute)

    assert np.array_equal(
        grad_without_recompute.asnumpy(), grad_with_recompute.asnumpy()
    ), "Selective recompute failed."


def run_gradient_checkpointed_recompute():
    """test gradient checkpoint"""
    seed = 42
    batch_size = 2
    seq_length = 16
    num_layers = 3
    hidden_size = 32
    num_attention_heads = 8
    tensor_parallel = 1

    ms.set_context(device_target="Ascend", mode=ms.PYNATIVE_MODE, deterministic="ON")

    init()
    initialize_model_parallel(tensor_model_parallel_size=tensor_parallel)
    ms.set_seed(seed)
    ms.manual_seed(seed)

    parallel_config = MindFormerConfig(expert_model_parallel_size=1, use_sequence_parallel=False)
    lora_config = MindFormerConfig(use_lora=False)
    config = MindFormerConfig(
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        hidden_size=hidden_size,
        attn_type="self_attn",
        qkv_has_bias=True,
        out_proj_has_bias=False,
        params_dtype=mstype.float32,
        compute_dtype=mstype.float32,
        softmax_compute_dtype=mstype.float32,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        parallel_config=parallel_config,
        lora_config=lora_config,
        mask_func_type="attn_mask_add",
        mlp_has_bias=True,
        ffn_hidden_size=4 * hidden_size,
        hidden_act="gelu",
        apply_residual_connection_post_norm=False,
        normalization="FusedRMSNorm",
        norm_epsilon=1.0e-5,
    )

    network = ParallelTransformerNet(config=config)
    network.set_train()
    # set input
    inputs = Tensor(shape=(None, None, None), dtype=mstype.float32)
    attn_mask = Tensor(shape=(None, None), dtype=mstype.float32)
    labels = Tensor(shape=(None, None), dtype=mstype.float32)
    network.set_inputs(inputs, attn_mask, labels)

    grad_fn = grad(network.construct, grad_position=(0), weights=None)

    inputs = Tensor(np.random.random((batch_size, seq_length, hidden_size)).astype(np.float32), ms.float32)
    attn_mask = Tensor(np.tril(np.ones(shape=(1, 1, seq_length, seq_length))).astype(np.uint8), ms.float32)
    labels = Tensor(np.zeros((batch_size, seq_length)).astype(np.float32), ms.float32)

    # without recompute
    ms.set_seed(seed)
    ms.manual_seed(seed)
    grad_without_recompute = grad_fn(inputs, attn_mask, labels)
    # with recompute
    # pylint: disable=W0212
    network.transformer._set_checkpointed_recompute(recompute_method="uniform", recompute_num_layers=2)
    ms.set_seed(seed)
    ms.manual_seed(seed)
    grad_with_recompute = grad_fn(inputs, attn_mask, labels)

    print("grad_without_recompute: ", grad_without_recompute)
    print("grad_with_recompute: ", grad_with_recompute)

    assert np.array_equal(
        grad_without_recompute.asnumpy(), grad_with_recompute.asnumpy()
    ), "Gradient checkpointed recompute failed."


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selective", help="Run selective recompute test.", action="store_true")

    args, rest_args = parser.parse_known_args()
    if args.selective:
        run_selective_recompute()
    else:
        run_gradient_checkpointed_recompute()
