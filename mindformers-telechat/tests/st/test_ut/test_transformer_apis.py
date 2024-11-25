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
""" test transformer apis."""
import numpy as np
import pytest

import mindspore
from mindspore import Tensor
from mindspore.common import dtype
from mindspore.ops import operations as ops
from mindspore.common.api import _cell_graph_executor

from mindformers.core import CrossEntropyLoss
from mindformers.modules import (
    MultiHeadAttention, FeedForward, TransformerEncoderLayer, TransformerEncoder,
    TransformerDecoder, TransformerDecoderLayer, Transformer, AttentionMask,
    FixedSparseAttention, LowerTriangularMaskWithDynamic)


class MyActivation(mindspore.nn.Cell):
    """An example of custom activation"""

    def __init__(self):
        super(MyActivation, self).__init__()
        self.add = ops.Add()

    def construct(self, x):
        return self.add(x, 0.1)

    def activation_shard(self, parallel_config):
        self.add.shard(((parallel_config.data_parallel, parallel_config.model_parallel), ()))


class MyActivationNoShard(mindspore.nn.Cell):
    """An example of custom activation without shard"""

    def __init__(self):
        super(MyActivationNoShard, self).__init__()
        self.add = ops.Add()

    def construct(self, x):
        return self.add(x, 0.1)


def test_transformer_encoder_only():
    """
    Feature: Transformer API
    Description: Test Transformer model with encoder only
    Expectation: No exception
    """
    model = Transformer(batch_size=2,
                        src_seq_length=20,
                        tgt_seq_length=10,
                        encoder_layers=2,
                        decoder_layers=0,
                        hidden_size=64,
                        ffn_hidden_size=64)

    encoder_input_value = Tensor(np.ones((2, 20, 64)), dtype.float32)
    encoder_input_mask = Tensor(np.ones((2, 20, 20)), dtype.float16)

    _cell_graph_executor.compile(model, encoder_input_value, encoder_input_mask)


def test_transformer_encoder_log_softmax():
    """
    Feature: Transformer API
    Description: Test Transformer model with unexpected hidden act
    Expectation: No exception
    """
    with pytest.raises(ValueError):
        model = Transformer(batch_size=2,
                            src_seq_length=20,
                            tgt_seq_length=10,
                            encoder_layers=2,
                            decoder_layers=0,
                            hidden_act='logsoftmax',
                            hidden_size=64,
                            ffn_hidden_size=64)

        encoder_input_value = Tensor(np.ones((2, 20, 64)), dtype.float32)
        encoder_input_mask = Tensor(np.ones((2, 20, 20)), dtype.float16)

        _cell_graph_executor.compile(model, encoder_input_value, encoder_input_mask)


def test_transformer_encoder_leakyrelu():
    """
    Feature: Transformer API
    Description: Test Transformer model with valid hidden act
    Expectation: No exception
    """
    model = Transformer(batch_size=2,
                        src_seq_length=20,
                        tgt_seq_length=10,
                        encoder_layers=2,
                        decoder_layers=0,
                        hidden_act='leakyrelu',
                        hidden_size=64,
                        ffn_hidden_size=64)

    encoder_input_value = Tensor(np.ones((2, 20, 64)), dtype.float32)
    encoder_input_mask = Tensor(np.ones((2, 20, 20)), dtype.float16)

    _cell_graph_executor.compile(model, encoder_input_value, encoder_input_mask)


def test_encoder_and_decoder():
    """
    Feature: Transformer API
    Description: Test Transformer model with encoder decode input
    Expectation: No exception
    """
    model = Transformer(batch_size=2,
                        src_seq_length=20,
                        tgt_seq_length=10,
                        encoder_layers=1,
                        decoder_layers=2,
                        hidden_size=64,
                        ffn_hidden_size=64)

    encoder_input_value = Tensor(np.ones((2, 20, 64)), dtype.float32)
    encoder_input_mask = Tensor(np.ones((2, 20, 20)), dtype.float16)

    decoder_input_value = Tensor(np.ones((2, 10, 64)), dtype.float32)
    decoder_input_mask = Tensor(np.ones((2, 10, 10)), dtype.float16)
    memory_mask = Tensor(np.ones((2, 10, 20)), dtype.float16)

    _cell_graph_executor.compile(model, encoder_input_value, encoder_input_mask,
                                 decoder_input_value, decoder_input_mask, memory_mask)


def test_transformer_encoder():
    """
    Feature: TransformerEncoder API
    Description: Test Transformer Encoder model with valid input
    Expectation: No exception
    """
    model = TransformerEncoder(batch_size=2,
                               seq_length=16,
                               num_layers=2,
                               hidden_size=8,
                               ffn_hidden_size=64,
                               num_heads=2)

    encoder_input_value = Tensor(np.ones((2, 16, 8)), dtype.float32)
    encoder_input_mask = Tensor(np.ones((2, 16, 16)), dtype.float16)

    _cell_graph_executor.compile(model,
                                 encoder_input_value,
                                 encoder_input_mask)


def test_transformer_encoder_layer():
    """
    Feature: TransformerEncoderLayer API
    Description: Test Transformer Encoder Layer model with valid input
    Expectation: No exception
    """
    model = TransformerEncoderLayer(batch_size=2,
                                    hidden_size=8,
                                    ffn_hidden_size=64,
                                    seq_length=16,
                                    num_heads=2)

    encoder_input_value = Tensor(np.ones((2, 16, 8)), dtype.float32)
    encoder_input_mask = Tensor(np.ones((2, 16, 16)), dtype.float16)

    _cell_graph_executor.compile(model,
                                 encoder_input_value,
                                 encoder_input_mask)


def test_transformer_encoder_layer_post_layernorm():
    """
    Feature: TransformerEncoderLayer API
    Description: Test Transformer Encoder Layer model with post_layernorm_residual=True
    Expectation: No exception
    """
    model = TransformerEncoderLayer(batch_size=2,
                                    seq_length=16,
                                    hidden_size=8,
                                    ffn_hidden_size=64,
                                    num_heads=2,
                                    post_layernorm_residual=True)

    encoder_input_value = Tensor(np.ones((2, 16, 8)), dtype.float32)
    encoder_input_mask = Tensor(np.ones((2, 16, 16)), dtype.float16)

    _cell_graph_executor.compile(model,
                                 encoder_input_value,
                                 encoder_input_mask)


def test_transformer_decoder():
    """
    Feature: TransformerDecoder API
    Description: Test Transformer Decoder model
    Expectation: No exception
    """
    model = TransformerDecoder(num_layers=1,
                               batch_size=2,
                               src_seq_length=20,
                               tgt_seq_length=10,
                               hidden_size=64,
                               ffn_hidden_size=64,
                               num_heads=2)

    encoder_input_value = Tensor(np.ones((2, 20, 64)), dtype.float32)

    decoder_input_value = Tensor(np.ones((2, 10, 64)), dtype.float32)
    decoder_input_mask = Tensor(np.ones((2, 10, 10)), dtype.float16)
    memory_mask = Tensor(np.ones((2, 10, 20)), dtype.float16)

    _cell_graph_executor.compile(model, decoder_input_value, decoder_input_mask,
                                 encoder_input_value,
                                 memory_mask)


def test_transformer_decoder_layer():
    """
    Feature: TransformerDecoderLayer API
    Description: Test Transformer DecoderLayer model
    Expectation: No exception
    """
    model = TransformerDecoderLayer(batch_size=2,
                                    src_seq_length=20,
                                    tgt_seq_length=10,
                                    hidden_size=64,
                                    ffn_hidden_size=64,
                                    num_heads=2)

    encoder_input_value = Tensor(np.ones((2, 20, 64)), dtype.float32)

    decoder_input_value = Tensor(np.ones((2, 10, 64)), dtype.float32)
    decoder_input_mask = Tensor(np.ones((2, 10, 10)), dtype.float16)
    memory_mask = Tensor(np.ones((2, 10, 20)), dtype.float16)

    _cell_graph_executor.compile(model, decoder_input_value, decoder_input_mask,
                                 encoder_input_value,
                                 memory_mask)


def test_multi_head_attention():
    """
    Feature: MultiHeadAttention API
    Description: Test MultiHeadAttention model
    Expectation: No exception
    """
    model = MultiHeadAttention(hidden_size=15,
                               src_seq_length=20,
                               tgt_seq_length=20,
                               batch_size=2,
                               num_heads=3)
    from_tensor = Tensor(np.ones((2, 20, 15)), dtype.float32)
    to_tensor = Tensor(np.ones((2, 20, 15)), dtype.float16)
    attention_mask = Tensor(np.ones((2, 20, 20)), dtype.float16)

    _cell_graph_executor.compile(model, from_tensor, to_tensor, to_tensor, attention_mask)


@pytest.mark.parametrize('batch_size', [1, 2, None, 4])
def test_multi_head_attention_diff_batch_size(batch_size):
    """
    Feature: MultiHeadAttention with different batch size for training
    Description: Test the batch size of MultiHeadAttention in [int, None]
    Expectation: No exception
    """
    model = MultiHeadAttention(hidden_size=15,
                               src_seq_length=20,
                               tgt_seq_length=20,
                               batch_size=batch_size,
                               num_heads=3)
    from_tensor = Tensor(np.ones((3, 20, 15)), dtype.float32)
    to_tensor = Tensor(np.ones((3, 20, 15)), dtype.float16)
    attention_mask = Tensor(np.ones((3, 20, 20)), dtype.float16)

    _cell_graph_executor.compile(model, from_tensor, to_tensor, to_tensor, attention_mask)


@pytest.mark.parametrize('from_tensor,to_tensor', [(Tensor(np.ones((20, 15)), dtype.float32),
                                                    Tensor(np.ones((20, 15)), dtype.float16)),
                                                   (Tensor(np.ones((3, 20, 15)), dtype.float32),
                                                    Tensor(np.ones((3, 20, 15)), dtype.float16))])
def test_multi_head_attention_wo_mask_diff_inputs(from_tensor, to_tensor):
    """
    Feature: MultiHeadAttention without mask
    Description: Test MultiHeadAttention without mask and different inputs
    Expectation: No exception
    """
    model = MultiHeadAttention(hidden_size=15,
                               src_seq_length=20,
                               tgt_seq_length=20,
                               batch_size=None,
                               num_heads=3)

    _cell_graph_executor.compile(model, from_tensor, to_tensor, to_tensor, None)


def test_multi_head_attention_fp32_dtype():
    """
    Feature: MultiHeadAttention with fp32
    Description: Test MultiHeadAttention with float32 compute dtype
    Expectation: No exception
    """
    model = MultiHeadAttention(hidden_size=15,
                               src_seq_length=20,
                               tgt_seq_length=20,
                               compute_dtype=dtype.float32,
                               batch_size=2,
                               num_heads=3)
    from_tensor = Tensor(np.ones((2, 20, 15)), dtype.float32)
    to_tensor = Tensor(np.ones((2, 20, 15)), dtype.float32)
    attention_mask = Tensor(np.ones((2, 20, 20)), dtype.float32)
    _cell_graph_executor.compile(model, from_tensor, to_tensor, to_tensor, attention_mask)


@pytest.mark.parametrize('batch_size', [1, 2, None, 4])
def test_transformer_encoder_layer_diff_batch_size(batch_size):
    """
    Feature: TransformerEncoderLayer with different batch size for training
    Description: Test the batch size of TransformerEncoderLayer in [int, None]
    Expectation: No exception
    """
    model = TransformerEncoderLayer(batch_size=batch_size,
                                    hidden_size=8,
                                    ffn_hidden_size=64,
                                    seq_length=16,
                                    num_heads=2)
    encoder_input_value = Tensor(np.ones((2, 16, 8)), dtype.float32)
    encoder_input_mask = Tensor(np.ones((2, 16, 16)), dtype.float16)

    model(encoder_input_value, encoder_input_mask)


@pytest.mark.parametrize('attention_mask', [Tensor(np.ones((2, 16, 16)), dtype.float16),
                                            None])
def test_transformer_encoder_layer_wo_mask(attention_mask):
    """
    Feature: TransformerEncoderLayer without mask
    Description: Test TransformerEncoderLayer without mask
    Expectation: No exception
    """
    model = TransformerEncoderLayer(batch_size=None,
                                    hidden_size=8,
                                    ffn_hidden_size=64,
                                    seq_length=16,
                                    num_heads=2)
    encoder_input_value = Tensor(np.ones((2, 16, 8)), dtype.float32)

    model(encoder_input_value, attention_mask)


@pytest.mark.parametrize('shape', [(2, 16, 8), (32, 8)])
def test_transformer_encoder_layer_diff_inputs(shape):
    """
    Feature: TransformerEncoderLayer with different inputs
    Description: Test TransformerEncoderLayer with different inputs
    Expectation: No exception
    """
    model = TransformerEncoderLayer(batch_size=None,
                                    hidden_size=8,
                                    ffn_hidden_size=64,
                                    seq_length=16,
                                    num_heads=2)
    encoder_input_value = Tensor(np.ones(shape), dtype.float32)

    model(encoder_input_value, None)


@pytest.mark.parametrize('batch_size', [1, 2, None, 4])
def test_transformer_decoder_layer_diff_batch_size(batch_size):
    """
    Feature: TransformerDecoderLayer with different batch size for training
    Description: Test the batch size of TransformerDecoderLayer in [int, None]
    Expectation: No exception
    """
    model = TransformerDecoderLayer(batch_size=batch_size,
                                    hidden_size=64,
                                    ffn_hidden_size=64,
                                    num_heads=2,
                                    src_seq_length=20,
                                    tgt_seq_length=10)
    encoder_input_value = Tensor(np.ones((2, 20, 64)), dtype.float32)
    decoder_input_value = Tensor(np.ones((2, 10, 64)), dtype.float32)
    decoder_input_mask = Tensor(np.ones((2, 10, 10)), dtype.float16)
    memory_mask = Tensor(np.ones((2, 10, 20)), dtype.float16)
    model(decoder_input_value, decoder_input_mask, encoder_input_value, memory_mask)


@pytest.mark.parametrize('decoder_input_mask,memory_mask',
                         [(None, None), (Tensor(np.ones((2, 10, 10)), dtype.float16), None),
                          (None, Tensor(np.ones((2, 10, 20)), dtype.float16))])
def test_transformer_decoder_layer_empty_mask(decoder_input_mask, memory_mask):
    """
    Feature: TransformerDecoderLayer with empty mask
    Description: Test TransformerDecoderLayer with empty mask
    Expectation: No exception
    """
    model = TransformerDecoderLayer(batch_size=4,
                                    hidden_size=64,
                                    ffn_hidden_size=64,
                                    num_heads=2,
                                    src_seq_length=20,
                                    tgt_seq_length=10)
    encoder_input_value = Tensor(np.ones((2, 20, 64)), dtype.float32)
    decoder_input_value = Tensor(np.ones((2, 10, 64)), dtype.float32)
    model(decoder_input_value, decoder_input_mask, encoder_input_value, memory_mask)


@pytest.mark.parametrize('activation',
                         [MyActivation, MyActivationNoShard])
def test_transformer_decoder_layer_custom_activation(activation):
    """
    Feature: TransformerDecoderLayer with custom activation
    Description: Test TransformerDecoderLayer with custom activation
    Expectation: No exception
    """
    model = TransformerDecoderLayer(batch_size=4,
                                    hidden_size=64,
                                    ffn_hidden_size=64,
                                    num_heads=2,
                                    hidden_act=activation,
                                    src_seq_length=20,
                                    tgt_seq_length=10)
    encoder_input_value = Tensor(np.ones((2, 20, 64)), dtype.float32)
    decoder_input_value = Tensor(np.ones((2, 10, 64)), dtype.float32)
    model(decoder_input_value, None, encoder_input_value, None)


@pytest.mark.parametrize('activation',
                         [0, None, -1])
def test_transformer_decoder_layer_wrong_activation(activation):
    """
    Feature: TransformerDecoderLayer with wrong activation
    Description: Test TransformerDecoderLayer with wrong activation
    Expectation: TypeError
    """
    with pytest.raises(TypeError):
        TransformerDecoderLayer(batch_size=4,
                                hidden_size=64,
                                ffn_hidden_size=64,
                                num_heads=2,
                                hidden_act=activation,
                                src_seq_length=20,
                                tgt_seq_length=10)


@pytest.mark.parametrize('encoder_shape,decoder_shape', [((2, 20, 64), (2, 10, 64)),
                                                         ((20, 64), (10, 64))])
def test_transformer_decoder_layer_diff_inputs(encoder_shape, decoder_shape):
    """
    Feature: TransformerDecoderLayer with different inputs
    Description: Test TransformerDecoderLayer with different inputs
    Expectation: No exception
    """
    model = TransformerDecoderLayer(batch_size=None,
                                    hidden_size=64,
                                    ffn_hidden_size=64,
                                    num_heads=2,
                                    src_seq_length=20,
                                    tgt_seq_length=10)
    encoder_input_value = Tensor(np.ones(encoder_shape), dtype.float32)
    decoder_input_value = Tensor(np.ones(decoder_shape), dtype.float32)
    model(decoder_input_value, None, encoder_input_value, None)


@pytest.mark.parametrize('hidden_act', [MyActivation, "relu"])
def test_transformer_hidden_act(hidden_act):
    """
    Feature: Transformer with different hidden activation
    Description: Test Transformer with different hidden activation
    Expectation: No exception
    """
    model = Transformer(batch_size=2,
                        encoder_layers=1,
                        decoder_layers=2,
                        hidden_size=64,
                        hidden_act=hidden_act,
                        ffn_hidden_size=64,
                        src_seq_length=20,
                        tgt_seq_length=10)
    encoder_input_value = Tensor(np.ones((2, 20, 64)), dtype.float32)
    encoder_input_mask = Tensor(np.ones((2, 20, 20)), dtype.float16)
    decoder_input_value = Tensor(np.ones((2, 10, 64)), dtype.float32)
    decoder_input_mask = Tensor(np.ones((2, 10, 10)), dtype.float16)
    memory_mask = Tensor(np.ones((2, 10, 20)), dtype.float16)
    model(encoder_input_value, encoder_input_mask, decoder_input_value,
          decoder_input_mask, memory_mask)


def test_transformer_with_wrong_hidden_act_lambda_func():
    """
    Feature: Transformer with hidden activation - lambda_func
    Description: Test Transformer with hidden activation - lambda_func
    Expectation: TypeError
    """
    with pytest.raises(TypeError):
        Transformer(batch_size=2,
                    encoder_layers=1,
                    decoder_layers=2,
                    hidden_size=64,
                    hidden_act=lambda x: x,
                    ffn_hidden_size=64,
                    src_seq_length=20,
                    tgt_seq_length=10)


def test_transformer_with_wrong_hidden_act_valid_str():
    """
    Feature: Transformer with hidden activation - valid string
    Description: Test Transformer with hidden activation - valid string
    Expectation: KeyError
    """
    with pytest.raises(KeyError):
        Transformer(batch_size=2,
                    encoder_layers=1,
                    decoder_layers=2,
                    hidden_size=64,
                    hidden_act="no_string",
                    ffn_hidden_size=64,
                    src_seq_length=20,
                    tgt_seq_length=10)


def test_feedforward():
    """
    Feature: Feedforward
    Description: Test Feedforward module
    Expectation: No exception
    """
    model = FeedForward(hidden_size=15,
                        ffn_hidden_size=30,
                        dropout_rate=0.1,
                        hidden_act='relu')
    tensor = Tensor(np.ones((2, 20, 15)), dtype.float32)
    _cell_graph_executor.compile(model, tensor)


def test_cross_entropy_loss():
    """
    Feature: CrossEntropyLoss
    Description: Test CrossEntropyLoss with fake data
    Expectation: No exception
    """
    model = CrossEntropyLoss()
    logits = Tensor(np.array([[3, 5, 6, 9, 12, 33, 42, 12, 32, 72]]), dtype.float32)
    labels_np = np.array([1]).astype(np.int32)
    input_mask = Tensor(np.ones(1).astype(np.float32))
    labels = Tensor(labels_np)
    _cell_graph_executor.compile(model, logits, labels, input_mask)


def test_attention_mask():
    """
    Feature: AttentionMask
    Description: Test AttentionMask module
    Expectation: No exception
    """
    model = AttentionMask(seq_length=19)
    inputs = Tensor(np.ones((2, 19)), dtype.float32)
    _cell_graph_executor.compile(model, inputs)


def test_lower_triangular_mask_with_dynamic():
    """
    Feature: LowerTriangularMaskWithDynamic
    Description: Test LowerTriangularMaskWithDynamic module
    Expectation: No exception
    """
    model = LowerTriangularMaskWithDynamic(seq_length=19)
    inputs = Tensor(np.ones((2, 19)), dtype.float32)
    _cell_graph_executor.compile(model, inputs)


def test_fixed_sparse_attention():
    """
    Feature: FixedSparseAttention
    Description: Test FixedSparseAttention module
    Expectation: No exception
    """
    model = FixedSparseAttention(batch_size=2,
                                 seq_length=1024,
                                 size_per_head=64,
                                 num_heads=8,
                                 block_size=64)
    q = Tensor(np.ones((2, 1024, 512)), dtype.float16)
    k = Tensor(np.ones((2, 1024, 512)), dtype.float16)
    v = Tensor(np.ones((2, 1024, 512)), dtype.float16)
    mask = Tensor(np.ones((2, 1024, 1024)), dtype.float32)
    _cell_graph_executor.compile(model, q, k, v, mask)


def test_multi_head_attention_with_wrong_3d_inputs():
    """
    Feature: MultiHeadAttention with batch_size=None and different batched inputs
    Description: Test MultiHeadAttention with batch_size=None and different batched 3d inputs
    Expectation: ValueError
    """
    model = MultiHeadAttention(hidden_size=15,
                               src_seq_length=20,
                               tgt_seq_length=20,
                               batch_size=None,
                               num_heads=3)
    from_tensor = Tensor(np.ones((3, 20, 15)), dtype.float32)
    to_tensor = Tensor(np.ones((5, 20, 15)), dtype.float16)
    attention_mask = Tensor(np.ones((3, 20, 20)), dtype.float16)
    with pytest.raises(ValueError):
        _cell_graph_executor.compile(model, from_tensor, to_tensor, to_tensor, attention_mask)


def test_multi_head_attention_with_wrong_2d_inputs():
    """
    Feature: MultiHeadAttention with batch_size=None and different batched inputs
    Description: Test MultiHeadAttention with batch_size=None and different batched 2d inputs
    Expectation: ValueError
    """
    model = MultiHeadAttention(hidden_size=15,
                               src_seq_length=20,
                               tgt_seq_length=20,
                               batch_size=None,
                               num_heads=3)
    from_tensor = Tensor(np.ones((60, 15)), dtype.float32)
    to_tensor = Tensor(np.ones((100, 15)), dtype.float16)
    attention_mask = Tensor(np.ones((3, 20, 20)), dtype.float16)
    with pytest.raises(ValueError):
        _cell_graph_executor.compile(model, from_tensor, to_tensor, to_tensor, attention_mask)


def test_incremental_prediction_first_iterator():
    """
    Feature: MultiHeadAttention with incremental prediction
    Description: Test MultiHeadAttention with incremental prediction in the first iterator
    Expectation: No Expectation
    """
    # Step 1: set is_first_iteration=True, and input the full sequence length's state.
    # We need to prepare the memory parameters for saving key and value states firstly.
    from_tensor = Tensor(np.ones((2, 20, 15)), dtype.float32)
    to_tensor = Tensor(np.ones((2, 20, 15)), dtype.float16)
    attention_mask = Tensor(np.ones((2, 20, 20)), dtype.float16)
    key_past = Tensor(np.zeros(shape=(2, 3, 5, 20)), dtype.float16)
    value_past = Tensor(np.zeros(shape=(2, 3, 20, 5)), dtype.float16)
    batch_valid_length = Tensor(np.ones((2,)), dtype.int32)

    model = MultiHeadAttention(batch_size=2,
                               hidden_size=15,
                               src_seq_length=20,
                               tgt_seq_length=20,
                               num_heads=3,
                               use_past=True)
    model.add_flags_recursive(is_first_iteration=True)
    model(from_tensor, to_tensor, to_tensor, attention_mask, key_past, value_past, batch_valid_length)


def test_incremental_prediction_second_iterator():
    """
    Feature: MultiHeadAttention with incremental prediction
    Description: Test MultiHeadAttention with incremental prediction in the second iterator
    Expectation: No Expectation
    """
    model = MultiHeadAttention(batch_size=2,
                               hidden_size=15,
                               src_seq_length=20,
                               tgt_seq_length=20,
                               num_heads=3,
                               use_past=True)
    key_past = Tensor(np.zeros(shape=(2, 3, 5, 20)), dtype.float16)
    value_past = Tensor(np.zeros(shape=(2, 3, 20, 5)), dtype.float16)
    batch_valid_length = Tensor(np.ones((2,)), dtype.int32)
    # Set is_first_iteration=True to generate the full memory states
    from_tensor = Tensor(np.ones((2, 1, 15)), dtype.float32)
    to_tensor = Tensor(np.ones((2, 1, 15)), dtype.float16)
    attention_mask = Tensor(np.ones((2, 1, 20)), dtype.float16)
    # Step 2: set is_first_iteration=False, and pass the single word to run the prediction rather than the
    # full sequence.
    model.add_flags_recursive(is_first_iteration=False)
    model(from_tensor, to_tensor, to_tensor, attention_mask, key_past, value_past, batch_valid_length)
