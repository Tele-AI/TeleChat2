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
"""whisper model"""

import math
from typing import Optional, Tuple

import numpy as np
import mindspore as ms
from mindspore import nn
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.context import ParallelMode
from mindspore.common.initializer import initializer
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation

from mindformers.core.loss.loss import CrossEntropyLoss
from mindformers.modules.layers import Linear, LayerNorm
from mindformers.modules.transformer import TransformerOpParallelConfig, LowerTriangularMaskWithDynamic
from mindformers.modules.flash_attention import FlashAttention
from mindformers.modules.activation import get_activation
from mindformers.version_control import get_dropout
from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType

from .configuration_whisper import WhisperConfig
from ..utils import lazy_inline


# pylint: disable=W0622
def set_layer_pipeline(layers, num_layers_of_each_stage, parallel_config, type="encoder"):
    """set pipeline stage for layers"""
    if (not isinstance(num_layers_of_each_stage, list)) or (len(num_layers_of_each_stage) != 2):
        raise TypeError(
            "num_layers_of_each_stage should be a list which contains 2 lists, eg. [[1,2],[3,4,5]]. "
            "[1,2] means num_encoder_layers=3, stage_0 has 1 layer, stage_1 has 2 layers. "
            "[3,4,5] means num_decoder_layers=12, stage_2 has 3 layers, stage_3 has 4 layers, stage_4 has 5 layers.")
    if (len(num_layers_of_each_stage[0]) + len(num_layers_of_each_stage[1])) != parallel_config.pipeline_stage:
        raise ValueError(f"sum(sum(num_layers_of_each_stage)) != parallel_config.pipeline_stage")
    if type == "encoder":
        index = 0
        pre_stages = 0
    elif type == "decoder":
        index = 1
        pre_stages = len(num_layers_of_each_stage[index])
    else:
        raise ValueError(f"type should be in ['encoder', 'decoder'], but get {type}")

    num_layers = len(layers)
    num_layer_list = num_layers_of_each_stage[index]
    if num_layers != sum(num_layer_list):
        raise ValueError(f"num_layers({num_layers}) != sum of num_layer_list({sum(num_layer_list)})")

    layer_list = np.array([np.sum(num_layer_list[:i + 1]) for i in range(len(num_layer_list))])
    for layer_id in range(num_layers):
        pp_id = int(np.sum(layer_list < layer_id + 1)) + pre_stages
        layers[layer_id].pipeline_stage = pp_id


class Conv1d(nn.Conv1d):
    r"""
    1D convolution layer.

    Applies a 1D convolution over an input tensor which is typically of shape :math:`(N, C_{in}, L_{in})`,
    where :math:`N` is batch size, :math:`C` is channel number, :math:`L` is input sequence width.

    The output is calculated based on formula:

    .. math::

        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{in} - 1} \text{ccor}({\text{weight}(C_{\text{out}_j}, k), \text{X}(N_i, k)})

    where :math:`bias` is the output channel bias, :math:`ccor` is
    the `cross-correlation <https://en.wikipedia.org/wiki/Cross-correlation>`_,
    :math:`weight` is the convolution kernel value and :math:`X` represents the input feature map.

    Here are the indices' meanings:

    - :math:`i` corresponds to the batch number, the range is :math:`[0, N-1]`,
      where :math:`N` is the batch size of the input.

    - :math:`j` corresponds to the output channel, the range is :math:`[0, C_{out}-1]`,
      where :math:`C_{out}` is the number of
      output channels, which is also equal to the number of kernels.

    - :math:`k` corresponds to the input channel, the range is :math:`[0, C_{in}-1]`,
      where :math:`C_{in}` is the number of
      input channels, which is also equal to the number of channels in the convolutional kernels.

    Therefore, in the above formula, :math:`{bias}(C_{\text{out}_j})` represents the bias of the :math:`j`-th
    output channel, :math:`{weight}(C_{\text{out}_j}, k)` represents the slice of the :math:`j`-th convolutional
    kernel in the :math:`k`-th channel, and :math:`{X}(N_i, k)` represents the slice of the :math:`k`-th input
    channel in the :math:`i`-th batch of the input feature map.

    The shape of the convolutional kernel is given by :math:`(\text{kernel_size})`,
    where :math:`\text{kernel_size}` is the width of the kernel.
    If we consider the input and output channels as well as the `group` parameter, the complete kernel shape
    will be :math:`(C_{out}, C_{in} / \text{group}, \text{kernel_size})`,
    where `group` is the number of groups dividing `x`'s input channel when applying group convolution.

    For more details about convolution layer, please refer to `Gradient Based Learning Applied to Document Recognition
    <http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf>`_
    and `ConvNets <http://cs231n.github.io/convolutional-networks/>`_ .

    Note:
        On Ascend platform, only group convolution in depthwise convolution scenarios is supported.
        That is, when `group>1`, condition `in\_channels` = `out\_channels` = `group` must be satisfied.

    Args:
        in_channels (int): The channel number of the input tensor of the Conv1d layer.
        out_channels (int): The channel number of the output tensor of the Conv1d layer.
        kernel_size (int): Specifies the width of the 1D convolution kernel.
        stride (int, optional): The movement stride of the 1D convolution kernel. Default: ``1`` .
        pad_mode (str, optional): Specifies the padding mode with a padding value of 0. It can be set to:
            ``"same"`` , ``"valid"`` or ``"pad"`` . Default: ``"same"`` .

            - ``"same"``: Pad the input at the begin and end so that the shape of input and output
              are the same when `stride` is set to ``1``.
              The amount of padding to is calculated by the operator internally. If the amount is even, it is
              uniformly distributed around the input, if it is odd, the excess padding is goes to the right side.
              If this mode is set, `padding` must be 0.
            - ``"valid"``: No padding is applied to the input, and the output returns the maximum
              possible length. Extra pixels that could not complete a full stride will
              be discarded. If this mode is set, `padding` must be 0.
            - ``"pad"``: Pad the input with a specified amount. In this mode, the amount of padding
              at the begin and end is determined by the `padding` parameter.
              If this mode is set, `padding` must be greater than or equal to 0.

        padding (Union(int, tuple[int], list[int]), optional):  Specifies the amount of padding to apply on
            both side of `input` when `pad_mode` is set to ``"pad"``. The
            paddings of left and right are the same, equal to padding or padding[0] when padding is a tuple of
            1 integer. Default: ``0`` .
        dilation (Union(int, tuple[int]), optional): Specifies the dilation rate to use for dilated convolution.
            It can be a single int or a tuple of 1 integer.
            Assuming :math:`dilation=(d0,)`, the convolutional kernel samples the input with a
            spacing of :math:`d0-1` elements in the width direction.
            The value should be in the ranges [1, L].
            Default: ``1`` .
        group (int, optional): Splits filter into groups, `in_channels` and `out_channels` must be
            divisible by `group`. Default: ``1`` .
        has_bias (bool, optional): Whether the Conv1d layer has a bias parameter. Default: ``False`` .
        weight_init (Union[Tensor, str, Initializer, numbers.Number], optional):
            Initialization method of weight parameter.
            It can be a Tensor, a string, an Initializer or a numbers.Number. When a string is specified,
            values from ``'TruncatedNormal'`` , ``'Normal'`` , ``'Uniform'`` , ``'HeUniform'`` and ``'XavierUniform'``
            distributions as well as constant 'One' and 'Zero' distributions are possible. Alias ``'xavier_uniform'`` ,
            ``'he_uniform'`` , ``'ones'`` and ``'zeros'`` are acceptable. Uppercase and lowercase are both acceptable.
            Refer to the values of
            `Initializer <https://www.mindspore.cn/docs/en/master/api_python/mindspore.common.initializer.html>`_,
            for more details. Default: ``None`` , weight will be initialized using ``'HeUniform'``.
        bias_init (Union[Tensor, str, Initializer, numbers.Number], optional): Initialization method of bias parameter.
            Available initialization methods are the same as 'weight_init'. Refer to the values of
            `Initializer <https://www.mindspore.cn/docs/en/master/api_python/mindspore.common.initializer.html>`_,
            for more details. Default: ``None`` , bias will be initialized using ``'Uniform'``.
        compute_dtype (:class:`mindspore.dtype`): Dtype of compute. Default: ``mstype.float16`` .
        dtype (:class:`mindspore.dtype`): Dtype of Parameters. Default: ``mstype.float32`` .

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, L_{in})` .

    Outputs:
        Tensor of shape :math:`(N, C_{out}, L_{out})`.

        pad_mode is ``'same'``:

        .. math::
            L_{out} = \left \lceil{\frac{L_{in}}{\text{stride}}} \right \rceil

        pad_mode is ``'valid'``:

        .. math::
            L_{out} = \left \lceil{\frac{L_{in} - \text{dilation} \times (\text{kernel_size} - 1) }
            {\text{stride}}} \right \rceil

        pad_mode is ``'pad'``:

        .. math::
            L_{out} = \left \lfloor{\frac{L_{in} + 2 \times padding - (\text{kernel_size} - 1) \times
            \text{dilation} - 1 }{\text{stride}} + 1} \right \rfloor

    Raises:
        TypeError: If `in_channels`, `out_channels`, `kernel_size`, `stride`, `padding` or `dilation` is not an int.
        ValueError: If `in_channels`, `out_channels`, `kernel_size`, `stride` or `dilation` is less than 1.
        ValueError: If `padding` is less than 0.
        ValueError: If `pad_mode` is not one of 'same', 'valid', 'pad'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> import numpy as np
        >>> net = nn.Conv1d(120, 240, 4, has_bias=False, weight_init='normal')
        >>> x = Tensor(np.ones([1, 120, 640]), mindspore.float32)
        >>> output = net(x).shape
        >>> print(output)
        (1, 240, 640)
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 pad_mode='same',
                 padding=0,
                 dilation=1,
                 group=1,
                 has_bias=False,
                 weight_init=None,
                 bias_init=None,
                 compute_dtype=mstype.float16,
                 dtype=mstype.float32):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            pad_mode,
            padding,
            dilation,
            group,
            has_bias,
            weight_init,
            bias_init,
            dtype=dtype)
        self.compute_dtype = compute_dtype

    def construct(self, x):
        """forward"""
        ori_dtype = F.dtype(x)
        x = self.expand_dims(x, 2)
        weight = self.cast(self.weight, self.compute_dtype)
        output = self.conv2d(x, weight)
        if self.has_bias:
            output = self.bias_add(output, self.cast(self.bias, self.compute_dtype))

        output = self.squeeze(output)
        output = F.cast(output, ori_dtype)
        return output


class Embedding(nn.Cell):
    """
    Embedding Layer.

    Args:
            - **vocab_size** (int): Size of the dictionary of embeddings.
            - **embedding_size** (int): The size of each embedding vector.
            - **param_init_type** (mstype): The param init type, default mstype.float32.
            - **param_init** (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the embedding_table.
                Refer to class `initializer` for the values of string when a string
                is specified. Default: 'normal'.
            - **parallel_optimizer (bool): It is used to filter the weight shard operation in semi auto or auto parallel
                mode. It works only when enable parallel optimizer in `mindspore.context.set_auto_parallel_context()`.
                Default: True.
    Inputs:
            - **input_ids** (Tensor) - The tokenized inputs with datatype int32 with shape (batch_size, seq_length)

    Outputs:
            - **output** (Tensor) - The embedding vector for the input with shape (batch_size,
              seq_length, embedding_size).
    """

    def __init__(self, vocab_table_size, embedding_size, param_init_type=mstype.float32, param_init='normal',
                 parallel_optimizer=False):
        super().__init__()
        self.vocab_table_size = vocab_table_size
        self.embedding_size = embedding_size
        self.embedding_weight = ms.Parameter(
            initializer(param_init, [self.vocab_table_size, self.embedding_size], dtype=param_init_type),
            name='embedding_weight', parallel_optimizer=parallel_optimizer)
        self.gather = P.Gather()

    def construct(self, input_ids):
        """Forward of vocab embedding."""
        output = self.gather(self.embedding_weight, input_ids, 0)
        return output, self.embedding_weight.value()

    def shard(self, parallel_config):
        """sharding for embedding"""
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        if parallel_config.vocab_emb_dp:
            self.gather.shard(((1, 1), (dp, 1)))
        else:
            if self.vocab_table_size % mp != 0:
                self.gather.shard(((1, 1), (dp, 1)))
            else:
                self.gather.shard(((mp, 1), (dp, 1)))


class LMHead(nn.Cell):
    """LM head"""
    def __init__(self, in_channels, out_channels, compute_type=mstype.float16):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dtype = compute_type
        self.matmul = P.MatMul(transpose_b=True)
        self.reshape = P.Reshape()
        self.shape = P.Shape()

    def construct(self, state, embedding_table):
        out_shape = self.shape(state)[:-1] + (self.out_channels,)
        state = self.reshape(state, (-1, self.in_channels))
        logits = self.matmul(self.cast(state, self.dtype), self.cast(embedding_table, self.dtype))
        logits = self.reshape(logits, out_shape)
        return logits


class WhisperPositionalEmbedding(nn.Cell):
    """PositionalEmbedding For Whisper Model"""
    def __init__(self, vocab_table_size, embedding_size, param_init_type=mstype.float32, param_init='normal',
                 parallel_optimizer=False):
        super().__init__()
        self.vocab_table_size = vocab_table_size
        self.embedding_size = embedding_size
        self.embedding_weight = ms.Parameter(
            initializer(param_init, [self.vocab_table_size, self.embedding_size], dtype=param_init_type),
            name='embedding_weight', parallel_optimizer=parallel_optimizer)
        self.shape = P.Shape()
        self.strideslice = P.StridedSlice().shard(((1, 1),))

    def construct(self, input_ids, past_key_values_length=0):
        """Forward of vocab embedding."""
        _, seq_len = self.shape(input_ids)
        length = past_key_values_length + seq_len
        output = self.strideslice(self.embedding_weight, (past_key_values_length, 0), (length, self.embedding_size),
                                  (1, 1))
        return output


class WhisperAttention(nn.Cell):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            is_decoder: bool = False,
            bias: bool = True,
            is_causal: bool = False,
            config: Optional[WhisperConfig] = None,
            compute_dtype=mstype.float16,
            softmax_compute_dtype=mstype.float32,
            param_init_type=mstype.float32,
            use_flash_attention=False,
            is_dynamic=False,
            use_attention_mask=True,
            parallel_config=TransformerOpParallelConfig()
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = get_dropout(dropout)
        self.head_dim = embed_dim // num_heads
        self.bias = bias
        self.config = config

        self.compute_dtype = compute_dtype
        self.use_flash_attention = use_flash_attention
        self.softmax_dtype = softmax_compute_dtype
        self.param_init_type = param_init_type
        self.is_dynamic = is_dynamic
        self.parallel_config = parallel_config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        self.k_proj = Linear(embed_dim,
                             embed_dim,
                             has_bias=False,
                             compute_dtype=self.compute_dtype,
                             param_init_type=param_init_type,
                             skip_redistribution=is_dynamic)
        self.v_proj = Linear(embed_dim,
                             embed_dim,
                             has_bias=bias,
                             compute_dtype=self.compute_dtype,
                             param_init_type=param_init_type,
                             skip_redistribution=is_dynamic)
        self.q_proj = Linear(embed_dim,
                             embed_dim,
                             has_bias=bias,
                             compute_dtype=self.compute_dtype,
                             param_init_type=param_init_type,
                             skip_redistribution=is_dynamic)
        self.out_proj = Linear(embed_dim,
                               embed_dim,
                               has_bias=bias,
                               compute_dtype=self.compute_dtype,
                               param_init_type=param_init_type,
                               skip_redistribution=is_dynamic)

        self.shape = P.Shape()
        self.transpose = P.Transpose()
        self.reshape = P.Reshape().add_prim_attr("skip_redistribution", True)
        self.cat = P.Concat(axis=2)
        self.merger_head_transpose = P.Transpose()
        self.batch_matmul = P.BatchMatMul()
        self.batch_matmul_q_k = P.BatchMatMul(transpose_b=True)
        self.mul = P.Mul()
        self.add = P.Add()
        self.softmax = P.Softmax()
        self.cast_attn = P.Cast()
        self.tile_kv = P.Tile()
        self.slice_qkv = P.StridedSlice()

        if self.use_flash_attention:
            self.flash_attention = FlashAttention(head_num=self.num_heads,
                                                  pre_tokens=65536,
                                                  next_tokens=0,
                                                  input_layout="BNSD",
                                                  keep_prob=1. - dropout,
                                                  scale_value=1. / math.sqrt(self.head_dim),
                                                  sparse_mode=0,
                                                  use_attention_mask=use_attention_mask)
            self.flash_attention.shard(parallel_config)
        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.shard(parallel_config)

    def shard(self, parallel_config):
        """set parallel config"""
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel

        self.k_proj.shard(((dp, 1), (mp, 1)))
        if self.bias:
            self.v_proj.shard(((dp, 1), (mp, 1)), ((dp, mp), (mp,)))
            self.q_proj.shard(((dp, 1), (mp, 1)), ((dp, mp), (mp,)))
            self.out_proj.shard(((dp, mp), (1, mp)), ((dp, 1), (1,)))
        else:
            self.v_proj.shard(((dp, 1), (mp, 1)))
            self.q_proj.shard(((dp, 1), (mp, 1)))
            self.out_proj.shard(((dp, mp), (1, mp)))

        self.mul.shard(((dp, mp, 1), ()))
        self.transpose.shard(((dp, 1, mp, 1),))
        self.batch_matmul_q_k.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
        self.add.shard(((dp, 1, 1, 1), (dp, mp, 1, 1)))
        self.softmax.shard(((dp, mp, 1, 1),))
        self.batch_matmul.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
        self.merger_head_transpose.shard(((dp, mp, 1, 1),))
        self.dropout.dropout.shard(((dp, mp, 1, 1),))

    # Copied from transformers.models.bart.modeling_bart.BartAttention._shape with BART->whisper
    def _shape(self, tensor: ms.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # Copied from transformers.models.bart.modeling_bart.BartAttention.forward with BART->whisper
    def construct(
            self,
            hidden_states: ms.Tensor,
            key_value_states: Optional[ms.Tensor] = None,
            attention_mask: Optional[ms.Tensor] = None,
    ):
        """Input shape: Batch x Time x Channel"""
        ori_dtype = hidden_states.dtype

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = self.shape(hidden_states)
        if is_cross_attention:
            _, key_value_tgt_len, _ = self.shape(key_value_states)
        else:
            key_value_tgt_len = tgt_len

        # get query proj
        query_states = self.q_proj(hidden_states)

        if is_cross_attention:
            # cross_attentions
            key_states = self.cast(self.k_proj(key_value_states), self.compute_dtype)
            value_states = self.cast(self.v_proj(key_value_states), self.compute_dtype)
        else:
            # self_attention
            key_states = self.cast(self.k_proj(hidden_states), self.compute_dtype)
            value_states = self.cast(self.v_proj(hidden_states), self.compute_dtype)

        query_states = self.transpose(self.reshape(query_states, (bsz, tgt_len, self.num_heads, self.head_dim)),
                                      (0, 2, 1, 3))
        key_states = self.transpose(self.reshape(key_states, (bsz, key_value_tgt_len, self.num_heads, self.head_dim)),
                                    (0, 2, 1, 3))
        value_states = self.transpose(
            self.reshape(value_states, (bsz, key_value_tgt_len, self.num_heads, self.head_dim)), (0, 2, 1, 3))

        if self.use_flash_attention:
            context_layer = self.flash_attention(query_states, key_states, value_states, attention_mask)
            context_layer = self._merge_heads(context_layer)
        else:
            context_layer = self._attn(query_states, key_states, value_states, attention_mask)

        attn_output = self.out_proj(context_layer)
        attn_output = self.cast(attn_output, ori_dtype)

        return attn_output

    def _repeat_kv(self, x, rep):
        if rep == 1:
            return x
        bs, n_kv_head, seqlen, head_dim = self.shape(x)
        x = self.reshape(x, (bs, n_kv_head, 1, seqlen * head_dim))
        x = self.tile_kv(x, (1, 1, rep, 1))
        x = self.reshape(x, (bs, n_kv_head * rep, seqlen, head_dim))
        return x

    def _merge_heads(self, x):
        """
        convert a 4d input to a 3d output

        Inputs:
            x: input tensor

        Output:
            x_merge: the 2d output
        """
        x = self.merger_head_transpose(x, (0, 2, 1, 3))
        bs, seq_len, n_head, head_dim = self.shape(x)
        new_shape = (bs, seq_len, n_head * head_dim)
        x_merge = self.reshape(x, new_shape)
        return x_merge

    def _attn(self, query, key, value, mask):
        """
        Get the weighted score along the seq_length

        Inputs:
            query: the query matrix
            key: the key matrix
            value: the value matrix
            mask: the attention mask adder matrix with shape (batch_size,
            1, seq_length, seq_length)
        Outputs:
            weighted_values: Tensor, the weighted sum scores
        """
        # q, k: [bs, n_head, seq/1, head_dim], [bs, n_head, seq, head_dim]
        query = self.cast(self.mul(query, self.scaling), self.compute_dtype)
        score = self.batch_matmul_q_k(query, key)

        # score: [bs, n_head, seq/1, seq]
        if mask is not None:
            score = self.add(mask, score)

        attention_probs = self.softmax(self.cast_attn(score, self.softmax_dtype))
        attention_probs = self.dropout(attention_probs)
        weighted_values = self.batch_matmul(self.cast(attention_probs, self.compute_dtype), value)
        attention_merge = self._merge_heads(weighted_values)
        return attention_merge


class WhisperEncoderLayer(nn.Cell):
    """Whisper EncoderLayer"""
    def __init__(self,
                 config: WhisperConfig,
                 ):
        super().__init__()
        self.compute_dtype = config.compute_dtype
        self.layernorm_compute_dtype = config.layernorm_compute_dtype

        self.embed_dim = config.d_model

        self.self_attn = WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
            use_flash_attention=config.use_flash_attention,
            parallel_config=config.parallel_config,
            param_init_type=config.param_init_type,
            compute_dtype=config.compute_dtype,
            use_attention_mask=False
        )
        self.self_attn_layer_norm = LayerNorm((self.embed_dim,), eps=1e-05)
        self.dropout = get_dropout(config.dropout)
        self.add = P.Add()
        self.activation_dropout = get_dropout(config.activation_dropout)
        self.fc1 = Linear(self.embed_dim,
                          config.encoder_ffn_dim,
                          has_bias=True,
                          activation=config.activation_function,
                          compute_dtype=self.compute_dtype,
                          param_init_type=config.param_init_type,
                          skip_redistribution=config.is_dynamic)
        self.fc2 = Linear(config.encoder_ffn_dim,
                          self.embed_dim,
                          has_bias=True,
                          compute_dtype=self.compute_dtype,
                          param_init_type=config.param_init_type,
                          skip_redistribution=config.is_dynamic)
        self.final_layer_norm = LayerNorm((self.embed_dim,), eps=1e-05)

        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.shard(config.parallel_config)

    def shard(self, parallel_config):
        """set parallel config"""
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel

        self.self_attn_layer_norm.shard(((dp, 1, 1),))
        self.dropout.dropout.shard(((dp, 1, 1),))
        self.add.shard(((dp, 1, 1), (dp, 1, 1)))
        self.final_layer_norm.shard(((dp, 1, 1),))

        self.fc1.shard(((dp, 1), (mp, 1)), ((dp, mp), (mp,)), ((dp, mp),))
        self.activation_dropout.dropout.shard(((dp, mp, 1),))
        self.fc2.shard(((dp, mp), (1, mp)), ((dp, 1), (1,)))

    def construct(
            self,
            hidden_states: ms.Tensor,
            attention_mask: ms.Tensor,
    ) -> ms.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        # layer norm
        hidden_states = self.cast(hidden_states, self.layernorm_compute_dtype)
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.cast(hidden_states, self.compute_dtype)
        # self attention
        hidden_states = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask)
        # dropout
        hidden_states = self.dropout(hidden_states)
        # residual
        hidden_states = self.add(residual, hidden_states)

        residual = hidden_states
        # layer norm
        hidden_states = self.cast(hidden_states, self.layernorm_compute_dtype)
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.cast(hidden_states, self.compute_dtype)
        # dense 1
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_dropout(hidden_states)
        # dense 2
        hidden_states = self.fc2(hidden_states)

        hidden_states = self.dropout(hidden_states)
        # residual
        hidden_states = self.add(residual, hidden_states)

        return hidden_states


class WhisperEncoder(nn.Cell):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`WhisperEncoderLayer`].

    Args:
        config: WhisperConfig
    """

    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        self.compute_dtype = config.compute_dtype
        self.layernorm_compute_dtype = config.layernorm_compute_dtype
        self.num_layers_of_each_stage = config.num_layers_of_each_stage

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions

        self.conv1 = Conv1d(self.num_mel_bins, embed_dim, has_bias=True, kernel_size=3, padding=1, pad_mode='pad',
                            compute_dtype=config.compute_dtype, dtype=config.param_init_type)
        self.conv2 = Conv1d(embed_dim, embed_dim, has_bias=True, kernel_size=3, stride=2, padding=1, pad_mode='pad',
                            compute_dtype=config.compute_dtype, dtype=config.param_init_type)
        self.gelu = get_activation("gelu")
        self.transpose = P.Transpose()
        self.add = P.Add()
        self.dropout = get_dropout(self.dropout)

        self.embed_positions = Embedding(vocab_table_size=self.max_source_positions,
                                         embedding_size=embed_dim,
                                         param_init_type=config.embedding_init_type)
        self.embed_positions.embedding_weight.requires_grad = False

        self.layers = nn.CellList([WhisperEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layer_norm = LayerNorm((config.d_model,), eps=1e-05)
        self.less = P.Less()
        self.select = P.Select()
        self.slice = P.Slice()
        self.shape = P.Shape()
        self.ones_like = P.OnesLike()
        self.zeros_like = P.ZerosLike()
        self.mul = P.Mul()

        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.shard(config.parallel_config)

    def shard(self, parallel_config):
        """set parallel config"""
        dp = parallel_config.data_parallel

        self.conv1.conv2d.shard(((dp, 1, 1, 1), (1, 1, 1, 1)))
        self.conv1.bias_add.shard(((dp, 1, 1, 1), (1,)))
        self.conv2.conv2d.shard(((dp, 1, 1, 1), (1, 1, 1, 1)))
        self.conv2.bias_add.shard(((dp, 1, 1, 1), (1,)))
        self.gelu.gelu.shard(((dp, 1, 1),))

        self.transpose.shard(((dp, 1, 1),))
        self.add.shard(((dp, 1, 1), (1, 1)))
        self.dropout.dropout.shard(((dp, 1, 1),))

        self.less.shard(((dp, 1), ()))
        self.slice.shard(((dp, 1),))
        self.ones_like.shard(((dp, 1, 1),))
        self.mul.shard(((dp, 1, 1), (dp, 1, 1)))
        self.zeros_like.shard(((dp, 1, 1),))

        self.select.shard(((dp, 1, 1), (dp, 1, 1), (dp, 1, 1)))
        self.layer_norm.shard(((dp, 1, 1),))

        self.embed_positions.pipeline_stage = 0
        self.conv1.pipeline_stage = 0
        self.conv2.pipeline_stage = 0
        if parallel_config.pipeline_stage > 1:
            set_layer_pipeline(self.layers, self.num_layers_of_each_stage, parallel_config, "encoder")
            self.layer_norm.pipeline_stage = len(self.num_layers_of_each_stage[0]) - 1

    def construct(
            self,
            input_features,
            attention_mask=None,
            dropout_probability=None,
    ):
        """forward"""
        bs, _, _ = self.shape(input_features)
        expected_seq_length = self.max_source_positions * self.conv1.stride[1] * self.conv2.stride[1]
        if input_features.shape[-1] != expected_seq_length:
            raise ValueError(
                f"Whisper expects the mel input features to be of length {expected_seq_length},"
                f"but found {input_features.shape[-1]}. Make sure to pad the input mel "
                f"features to {expected_seq_length}."
            )

        input_features = self.cast(input_features, self.compute_dtype)
        inputs_embeds = self.conv1(input_features)
        inputs_embeds = self.gelu(inputs_embeds)
        inputs_embeds = self.conv2(inputs_embeds)
        inputs_embeds = self.gelu(inputs_embeds)

        inputs_embeds = self.transpose(inputs_embeds, (0, 2, 1))

        hidden_states = self.add(inputs_embeds, self.embed_positions.embedding_weight)
        hidden_states = self.dropout(hidden_states)

        if self.training and dropout_probability is not None:
            to_drop = self.less(dropout_probability, self.layerdrop)
        else:
            to_drop = False
        for idx, encoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if self.training and dropout_probability is not None:
                to_drop_layer = self.slice(to_drop, (0, idx), (bs, 1))
                to_drop_layer = ms.ops.expand_dims(to_drop_layer, 2)
                drop_matrix = self.ones_like(hidden_states)
                drop_matrix = self.mul(drop_matrix, to_drop_layer)
            else:
                drop_matrix = self.zeros_like(hidden_states)
            drop_matrix = self.cast(drop_matrix, mstype.bool_)

            hidden_states_ori = hidden_states
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask,
            )

            hidden_states = self.select(drop_matrix, hidden_states_ori, hidden_states)

        hidden_states = self.cast(hidden_states, self.layernorm_compute_dtype)
        hidden_states = self.layer_norm(hidden_states)

        return hidden_states


class WhisperDecoderLayer(nn.Cell):
    """WhisperDecoderLayer"""

    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.compute_dtype = config.compute_dtype
        self.layernorm_compute_dtype = config.layernorm_compute_dtype

        self.self_attn = WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
            config=config,
            param_init_type=config.param_init_type,
            use_flash_attention=config.use_flash_attention,
            compute_dtype=config.compute_dtype,
            parallel_config=config.parallel_config
        )
        self.dropout = get_dropout(config.dropout)
        self.activation_dropout = get_dropout(config.activation_dropout)
        self.add = P.Add()

        self.self_attn_layer_norm = LayerNorm((self.embed_dim,), eps=1e-05)
        self.encoder_attn = WhisperAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
            use_flash_attention=config.use_flash_attention,
            param_init_type=config.param_init_type,
            compute_dtype=config.compute_dtype,
            use_attention_mask=False,
            parallel_config=config.parallel_config
        )
        self.encoder_attn_layer_norm = LayerNorm((self.embed_dim,), eps=1e-05)
        self.fc1 = Linear(self.embed_dim,
                          config.decoder_ffn_dim,
                          has_bias=True,
                          activation=config.activation_function,
                          compute_dtype=self.compute_dtype,
                          param_init_type=config.param_init_type,
                          skip_redistribution=config.is_dynamic)
        self.fc2 = Linear(config.decoder_ffn_dim,
                          self.embed_dim,
                          has_bias=True,
                          compute_dtype=self.compute_dtype,
                          param_init_type=config.param_init_type,
                          skip_redistribution=config.is_dynamic)
        self.final_layer_norm = LayerNorm((self.embed_dim,), eps=1e-05)

        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.shard(config.parallel_config)

    def shard(self, parallel_config):
        """set parallel config"""
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel

        self.self_attn_layer_norm.shard(((dp, 1, 1),))
        self.dropout.dropout.shard(((dp, 1, 1),))
        self.add.shard(((dp, 1, 1), (dp, 1, 1)))

        self.encoder_attn_layer_norm.shard(((dp, 1, 1),))

        self.final_layer_norm.shard(((dp, 1, 1),))
        self.fc1.shard(((dp, 1), (mp, 1)), ((dp, mp), (mp,)), ((dp, mp),))
        self.activation_dropout.dropout.shard(((dp, mp, 1),))
        self.fc2.shard(((dp, mp), (1, mp)), ((dp, 1), (1,)))

    def construct(
            self,
            hidden_states: ms.Tensor,
            attention_mask: Optional[ms.Tensor] = None,
            encoder_hidden_states: Optional[ms.Tensor] = None,
            encoder_attention_mask: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        """forward"""
        residual = hidden_states

        hidden_states = self.cast(hidden_states, self.layernorm_compute_dtype)
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.cast(hidden_states, self.compute_dtype)

        # Self Attention
        hidden_states = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.add(residual, hidden_states)

        # Cross-Attention Block
        if encoder_hidden_states is not None:
            residual = hidden_states

            hidden_states = self.cast(hidden_states, self.layernorm_compute_dtype)
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            hidden_states = self.cast(hidden_states, self.compute_dtype)

            hidden_states = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.add(residual, hidden_states)

        # Fully Connected
        residual = hidden_states

        hidden_states = self.cast(hidden_states, self.layernorm_compute_dtype)
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.cast(hidden_states, self.compute_dtype)

        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.add(residual, hidden_states)

        outputs = hidden_states

        return outputs


class WhisperDecoder(nn.Cell):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`WhisperDecoderLayer`]

    Args:
        config: WhisperConfig
    """

    main_input_name = "input_ids"

    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.compute_dtype = config.compute_dtype
        self.layernorm_compute_dtype = config.layernorm_compute_dtype
        self.num_layers_of_each_stage = config.num_layers_of_each_stage

        self.dropout = get_dropout(config.dropout)
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_target_positions
        self.max_source_positions = config.max_source_positions

        self.embed_tokens = Embedding(vocab_table_size=config.vocab_size,
                                      embedding_size=config.d_model,
                                      param_init_type=config.embedding_init_type)
        self.embed_positions = WhisperPositionalEmbedding(self.max_target_positions, config.d_model,
                                                          param_init_type=config.embedding_init_type)

        self.layers = nn.CellList([WhisperDecoderLayer(config) for _ in range(config.decoder_layers)])

        self.layer_norm = LayerNorm((config.d_model,), eps=1e-05)
        self.add = P.Add()
        self.casual_mask = LowerTriangularMaskWithDynamic(seq_length=config.max_target_positions,
                                                          compute_type=config.compute_dtype,
                                                          is_dynamic=config.is_dynamic,
                                                          pad_token_id=-1,
                                                          use_flash_attention=config.use_flash_attention)
        self.select = P.Select()
        self.slice = P.Slice()
        self.less = P.Less()
        self.shape = P.Shape()
        self.ones_like = P.OnesLike()
        self.mul = P.Mul()
        self.zeros_like = P.ZerosLike()

        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.shard(config.parallel_config)

    def shard(self, parallel_config):
        """set parallel config"""
        dp = parallel_config.data_parallel

        self.casual_mask.shard(parallel_config)
        self.embed_tokens.shard(parallel_config)
        self.add.shard(((dp, 1, 1), (1, 1)))
        self.dropout.dropout.shard(((dp, 1, 1),))

        self.less.shard(((dp, 1), ()))
        self.slice.shard(((dp, 1),))
        self.ones_like.shard(((dp, 1, 1),))
        self.mul.shard(((dp, 1, 1), (dp, 1, 1)))
        self.zeros_like.shard(((dp, 1, 1),))

        self.select.shard(((dp, 1, 1), (dp, 1, 1), (dp, 1, 1)))
        self.layer_norm.shard(((dp, 1, 1),))

        if parallel_config.pipeline_stage > 1:
            set_layer_pipeline(self.layers, self.num_layers_of_each_stage, parallel_config, "decoder")
            pp_id_first = len(self.num_layers_of_each_stage[0])
            self.embed_tokens.pipeline_stage = pp_id_first
            self.embed_positions.pipeline_stage = pp_id_first
            self.layer_norm.pipeline_stage = parallel_config.pipeline_stage - 1

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def construct(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_hidden_states=None,
            inputs_embeds=None,
            dropout_probability=None,
    ):
        """forward"""
        bs, _ = self.shape(input_ids)

        embedding_table = None
        if inputs_embeds is None:
            inputs_embeds, embedding_table = self.embed_tokens(input_ids)

        attention_mask = self.casual_mask(input_ids)

        # embed positions
        if input_ids is not None:
            positions = self.embed_positions(input_ids)
        else:
            positions = self.embed_positions(inputs_embeds)

        inputs_embeds = self.cast(inputs_embeds, self.compute_dtype)
        positions = self.cast(positions, self.compute_dtype)
        hidden_states = self.add(inputs_embeds, positions)
        hidden_states = self.dropout(hidden_states)

        if self.training and dropout_probability is not None:
            to_drop = self.less(dropout_probability, self.layerdrop)
        else:
            to_drop = False

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if self.training and dropout_probability is not None:
                to_drop_layer = self.slice(to_drop, (0, idx), (bs, 1))
                to_drop_layer = ms.ops.expand_dims(to_drop_layer, 2)
                drop_matrix = self.ones_like(hidden_states)
                drop_matrix = self.mul(drop_matrix, to_drop_layer)
            else:
                drop_matrix = self.zeros_like(hidden_states)
            drop_matrix = self.cast(drop_matrix, mstype.bool_)

            hidden_states_ori = hidden_states
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
            )

            hidden_states = self.select(drop_matrix, hidden_states_ori, hidden_states)

        hidden_states = self.cast(hidden_states, self.layernorm_compute_dtype)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.cast(hidden_states, self.compute_dtype)

        return hidden_states, embedding_table


class WhisperPreTrainedModel(PreTrainedModel):
    config_class = WhisperConfig
    base_model_prefix = "model"


class WhisperModel(WhisperPreTrainedModel):
    """whisper model"""
    def __init__(self, config: WhisperConfig):
        super().__init__(config)

        self.encoder = WhisperEncoder(config)
        self.decoder = WhisperDecoder(config)

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def construct(
            self,
            input_features: Optional[ms.Tensor] = None,
            attention_mask: Optional[ms.Tensor] = None,
            decoder_input_ids: Optional[ms.Tensor] = None,
            decoder_attention_mask: Optional[ms.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[ms.Tensor]]] = None,
            decoder_inputs_embeds: Optional[Tuple[ms.Tensor]] = None,
            encoder_dropout_probability=None,
            decoder_dropout_probability=None,
    ):
        """forward"""
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_features,
                attention_mask=attention_mask,
                dropout_probability=encoder_dropout_probability
            )

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs,
            dropout_probability=decoder_dropout_probability,
            inputs_embeds=decoder_inputs_embeds,
        )

        return decoder_outputs, encoder_outputs


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class WhisperForConditionalGeneration(WhisperPreTrainedModel):
    """Whisper Model For ConditionalGeneration"""
    base_model_prefix = "model"

    @lazy_inline
    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.compute_dtype = config.compute_dtype
        self.pad_token_id = config.pad_token_id
        self.slice = P.StridedSlice()
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        if config.is_dynamic:
            self.reshape.add_prim_attr("skip_redistribution", True)
        self.cast = P.Cast()
        self.not_equal = P.NotEqual()
        self.model = WhisperModel(config)
        check_for_nan_in_loss_and_grad = getattr(config, "check_for_nan_in_loss_and_grad", False)
        calculate_per_token_loss = getattr(config, "calculate_per_token_loss", False)
        self.loss = CrossEntropyLoss(parallel_config=config.parallel_config,
                                     check_for_nan_in_loss_and_grad=check_for_nan_in_loss_and_grad,
                                     calculate_per_token_loss=calculate_per_token_loss)
        self.proj_out = LMHead(config.d_model, config.vocab_size, config.compute_dtype)

        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.shard(config.parallel_config)

    def shard(self, parallel_config):
        """set parallel config"""
        dp = parallel_config.data_parallel

        self.slice.shard(((dp, 1),))
        self.proj_out.matmul.shard(((dp, 1), (1, 1)))
        self.not_equal.shard(((dp, 1), ()))

        if parallel_config.pipeline_stage > 1:
            self.proj_out.pipeline_stage = parallel_config.pipeline_stage - 1

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def get_output_embeddings(self):
        return self.proj_out

    def set_output_embeddings(self, new_embeddings):
        self.proj_out = new_embeddings

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def construct(
            self,
            input_features=None,
            decoder_input_ids=None,
            encoder_dropout_probability=None,
            decoder_dropout_probability=None,
            attention_mask=None,
            decoder_attention_mask=None,
            encoder_outputs=None,
            decoder_inputs_embeds=None,
            labels=None,
    ):
        """forward"""
        bsz, seqlen = self.shape(decoder_input_ids)

        if self.training:
            tokens = self.slice(decoder_input_ids, (0, 0), (bsz, seqlen - 1), (1, 1))
        else:
            tokens = decoder_input_ids

        outputs = self.model(
            input_features,
            attention_mask=attention_mask,
            decoder_input_ids=tokens,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            decoder_inputs_embeds=decoder_inputs_embeds,
            encoder_dropout_probability=encoder_dropout_probability,
            decoder_dropout_probability=decoder_dropout_probability,
        )
        hidden_states, embedding_table = outputs[0]
        lm_logits = self.proj_out(hidden_states, embedding_table)

        if labels is None:
            labels = self.slice(decoder_input_ids, (0, 1), (bsz, seqlen), (1, 1))

        input_mask = self.cast(self.not_equal(labels, self.pad_token_id), mstype.float32)

        if not self.training:
            lm_logits = self.cast(lm_logits, mstype.float32)
            return lm_logits, tokens, input_mask

        if lm_logits.ndim > 2:
            lm_logits = self.reshape(lm_logits, (-1, lm_logits.shape[-1]))
        lm_logits = self.cast(lm_logits, mstype.float32)
        labels = self.reshape(labels, (-1,))
        input_mask = self.reshape(input_mask, (-1,))

        loss = self.loss(lm_logits, labels, input_mask)
        return loss
