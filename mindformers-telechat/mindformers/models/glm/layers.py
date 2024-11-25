"""base transformer layer."""
import numpy as np

from mindspore import Parameter, Tensor
from mindspore import dtype as mstype
from mindspore import nn
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore import ops

from mindformers.modules.layers import LayerNorm, Linear
from mindformers.version_control import get_dropout
from mindformers.models.glm.attention import RotaryEmbeddingFP32SoftmaxSelfAttention


class GEGLU(nn.Cell):
    """GEGLU activation"""

    def __init__(self, parallel_config):
        super(GEGLU, self).__init__()
        self.split = ops.Split(output_num=2, axis=-1)
        self.activation_fn = P.GeLU()
        self.parallel_config = parallel_config

    def construct(self, x):
        x1, x2 = self.split(x)
        return x1 * self.activation_fn(x2)


class MLPWithGEGLU(nn.Cell):
    """MLP layer with GEGLU"""
    def __init__(self,
                 hidden_size,
                 output_dropout_prob,
                 inner_hidden_size=None,
                 layer_id=None,
                 bias=True,
                 activation_func='GELU',
                 params_dtype=mstype.float32,
                 compute_dtype=mstype.float16,
                 parallel_config=None):
        super(MLPWithGEGLU, self).__init__()
        self.layer_id = layer_id
        # Project to 4h.
        self.hidden_size = hidden_size

        if inner_hidden_size is None:
            inner_hidden_size = 4 * hidden_size

        self.inner_hidden_size = inner_hidden_size
        if activation_func == 'GEGLU':
            self.activation_func = GEGLU(parallel_config)
            h_to_4h_output_channel = 2 * self.inner_hidden_size
        elif activation_func == 'GELU':
            self.activation_func = P.GeLU()
            self.activation_func.shard(((parallel_config.data_parallel, 1, parallel_config.model_parallel),))
            h_to_4h_output_channel = self.inner_hidden_size

        self.dense_h_to_4h = Linear(
            self.hidden_size,
            h_to_4h_output_channel,
            has_bias=bias,
            transpose_b=True,
            param_init_type=params_dtype,
            compute_dtype=compute_dtype,
        )
        self.dense_h_to_4h.shard(
            strategy_matmul=((parallel_config.data_parallel, 1), (parallel_config.model_parallel, 1)),
            strategy_bias=((parallel_config.data_parallel, parallel_config.model_parallel),
                           (parallel_config.model_parallel,)))

        # Project back to h.
        self.dense_4h_to_h = Linear(
            self.inner_hidden_size,
            self.hidden_size,
            has_bias=bias,
            param_init_type=params_dtype,
            compute_dtype=compute_dtype,
        )
        self.dense_4h_to_h.shard(
            strategy_matmul=((parallel_config.data_parallel, parallel_config.model_parallel),
                             (1, parallel_config.model_parallel)),
            strategy_bias=((parallel_config.data_parallel, 1), (1,))
            )

        self.dropout = get_dropout(output_dropout_prob)
        self.dropout.dropout.shard(((parallel_config.data_parallel, parallel_config.model_parallel),))

    def mlp_forward(self, hidden_states):
        """mlp forward."""
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        output = self.dense_4h_to_h(intermediate_parallel)
        return output

    def construct(self, hidden_states):
        output = self.mlp_forward(hidden_states)

        if self.training:
            output = self.dropout(output)
        return output


class DeepNormWithGLULayer(nn.Cell):
    """
    GLM base layer

    Args:
        num_layers (int): Number of layers.
        hidden_size (int): Hidden layer size.
        num_attention_heads (int): Number of attention heads.
        batch_size (int): Batch size.
        attention_dropout_prob (float, [0, 1.0]): Attention layer dropout probability.
        output_dropout_prob (float, [0, 1.0]): Output dropout probability.
        layernorm_epsilon (float): Layernorm epsilon.
        layer_id (int): Layer id.
        max_seq_len (int): Max sequence length.
        inner_hidden_size (optional): Inner hidden layer size. Default: None.
        hidden_size_per_attention_head (optional): Default: None.
        layernorm_order (str, optional): Which order to use layernorm. Default: 'pre'.
        layernorm (optional): Layernorm function. Default: LayerNorm.
        use_bias (bool, optional): Use bias or not. Default: True.
        activation_func (str, optional, 'GEGLU'/'GELU'): Choose activation function. Default: 'GEGLU'.
        position_encoding_2d (bool, optional): Use 2d position encoding or not. Default: True.
        params_dtype (ms.dtype, optional): Parameter data type. Default: mstype.float32.
        layernorm_dtype (ms.dtype, optional): Calculate layernorm data type. Default: mstype.float32.
        softmax_dtype (ms.dtype, optional): Calculate softmax data type. Default: mstype.float32.
        compute_dtype (ms.dtype, optional): Other compute data type. Default: mstype.float16.
        parallel_config (optional): Operator parallel strategy, Default: None.
        use_past (bool, optional): Use infer cache or not. Default: False.
    """

    def __init__(self,
                 num_layers,
                 hidden_size,
                 num_attention_heads,
                 batch_size,
                 attention_dropout_prob,
                 output_dropout_prob,
                 layernorm_epsilon,
                 layer_id,
                 max_seq_len=512,
                 inner_hidden_size=None,
                 hidden_size_per_attention_head=None,
                 layernorm_order='pre',
                 layernorm=LayerNorm,
                 use_bias=True,
                 activation_func='GEGLU',
                 position_encoding_2d=True,
                 params_dtype=mstype.float32,
                 layernorm_dtype=mstype.float32,
                 softmax_dtype=mstype.float32,
                 compute_dtype=mstype.float16,
                 parallel_config=None,
                 use_past=False):
        super(DeepNormWithGLULayer, self).__init__()
        # Set output layer initialization if not provided.
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.layer_id = layer_id
        if inner_hidden_size is None:
            inner_hidden_size = 4 * hidden_size * 2 // 3
        self.inner_hidden_size = inner_hidden_size
        self.position_encoding_2d = position_encoding_2d
        self.layernorm_order = layernorm_order
        self.use_past = use_past

        self.params_dtype = params_dtype
        self.layernorm_dtype = layernorm_dtype
        self.softmax_dtype = softmax_dtype
        self.compute_dtype = compute_dtype

        # Layernorm on the input data.
        self.input_layernorm = layernorm(hidden_size, param_init_type=layernorm_dtype, eps=layernorm_epsilon)
        self.input_layernorm.shard(((parallel_config.data_parallel, 1, 1),))
        self.input_layernorm.set_comm_fusion(parallel_config.gradient_aggregation_group)

        # Self attention.
        self.attention = RotaryEmbeddingFP32SoftmaxSelfAttention(
            hidden_size,
            batch_size,
            num_attention_heads,
            parallel_config,
            attention_dropout_prob,
            output_dropout_prob,
            layer_id,
            hidden_size_per_attention_head=hidden_size_per_attention_head,
            position_encoding_2d=self.position_encoding_2d,
            bias=use_bias,
            params_dtype=params_dtype,
            softmax_dtype=softmax_dtype,
            compute_dtype=compute_dtype,
            max_seq_len=max_seq_len,
            use_past=use_past,
        )

        # Layernorm on the input data.
        self.post_attention_layernorm = layernorm(hidden_size, param_init_type=layernorm_dtype, eps=layernorm_epsilon)
        self.post_attention_layernorm.shard(((parallel_config.data_parallel, 1, 1),))
        self.post_attention_layernorm.set_comm_fusion(parallel_config.gradient_aggregation_group)

        if self.layernorm_order == 'sandwich':
            self.third_layernorm = layernorm(hidden_size, param_init_type=layernorm_dtype, eps=layernorm_epsilon)
            self.fourth_layernorm = layernorm(hidden_size, param_init_type=layernorm_dtype, eps=layernorm_epsilon)

        # MLP
        self.mlp = MLPWithGEGLU(
            hidden_size,
            output_dropout_prob,
            inner_hidden_size=inner_hidden_size,
            bias=use_bias,
            layer_id=layer_id,
            activation_func=activation_func,
            params_dtype=params_dtype,
            parallel_config=parallel_config
        )

        self.key_past = None
        self.value_past = None
        if use_past:
            # operator used for state reuse
            self.reducesum = P.ReduceSum().shard(((1, 1, 1, 1),))
            self.not_equal = P.NotEqual().shard(((1, 1, 1, 1), ()))
            self.slice = P.StridedSlice().shard(((1, 1, 1, 1),))
            size_per_head = int(hidden_size / num_attention_heads)
            self.kv_shape = (batch_size, num_attention_heads, size_per_head, max_seq_len)
            # parameters saving key and value states
            self.key_past = Parameter(Tensor(np.zeros(shape=self.kv_shape), self.params_dtype), name="key_past")
            self.value_past = Parameter(Tensor(np.zeros(shape=self.kv_shape), self.params_dtype), name="value_past")
            self.tile = P.Tile().shard(((1, 1),))
            self.mul = P.Mul().shard(((1, 1, 1, 1), ()))
            self.assign = P.Assign().shard(((1, 1, 1, 1), (1, 1, 1, 1)))
        self.mul = P.Mul()
        self.mul.shard(((parallel_config.data_parallel, 1, 1), ()))
        self.mul_4 = P.Mul()
        self.mul_4.shard(((parallel_config.data_parallel, 1, 1, 1), (parallel_config.data_parallel,)))

    def layer_forward(self, hidden_states, mask, position_ids, init_reset=True, batch_valid_length=None):
        """
            hidden_states: [seq_len, batch, hidden_size] with (1, dp, 1)
            mask: [(1, 1), seq_len, seq_len]
        Inputs:
            hidden_states (Tensor): Hidden layer output.
            mask (Tensor): Used when batching sequences together.
            position_ids (Tensor): Used to identify each token's position in the list of tokens.
            init_reset (bool, optional): Default: True.
            batch_valid_length (Tensor, optional): Default: None.

        Return:
            output (Tensor): Layer output.
        """
        # Layer norm at the beginning of the transformer layer.
        attention_input = self.input_layernorm(hidden_states)

        # indicate whether reset saved states
        key_reset = None
        value_reset = None

        if self.use_past:
            # reset states, init_reset True for reuse and False for reset
            key_reset = self.assign(self.key_past, self.mul_4(self.key_past, F.cast(init_reset, self.params_dtype)))
            value_reset = self.assign(self.value_past,
                                      self.mul_4(self.value_past, F.cast(init_reset, self.params_dtype)))
            # add dependency for desired execution order
            attention_input = F.depend(attention_input, key_reset)
            attention_input = F.depend(attention_input, value_reset)
        attention_output, layer_present = self.attention(attention_input, mask, position_ids, self.layer_id,
                                                         self.key_past, self.value_past, batch_valid_length)

        # Residual connection.
        alpha = (2 * self.num_layers) ** 0.5
        hidden_states = self.mul(attention_input, alpha) + attention_output

        mlp_input = self.post_attention_layernorm(hidden_states)

        # MLP.
        mlp_output = self.mlp(mlp_input)

        value_update = None
        key_update = None
        if self.use_past:
            # current key and value
            key_present, value_present = layer_present
            # update key and value calculated this step
            key_update = self.assign(self.key_past, key_present)
            value_update = self.assign(self.value_past, value_present)
            # add dependency for desired execution order
            key_update = F.depend(key_update, key_reset)
            value_update = F.depend(value_update, value_reset)

        # add dependency for desired execution order
        mlp_output = F.depend(mlp_output, value_update)
        mlp_output = F.depend(mlp_output, key_update)
        # Second residual connection.
        output = mlp_input * alpha + mlp_output

        return output

    def construct(self, hidden_states, mask, position_ids, init_reset=True, batch_valid_length=None):
        '''
            hidden_states: [batch, seq_len, hidden_size]
            mask: [(1, 1), seq_len, seq_len]
        '''
        return self.layer_forward(hidden_states, mask, position_ids, init_reset, batch_valid_length)
