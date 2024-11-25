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
"""GPT modules."""

import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F

from mindformers.modules.transformer.moe import default_moe_config
from mindformers.modules.transformer import TransformerEncoderLayer
from mindformers.modules.transformer.op_parallel_config import default_dpmp_config


class GPTTransformerDecoderLayer(TransformerEncoderLayer):
    r"""GPT Transformer Decoder Layer.

        Args:
            batch_size(int): The batch size of the input tensor when do increnmental prediction. Should be a positive
                value. When do training or prediction, the argument will not work and the user can just pass None to
                the argument.
            hidden_size(int): The hidden size of the input.
            ffn_hidden_size(int): The hidden size of bottleneck in the feedforward layer.
            num_heads(int): The number of the heads.
            seq_length(int): The input sequence length.
            attention_dropout_rate(float): The dropout rate of the attention scores. Default:0.1.
            hidden_dropout_rate(float): The dropout rate of the final output of the layer. Default:0.1.
            post_layernorm_residual(bool): Do residuals adds before the layernorm. Default False.
            layernorm_compute_type(dtype.Number): The computation type of the layernorm.
                Should be mstype.float32 or mstype.float16. Default mstype.float32.
            softmax_compute_type(dtype.Number): The computation type of the softmax in the attention.
                Should be mstype.float32 or mstype.float16. Default mstype.float32.
            param_init_type(dtype.Number): The parameter initialization type of the module.
                Should be mstype.float32 or mstype.float16. Default mstype.float32.
            hidden_act (str, nn.Cell): The activation of the internal feedforward layer. Supports 'relu',
                'relu6', 'tanh', 'gelu', 'fast_gelu', 'elu', 'sigmoid', 'prelu', 'leakyrelu', 'hswish',
                'hsigmoid', 'logsigmoid' and so on. User can provide custom activition to the argument.
                If user wants to run the net in the parallel mode, the custom activation must also provide
                the `activation_shard` function. Please see the examples of the
                class:`mindformers.modules.transformer.FeedForward`. Default: gelu.
            use_past(bool): Use the past state to compute, used for incremental prediction. For example, if we have two
                words and want to generate the ten more words. We just need to compute the two words' state only once,
                and generate the next word one by one. When use_past is True, there are two steps to run the prediction.
                In the first step, set the is_first_iteration to be True by
                `model.add_flags_recursive(is_first_iteration=True)`, and pass the full inputs. Then, set the
                is_first_iteration to be False by `model.add_flags_recursive(is_first_iteration=False)`.
                At this moment, pass the single step's input tensor, and loop it. Default False.
            moe_config(MoEConfig): The configuration of MoE (Mixture of Expert). Default is an instance of MoEConfig
                with default values. Please see `MoEConfig`.
            parallel_config(OpParallelConfig, MoEParallelConfig): The parallel configure. When MoE is applied,
                MoEParallelConfig is effective, otherwise OpParallelConfig is effective. Default `default_dpmp_config`,
                an instance of `OpParallelConfig` with default args.

        Inputs:
            - **x** (Tensor) - Float Tensor, shape should be [batch_size, seq_length, hidden_size] or
              [batch_size * seq_length, hidden_size], if the use_past is False or is_first_iteration=True. Otherwise,
              should be [batch_size, 1, hidden_size]
            - **input_mask** (Tensor) - Float Tensor, If the use_past is False or is_first_iteration=True,
              the attention mask matrix should ba [batch_size, seq_length, seq_length], or None. None means there will
              be no mask in softmax computation. Otherwise, should be [batch_size, 1, hidden_size]
            - **init_reset** (Tensor) - A bool tensor with shape [1], used to clear the past key parameter and
              past value parameter used in the incremental prediction. Only valid when use_past is True. Default True.
            - **batch_valid_length** (Tensor) - Int32 tensor with shape [batch_size] the past calculated the index.
              Used for incremental prediction when the use_past is True. Default None.

        Outputs:
            Tuple, a tuple contains(`output`, `layer_present`).

            - **output** (Tensor) - The float tensor of the output of the layer with
              shape (batch_size, seq_length, hidden_size) or (batch_size * seq_length, hidden_size), if the use_past is
              False or is_first_iteration=True. Otherwise, it will be (batch_size, 1, hidden_size)

            - **layer_present** (Tuple) - A tuple of the Tensor of the projected key and value vector with
              ((batch_size, num_heads, size_per_head, seq_length),
              (batch_size, num_heads, seq_length, size_per_head)).

        Supported Platforms:
            ``Ascend`` ``GPU``
    """

    def __init__(self,
                 batch_size,
                 hidden_size,
                 ffn_hidden_size,
                 num_heads,
                 seq_length,
                 attention_dropout_rate=0.1,
                 hidden_dropout_rate=0.1,
                 post_layernorm_residual=False,
                 layernorm_compute_type=mstype.float32,
                 softmax_compute_type=mstype.float32,
                 param_init_type=mstype.float32,
                 hidden_act='gelu',
                 use_past=False,
                 moe_config=default_moe_config,
                 parallel_config=default_dpmp_config,
                 use_flash_attention=False,
                 use_prompt_flash_attention=False):
        super(GPTTransformerDecoderLayer, self).__init__(
            batch_size=batch_size,
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            num_heads=num_heads,
            seq_length=seq_length,
            attention_dropout_rate=attention_dropout_rate,
            hidden_dropout_rate=hidden_dropout_rate,
            post_layernorm_residual=post_layernorm_residual,
            layernorm_compute_type=layernorm_compute_type,
            softmax_compute_type=softmax_compute_type,
            param_init_type=param_init_type,
            hidden_act=hidden_act,
            use_past=use_past,
            moe_config=moe_config,
            parallel_config=parallel_config,
            use_flash_attention=use_flash_attention,
            use_prompt_flash_attention=use_prompt_flash_attention
        )

    def construct(self, x, input_mask=None, init_reset=True, batch_valid_length=None):
        """forward process"""
        self._check_input(x, input_mask, init_reset, batch_valid_length)
        x_shape = F.shape(x)
        x = F.reshape(x, (-1, x_shape[-1]))
        if self.post_layernorm_residual:
            input_x = x
        else:
            input_x = self.layernorm1(x)
        input_x = F.cast(input_x, self.dtype)

        # indicate whether reset saved states
        key_reset = None
        value_reset = None

        if self.use_past:
            # reset states, init_reset True for reuse and False for reset
            self.assign(self.key_past, self.mul(self.key_past, F.cast(init_reset, self.dtype)))
            key_reset = self.key_past
            self.assign(self.value_past, self.mul(self.value_past, F.cast(init_reset, self.dtype)))
            value_reset = self.value_past
            # add dependency for desired execution order
            input_x = F.depend(input_x, key_reset)
            input_x = F.depend(input_x, value_reset)

        attention, layer_present = self.attention(input_x, input_x, input_x, input_mask,
                                                  self.key_past, self.value_past, batch_valid_length)
        # For post-layernorm the inputs for residual path are output of self-attention and output of layernorm
        if self.post_layernorm_residual:
            x = self.add(input_x, attention)
        # For pre-layernorm the inputs for residual path are output of self-attention and input of this layer
        else:
            x = self.add(x, attention)

        output_x = self.layernorm2(x)
        output_x = F.cast(output_x, self.dtype)
        aux_loss = None
        if self.use_moe:
            mlp_logit, aux_loss = self.output(output_x)
        else:
            mlp_logit = self.output(output_x)

        value_update = None
        key_update = None
        if self.use_past:
            # current key and value
            key_present, value_present = layer_present
            # update key and value calculated this step
            self.assign(self.key_past, key_present)
            key_update = self.key_past
            self.assign(self.value_past, value_present)
            value_update = self.value_past
            # add dependency for desired execution order
            key_update = F.depend(key_update, key_reset)
            value_update = F.depend(value_update, value_reset)

        # add dependency for desired execution order
        mlp_logit = F.depend(mlp_logit, value_update)
        mlp_logit = F.depend(mlp_logit, key_update)

        # if shape is 3d, we reshape the inputs of the add
        if len(x_shape) == 3:
            output_x = P.Reshape()(output_x, x_shape)
            mlp_logit = P.Reshape()(mlp_logit, x_shape)
            x = P.Reshape()(x, x_shape)

            if self.post_layernorm_residual:
                output = self.add_3d(output_x, mlp_logit)
                output = F.reshape(output, (-1, x_shape[-1]))
                output = self.layernorm1(output)
                output = F.reshape(output, x_shape)
            else:
                output = self.add_3d(x, mlp_logit)
        else:
            if self.post_layernorm_residual:
                output = self.add(output_x, mlp_logit)
                output = self.layernorm1(output)
            else:
                output = self.add(x, mlp_logit)
            output = F.reshape(output, x_shape)

        if self.use_moe:
            return output, aux_loss
        return output
