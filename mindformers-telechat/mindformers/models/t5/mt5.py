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
"""mt5 model."""
import mindspore.common.dtype as mstype
from mindspore import nn, ops
from mindspore.common.tensor import Tensor
from mindspore.ops import functional as F
from mindspore.ops import operations as P

from mindformers.core.loss import CrossEntropyLoss
from mindformers.modules import FeedForward
from mindformers.modules.layers import Linear
from mindformers.modules.transformer.transformer import (
    OpParallelConfig,
    default_dpmp_config,
)

from ...mindformer_book import MindFormerBook
from ...tools.register import MindFormerModuleType, MindFormerRegister
from .t5 import T5Config, T5Model, T5PreTrainedModel

__all__ = ["MT5ForConditionalGeneration"]


class T5FeedFowardGatedGelu(FeedForward):
    """
    T5 feedfoward cell with gated-gelu as hidden act
    """

    def __init__(self,
                 hidden_size,
                 ffn_hidden_size,
                 dropout_rate,
                 hidden_act="gelu",
                 expert_num=1,
                 expert_group_size=None,
                 param_init_type=mstype.float32,
                 parallel_config=default_dpmp_config):
        super(T5FeedFowardGatedGelu, self).__init__(
            hidden_size,
            ffn_hidden_size,
            dropout_rate=dropout_rate,
            hidden_act=hidden_act,
            expert_num=expert_num,
            expert_group_size=expert_group_size,
            param_init_type=param_init_type,
            parallel_config=parallel_config,
        )
        mp = parallel_config.model_parallel
        if expert_num > 1:
            ep = parallel_config.expert_parallel
        else:
            ep = 1
        # ffn use less dp than other ops when use_moe, due to there are ops use dp and ep.
        dp = int(parallel_config.data_parallel / ep)

        del self.mapping
        # replace origin mapping with 2 mappings
        self.mapping_0 = Linear(
            in_channels=hidden_size,
            out_channels=ffn_hidden_size,
            has_bias=False,
            activation=hidden_act,
            transpose_b=False,
            expert_num=expert_num,
            expert_group_size=expert_group_size,
            outer_batch=dp,
            param_init_type=param_init_type,
        )
        self.mapping_0.shard(
            strategy_matmul=((dp, 1), (1, mp)),
            strategy_bias=((dp, mp), (mp,)),
            strategy_activation=((dp, mp),),
        )
        self.mapping_1 = Linear(
            in_channels=hidden_size,
            out_channels=ffn_hidden_size,
            has_bias=False,
            transpose_b=False,
            expert_num=expert_num,
            expert_group_size=expert_group_size,
            outer_batch=dp,
            param_init_type=param_init_type,
        )
        self.mapping_1.shard(
            strategy_matmul=((dp, 1), (1, mp)),
            strategy_bias=((dp, mp), (mp,)),
        )
        self.projection = Linear(
            in_channels=ffn_hidden_size,
            out_channels=hidden_size,
            has_bias=False,
            transpose_b=False,
            expert_num=expert_num,
            expert_group_size=expert_group_size,
            outer_batch=dp,
            param_init_type=param_init_type,
        )
        self.projection.shard(
            strategy_matmul=((dp, mp), (mp, 1)), strategy_bias=((dp, 1), (1,))
        )
        self.mul = ops.Mul()
        self.mul.shard(in_strategy=((dp, 1), (dp, 1)))

    def construct(self, x):
        """construct of T5 FeedFoward-GatedGelu"""
        x = self.cast(x, mstype.float16)
        hidden_gelu = self.mapping_0(x)
        hidden_linear = self.mapping_1(x)
        hidden_states = self.mul(hidden_gelu, hidden_linear)
        if len(F.shape(hidden_states)) == 3:
            hidden_states = self.dropout_3d(hidden_states)
        elif len(F.shape(hidden_states)) == 2:
            hidden_states = self.dropout(hidden_states)
        hidden_states = self.projection(hidden_states)
        return hidden_states


class MT5Head(nn.Cell):
    """MT5 model Head.
    No parameter share between embedding tale and head liner.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 compute_dtype=mstype.float16,
                 parallel_config=default_dpmp_config):
        super(MT5Head, self).__init__()
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel

        self.linear = Linear(
            in_channels=in_channels, out_channels=out_channels, has_bias=False
        )
        self.linear.shard(
            strategy_matmul=((dp, 1), (1, mp)),
            strategy_bias=((dp, mp), (mp,)),
            strategy_activation=((dp, mp),),
        )
        self.dtype = compute_dtype
        self.cast = ops.Cast()

    def construct(self, state, emb_table):
        # to fit t5 head usage
        _ = emb_table
        # output logits over vocabulary [bs, seq_length, vocab_size]
        return self.linear(self.cast(state, self.dtype))


class MT5Model(T5Model):
    """MT5 Model, a variant of T5 model.
    Main changes are: use gated-gelu feedforward, no share parameters
    between embedding tale and head liner.
    here is the paper: <https://arxiv.org/pdf/2010.11934.pdf>

    Args:
        config(T5Config) : The network config of mt5.
    """

    def __init__(self, config):
        super(MT5Model, self).__init__(config)
        for block in self.tfm_encoder.blocks:
            block.output = T5FeedFowardGatedGelu(
                hidden_size=config.hidden_size,
                dropout_rate=config.hidden_dropout_rate,
                ffn_hidden_size=config.d_ff,
                hidden_act=config.hidden_act,
                parallel_config=default_dpmp_config,
            )
        for block in self.tfm_decoder.blocks:
            block.output = T5FeedFowardGatedGelu(
                hidden_size=config.hidden_size,
                dropout_rate=config.hidden_dropout_rate,
                ffn_hidden_size=config.d_ff,
                hidden_act=config.hidden_act,
                parallel_config=default_dpmp_config,
            )
        # no parameter sharing between embedding and classifier layer
        self.projection = MT5Head(
            in_channels=config.hidden_size,
            out_channels=config.vocab_size,
            parallel_config=default_dpmp_config,
        )


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class MT5ForConditionalGeneration(T5PreTrainedModel):
    """
    An MT5 model with the loss added.

    Args:
        config(T5Config) : The network config of mt5.
    """

    _support_list = MindFormerBook.get_model_support_list()["mt5"]

    def __init__(self, config: T5Config):
        super(MT5ForConditionalGeneration, self).__init__(config)
        parallel_config = config.parallel_config
        self.mt5_model = MT5Model(config=config)
        self.vocab_size = config.vocab_size

        check_for_nan_in_loss_and_grad = getattr(config, "check_for_nan_in_loss_and_grad", False)
        calculate_per_token_loss = getattr(config, "calculate_per_token_loss", False)
        self.loss = CrossEntropyLoss(
            parallel_config=OpParallelConfig(
                data_parallel=parallel_config.data_parallel,
                model_parallel=parallel_config.model_parallel,
            ),
            check_for_nan_in_loss_and_grad=check_for_nan_in_loss_and_grad,
            calculate_per_token_loss=calculate_per_token_loss
        )
        self.cast = ops.Cast()
        self.shape = ops.Shape()

        # The value of start and end should get from the tokenizer
        start_token_id = config.start_token_id
        eos_token_id = config.eos_token_id
        self.start_token_id = Tensor(((start_token_id,),), dtype=mstype.int32)  # [CLS]
        self.eos_token_id = Tensor(((eos_token_id,),), dtype=mstype.int32)  # [SEP]
        self.concat = P.Concat(axis=1)
        self.tile = P.Tile()
        self.reshape = ops.Reshape()

        # disable the bias
        for param in self.trainable_params():
            if "bias" in param.name or "beta" in param.name and "relative" not in param.name:
                param.requires_grad = False
        self.load_checkpoint(config)

    def _add_start_to_inputs(self, target_ids):
        """concat the start id to the decoder inputs"""
        start_token = self.tile(self.start_token_id, (F.shape(target_ids)[0], 1))
        decoder_inputs = self.concat((start_token, target_ids))
        return decoder_inputs

    def _add_eos_to_inputs(self, target_ids):
        """concat the eos id to the end of the decoder inputs"""
        eos_token = self.tile(self.eos_token_id, (F.shape(target_ids)[0], 1))
        inputs_with_eos = self.concat((target_ids, eos_token))
        return inputs_with_eos

    def encoder_forward(self, source_ids, source_mask):
        """Execute the encoder forward process"""
        return self.mt5_model.encoder_forward(source_ids, source_mask)

    def construct(self,
                  input_ids,
                  attention_mask,
                  labels=None,
                  decoder_input_ids=None,
                  decoder_attention_mask=None,
                  memory_mask=None,
                  encoder_outputs=None):
        """t5_model network with loss."""
        if decoder_attention_mask is None:
            decoder_attention_mask = F.cast(labels != 0, mstype.float32)

        if decoder_input_ids is None:
            decoder_input_ids = labels
        # replace start token using self.start_token_id
        decoder_input_ids = self._add_start_to_inputs(decoder_input_ids[:, 1:])

        # shape: [bs, seq_len, hidden_size]
        logits = self.mt5_model(
            input_ids,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            memory_mask,
            encoder_cache=encoder_outputs,
        )

        total_loss = None
        if labels is not None:
            # on training
            label_ids = ops.Reshape()(decoder_input_ids[:, 1:], (-1,))
            label_weights = ops.Reshape()(decoder_attention_mask[:, 1:], (-1,))
            logits = logits[:, :-1]  # logits keep shape same with label_ids
            total_loss = self.loss(
                self.reshape(logits, (-1, self.vocab_size)),
                label_ids,
                self.cast(label_weights, mstype.float32),
            )

        # reshape: [bs, seq_len, hidden_size] -> [bs*seq, hidden_size]
        logits = self.reshape(logits, (-1, self.vocab_size))

        if self.training:
            return total_loss

        return logits
