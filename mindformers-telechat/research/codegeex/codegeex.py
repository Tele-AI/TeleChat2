# Copyright 2021 Huawei Technologies Co., Ltd
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
"""CodeGeex training wrapper"""

import numpy as np
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.ops import functional as F
from mindspore.ops import operations as P


from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.core.loss import CrossEntropyLoss
from mindformers.models.pangualpha import PanguAlphaHeadModel


__all__ = ['CodeGeexHeadModel']


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class CodeGeexHeadModel(PanguAlphaHeadModel):
    """
    CodeGeex training loss for generation.
    Args:
        config(CodeGeexConfig)
    Returns:
        output: Tensor, the loss of the network
    """

    def __init__(self, config):
        super(CodeGeexHeadModel, self).__init__(config)
        self.pad_token = Tensor(config.pad_token_id)
        dp = config.parallel_config.data_parallel
        self.eod_token = config.eod_token
        check_for_nan_in_loss_and_grad = getattr(config, "check_for_nan_in_loss_and_grad", False)
        self.loss = CrossEntropyLoss(config.parallel_config.dp_mp_config,
                                     check_for_nan_in_loss_and_grad=check_for_nan_in_loss_and_grad)

        self.slice = P.StridedSlice().shard(((dp, 1),))
        self.not_equal = P.NotEqual().shard(((dp, 1), ()))
        self.slice2 = P.StridedSlice().shard(((dp, 1, 1),))
        self.eod_mask_loss = config.eod_mask_loss
        if config.single_checkpoint_name_or_path != "":
            config.checkpoint_name_or_path = config.single_checkpoint_name_or_path
            self.load_checkpoint(config)

    # pylint: disable=W0613
    def construct(self, input_ids, input_position=None, attention_mask=None, position_ids=None,
                  input_embeds=None, labels=None, init_reset=True, batch_valid_length=None):
        r"""Forward process of the codegeex model"""
        batch_size, seq_length = input_ids.shape

        if self.training:
            seq_length = seq_length - 1
            tokens = self.slice(input_ids, (0, 0),
                                (batch_size, seq_length), (1, 1))
            input_position = self.slice(
                input_position, (0, 0), (batch_size, seq_length), (1, 1))
            attention_mask = self.cast(attention_mask, mstype.float16)
            input_mask = F.ones_like(tokens)
            if self.eod_mask_loss:
                input_mask = F.cast(self.not_equal(
                    tokens, self.eod_token), mstype.float32)
        else:
            tokens = input_ids
            input_position = F.tuple_to_array(F.make_range(seq_length))
            input_position = P.Tile()(input_position, (batch_size, 1))
            input_mask = F.cast(F.not_equal(
                tokens, self.pad_token), mstype.float32)
            if self.is_first_iteration is False:
                attention_mask = P.Tile()(
                    Tensor(np.ones((1, 1, 2048)), mstype.float32), (batch_size, 1, 1))
            else:
                attention_mask = self.get_attention_mask(input_mask)
            batch_valid_length -= 1
        logits, vocab_table = self.backbone(
            tokens, input_position, attention_mask, init_reset, batch_valid_length)
        logits = self.head(logits, vocab_table)
        if not self.training:
            return (logits,)
        # Get label corresponding to input tokens
        labels = self.slice(input_ids, (0, 1),
                            (batch_size, seq_length + 1), (1, 1))
        labels = P.Reshape()(labels, (-1,))
        input_mask = P.Reshape()(input_mask, (-1,))
        output = self.loss(logits, labels, input_mask)
        return output
