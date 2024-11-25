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

"""bloom reward model"""
import copy
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F

from mindformers.modules.layers import Linear
from mindformers.core.loss import CompareLoss
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.tools.logger import logger
from .bloom_config import BloomConfig
from .bloom import BloomModel, BloomPreTrainedModel

__all__ = ['BloomRewardModel', 'VHead']


class VHead(nn.Cell):
    r"""Head for Bloom to get the logits of each token in the vocab."""
    def __init__(self, config=None):
        super().__init__()
        dp = config.parallel_config.data_parallel
        mp = 1
        self.vhead = Linear(in_channels=config.hidden_size,
                            out_channels=1,
                            has_bias=False).to_float(mstype.float16)
        self.vhead.shard(strategy_matmul=((dp, mp), (mp, 1)))
        self.vhead.pipeline_stage = config.parallel_config.pipeline_stage - 1

    def construct(self, output_states):
        """
        construct function for vhead
        """
        return self.vhead(output_states)


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class BloomRewardModel(BloomPreTrainedModel):
    """Provide bloom reward model training loss or logits through network.
        Args:
            config (BloomConfig): The config of BloomModel.

        Returns:
            Tensor, the loss or logits of the network.
    """

    def __init__(self, config=None):
        config = config if config is not None else BloomConfig()
        super(BloomRewardModel, self).__init__(config, auto_prefix=False)
        mp = config.parallel_config.model_parallel
        self.eos_token_id = self.config.eos_token_id
        self.seq_length = config.seq_length
        self.eos_token = self.config.eos_token
        parallel_config = self.config.parallel_config
        self.stridedslice = P.StridedSlice().shard(((parallel_config.data_parallel, 1),))
        self.not_equal = P.NotEqual().shard(((parallel_config.data_parallel, 1), ()))
        self.transformer = BloomModel(self.config)
        self.vhead = VHead(self.config)
        if parallel_config.pipeline_stage > 1:
            self.vhead.pipeline_stage = parallel_config.pipeline_stage - 1
            self.transformer.embedding.word_embedding.embedding_table.add_pipeline_stage(self.vhead.pipeline_stage)

        vocab_size = config.vocab_size
        loss_parallel_config = copy.deepcopy(parallel_config)
        if vocab_size % mp != 0:
            logger.warning("The vocab size of Bloom Loss is: %s, it is not divide by model_parallel: %s",
                           vocab_size, mp)
            logger.warning("Now, the model_parallel num of Bloom Loss will be changed: mp = 1")
            loss_parallel_config.model_parallel = 1

        self.loss = CompareLoss(config=loss_parallel_config)
        self.load_checkpoint(config)

    def construct(self,
                  input_ids,
                  position_id=None,
                  attention_mask=None,
                  loss_mask=None,
                  end_ind=None):
        """
        construct function for reward model
        """
        _ = position_id
        if attention_mask is None:
            attention_mask = self.not_equal(input_ids, self.eos_token_id).astype(mstype.float32)
        output_states, _ = self.transformer(input_ids, attention_mask)
        # [bs, seq, hidden_size]
        logits = self.vhead(output_states)
        # [bs, seq, 1]
        logits = logits.squeeze(-1)
        logits = F.reshape(logits, (-1, self.seq_length))
        loss, chosen_end_scores, reject_end_scores = self.loss(logits, loss_mask, end_ind)
        if self.training:
            return loss
        return loss, chosen_end_scores, reject_end_scores
