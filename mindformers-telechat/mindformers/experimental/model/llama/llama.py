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
"""mindformers Llama model"""
import numpy as np

from mindspore import Tensor
from mindspore import dtype
from mindspore.ops import operations as P
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation
from mindformers import LlamaConfig
from mindformers.core.context import get_context
from mindformers.experimental.parallel_core.pynative.transformer.enums import AttnMaskType
from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.core.loss.loss import CrossEntropyLoss
from mindformers.experimental.graph.transformer.transformer_config import TransformerConfig
from mindformers.experimental.parallel_core import get_language_model, ParallelLMLogits
from mindformers.experimental.graph.transformer.transformer_config_utils import convert_to_transformer_config
from mindformers.experimental.parallel_core.pynative.transformer.module import Module
from mindformers.tools.logger import logger
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.models.utils import lazy_inline

__all__ = ['LlamaPretrainedModel',
           'LlamaForCausalLM']


class LlamaPretrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and a simple interface for downloading and
    loading pretrained models.
    """
    config_class = LlamaConfig
    base_model_prefix = "llama"


@MindFormerRegister.register(MindFormerModuleType.MODELS, alias='LlamaForCausalLMUnify')
class LlamaForCausalLM(Module, LlamaPretrainedModel):
    """Provide llama training loss or logits through network.

    Args:
        config (TransformerConfig): Configuration for the model.
        num_tokentypes (int): Number of token types.
        parallel_output (bool): Whether to output in parallel.
        pre_process (bool): Whether to add pre-process.
        post_process (bool): Whether to add post-process.
    """
    @lazy_inline
    def __init__(self,
                 config: LlamaConfig,
                 num_tokentypes: int = 0,
                 parallel_output: bool = True,
                 pre_process: bool = True,
                 post_process: bool = True,
                 **kwargs
                 ):
        logger.info(f'Init LlamaForCausalLM in mode: {get_context("mode")} .')
        transformer_config = TransformerConfig()
        convert_to_transformer_config(config, transformer_config)
        super().__init__(transformer_config, False, config, auto_prefix=True, **kwargs)
        self.config = config
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = transformer_config.fp16_lm_cross_entropy
        self.compute_dtype = transformer_config.compute_dtype
        self.pad_token_id = transformer_config.pad_token_id
        self.parallel_output = parallel_output
        self.mode = get_context("mode")
        self.default_attn_mask = Tensor(
            np.triu(np.ones((transformer_config.seq_length, transformer_config.seq_length), dtype=np.uint8),
                    k=1),
            dtype.uint8
        )

        self.language_model, _ = get_language_model(
            config=transformer_config,
            num_tokentypes=num_tokentypes,
            add_pooler=False,
            encoder_attn_mask_type=AttnMaskType.causal,
            decoder_attn_mask_type=None,
            pre_process=self.pre_process,
            post_process=self.post_process)

        if self.post_process:
            self.parallel_lm_logits = ParallelLMLogits(config=transformer_config, bias=False)

            # fft1374 Awaiting the implementation of the specific `loss` function
            transformer_config.model_parallel = transformer_config.tensor_parallel
            self.loss = CrossEntropyLoss(parallel_config=transformer_config)

        self.ignore_token_id = transformer_config.ignore_token_id
        self.slice = P.StridedSlice()
        self.not_equal = P.NotEqual()
        self.reshape = P.Reshape()
        self.mul = P.Mul()
        if _get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation():
            self.sharding_propagation(transformer_config)
        else:
            self.shard(transformer_config)

    def construct(self,
                  input_ids: Tensor,
                  position_ids: Tensor = None,
                  attention_mask: Tensor = None,
                  retriever_input_ids: Tensor = None,
                  retriever_position_ids: Tensor = None,
                  retriever_attn_mask: Tensor = None,
                  labels: Tensor = None,
                  tokentype_ids=None,
                  inference_params=None,
                  prefix_keys_values=None,
                  input_embeds: Tensor = None,
                  loss_mask: Tensor = None
                  ) -> Tensor:
        """Forward of llama model.

        Args:
            input_ids (Tensor): Input token ids.
            position_ids (Tensor): Position ids.
            attention_mask (Tensor): Attention mask.
            retriever_input_ids (Tensor): Input token ids for retriever.
            retriever_position_ids (Tensor): Position ids for retriever.
            retriever_attn_mask (Tensor): Attention mask for retriever.
            labels (Tensor): Labels.
            tokentype_ids: Token type ids.
            inference_params: Inference parameters.
            prefix_keys_values: Prefix keys and values.
            input_embeds(Tensor): Reserved param, not used.
            loss_mask (Tensor): Loss mask.

        Returns:
            output (Tensor): Output logits or loss.
        """
        if inference_params is not None:
            raise ValueError("LlamaModel does not need to support inference_params in training.")
        if tokentype_ids is not None:
            raise ValueError("LlamaModel does not support tokentype_ids for now.")
        if input_embeds is not None:
            raise ValueError("LlamaModel does not support input_embeds for now.")
        tokens, labels, attention_mask, loss_mask = self._preprocess_input_labels_and_masks(input_ids,
                                                                                            labels, attention_mask)
        lm_output = self.language_model(
            tokens,
            position_ids,
            attention_mask,
            retriever_input_ids=retriever_input_ids,
            retriever_position_ids=retriever_position_ids,
            retriever_attn_mask=retriever_attn_mask,
            inference_params=inference_params,
            prefix_keys_values=prefix_keys_values)

        if self.post_process:
            return self.post_language_model_processing(lm_output, labels, self.language_model.output_layer.weight,
                                                       self.parallel_output, self.fp16_lm_cross_entropy, loss_mask)
        return lm_output

    def post_language_model_processing(self,
                                       lm_output: Tensor,
                                       labels: Tensor,
                                       logit_weights: Tensor,
                                       parallel_output: bool,
                                       fp16_lm_cross_entropy: bool,
                                       loss_mask: Tensor
                                       ) -> Tensor:
        """Post-processing of language model output.

        Args:
            lm_output (Tensor): Language model output.
            labels (Tensor): Labels.
            logit_weights (Tensor): Logit weights.
            parallel_output (bool): Whether to output in parallel.
            fp16_lm_cross_entropy (bool): Whether to use fp16 for loss computation.
            loss_mask (Tensor): Loss mask.

        Returns:
            output (Tensor): Output loss.
        """
        if fp16_lm_cross_entropy:
            raise ValueError("LlamaModel does not need to support fp16_lm_cross_entropy.")
        # Output. Format [s b h]
        # pylint: disable=E1102
        output = self.parallel_lm_logits(lm_output, logit_weights, parallel_output)

        if labels is None:
            return output.contiguous()

        if output.ndim > 2:
            output = self.reshape(output, (-1, output.shape[-1]))
        output = self.cast(output, dtype.float32)
        return self.loss(output, labels, loss_mask)

    def set_input_tensor(self, input_tensor):
        """Set input tensor to model"""
        self.language_model.set_input_tensor(input_tensor)

    def set_dynamic_inputs(self, **kwargs):
        dynamic_input_ids = Tensor(shape=[None, None], dtype=dtype.int32)
        dynamic_input_position = Tensor(shape=[None], dtype=dtype.int32)
        dynamic_init_reset = Tensor([False], dtype.bool_)
        dynamic_batch_valid_length = Tensor(shape=[None, None], dtype=dtype.int32)
        dynamic_block_tables = Tensor(shape=[None, None], dtype=dtype.int32)
        dynamic_slot_mapping = Tensor(shape=[None], dtype=dtype.int32)
        have_prefix_keys_values = getattr(kwargs, "have_prefix_keys_values", False)
        if have_prefix_keys_values:
            dynamic_prefix_keys_values = Tensor(shape=[2, None, None, None, None], dtype=dtype.float16)
            self.set_inputs(dynamic_input_ids, None, dynamic_input_position, None, None, None, dynamic_init_reset,
                            dynamic_batch_valid_length, None, None, dynamic_block_tables,
                            dynamic_slot_mapping, dynamic_prefix_keys_values)
        else:
            self.set_inputs(dynamic_input_ids, None, dynamic_input_position, None, None, None, dynamic_init_reset,
                            dynamic_batch_valid_length, None, None, dynamic_block_tables,
                            dynamic_slot_mapping, None)
        logger.info("Set dynamic input for llama.")

    def _preprocess_input_labels_and_masks(self, input_ids: Tensor, labels: Tensor = None,
                                           attention_mask: Tensor = None) -> (Tensor, Tensor, Tensor, Tensor):
        """Preprocess input_ids and generate labels and masks if they are None.

        Args:
            labels (Tensor): Labels.
            attention_mask (Tensor): Attention mask.

        Returns:
            tokens (Tensor): Processed tokens if in training.
            labels (Tensor): Labels if input is none.
            attention_mask (Tensor): Attention mask if input is none.
            loss_mask (Tensor): Loss mask.
        """
        bs, seq_len = input_ids.shape
        if self.training:
            tokens = self.slice(input_ids, (0, 0), (bs, seq_len - 1), (1, 1))
        else:
            tokens = input_ids
        loss_mask = self.cast(self.not_equal(tokens, self.pad_token_id), dtype.float32)
        if labels is None:
            labels = self.slice(input_ids, (0, 1), (bs, seq_len), (1, 1))
        else:
            if labels.ndim > 1:
                if self.training:
                    labels = self.slice(labels, (0, 1), (bs, seq_len), (1, 1))
                label_mask = self.cast(self.not_equal(labels, self.ignore_token_id), dtype.float32)
                loss_mask = self.mul(loss_mask, label_mask)
        loss_mask = self.reshape(loss_mask, (-1,))
        labels = self.reshape(labels, (-1,))
        if attention_mask is None and self.mode == 1:
            attention_mask = self.default_attn_mask
        return tokens, labels, attention_mask, loss_mask

    def shard(self, config: TransformerConfig):
        dp = config.data_parallel
        slice_in_strategy = ((dp, 1),)
        self.slice.shard(in_strategy=slice_in_strategy)
        not_equal_in_strategy = ((dp, 1), ())
        self.not_equal.shard(in_strategy=not_equal_in_strategy)
        mul_in_strategy = ((dp, 1), (dp, 1))
        self.mul.shard(in_strategy=mul_in_strategy)

    def sharding_propagation(self, config: TransformerConfig):
        pass
