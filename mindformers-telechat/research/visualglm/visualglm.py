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
"""visualglm model implementation."""

import os
from collections import OrderedDict

import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import ops
from mindspore.nn import CrossEntropyLoss
from mindspore.ops import operations as P
from mindpet.graph import freeze_delta

from mindformers import MindFormerBook, LoraAdapter
from mindformers.modules.layers import Linear
from mindformers.tools.logger import logger
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType

from visualglm_base import VisualGLMBase
from visualglm_config import VisualGLMConfig
from visualglm_text_generation_pipeline import register_pipeline_task

__all__ = ['VisualGLMModel', 'VisualGLMImageToTextGeneration', 'VisualglmWithLora']


def register_trainer_task():
    """ register trainer task for visualglm """
    cur_path = os.path.dirname(os.path.realpath(__file__))
    MindFormerBook.get_trainer_support_task_list()['text_generation'] = OrderedDict([
        ("visualglm_6b", os.path.join(
            cur_path, "run_visualglm_lora.yaml"))])


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class VisualGLMModel(VisualGLMBase):
    """
    visualglm with llm model.
    """

    def __init__(self, config: VisualGLMConfig, **kwargs):
        super(VisualGLMModel, self).__init__(config, **kwargs)
        self.batch_size = None
        self.config = config if config is not None else VisualGLMConfig()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder()
        if config.freeze_vision:
            logger.info("freeze vision encoder")
            for param in self.visual_encoder.trainable_params():
                param.requires_grad = False
            self.visual_encoder.set_train(False)

        self.qformer, self.query_tokens = self.init_qformer()

        self.qformer.cls = None
        self.qformer.bert.embeddings.word_embeddings = None
        self.qformer.bert.embeddings.position_embeddings = None
        for layer in self.qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        if config.text_config.seq_length < config.max_txt_len + config.qformer_config.query_length:
            raise ValueError(
                f"seq_length should be greater than sum of max_text_len and num_query_token "
                f"{config.max_txt_len + config.qformer_config.query_length:d}, but got "
                f"{config.text_config.seq_length:d}")

        self.llm_model = self.init_llm()

        if config.freeze_text:
            logger.info("freeze llm model")
            for param in self.llm_model.trainable_params():
                param.requires_grad = False
            self.llm_model.set_train(False)

        dp = config.parallel_config.data_parallel

        micro_batch_interleave_num = config.micro_batch_interleave_num

        self.init_batch_size(dp, micro_batch_interleave_num)

        self.llm_proj = Linear(in_channels=self.config.qformer_config.hidden_size,
                               out_channels=self.config.text_config.hidden_size,
                               param_init_type=config.dtype,
                               compute_dtype=config.compute_dtype)

        pp = config.parallel_config.pipeline_stage
        if pp > 1:
            self.visual_encoder.pipeline_stage = 0
            self.qformer.pipeline_stage = 0
            self.llm_proj.pipeline_stage = 0

        if config.checkpoint_name_or_path:
            logger.info(
                "load blip2 first stage pretrained model for vision model and qformer, checkpoint_name_or_path: "
                "%s. pretrained llm model: %s", config.checkpoint_name_or_path,
                config.text_config.checkpoint_name_or_path)
            self.load_checkpoint(config)
        else:
            self.init_checkpoint(config)

        self.eos_token_id = config.text_config.eos_token_id
        self.pad_token_id = config.text_config.pad_token_id
        self.ignore_token_id = config.text_config.ignore_token_id
        self.max_txt_len = config.max_txt_len
        self.prompt = config.prompt
        self.prompt_length = config.prompt_length

        self.broadcast_to = P.BroadcastTo((self.batch_size,
                                           self.config.qformer_config.query_length,
                                           self.config.qformer_config.hidden_size)).shard(((1, 1, 1),))
        self.fill = P.Fill().shard(((dp, 1),))
        self.masked_fill = P.MaskedFill().shard(((dp, 1), ()))
        self.ones = P.Ones().shard(((dp, 1),))
        self.concat_2d = P.Concat(axis=1).shard(((dp, 1), (dp, 1)))
        self.concat_3d = P.Concat(axis=1).shard(((dp, 1, 1), (dp, 1, 1)))
        self.not_equal = P.NotEqual().shard(((dp, 1), ()))
        self.slice = P.StridedSlice().shard(((dp, 1),))
        self.cast = P.Cast()
        self.loss_fct = CrossEntropyLoss(ignore_index=-100)

        register_pipeline_task()

    def init_batch_size(self, dp, micro_batch_interleave_num):
        """
        init batch size
        :param dp: data parallel config
        :param micro_batch_interleave_num: micro batch interleave num
        """
        batch_size = self.config.batch_size
        parallel_mode = ms.get_auto_parallel_context("parallel_mode")
        if parallel_mode in ["semi_auto_parallel", "auto_parallel"]:
            full_batch = ms.get_auto_parallel_context("full_batch")
            if full_batch:
                self.batch_size = batch_size * dp * micro_batch_interleave_num
            else:
                card_num = int(os.getenv('RANK_SIZE', '1'))
                self.batch_size = int(card_num * batch_size / micro_batch_interleave_num)
        else:
            self.batch_size = batch_size

    @staticmethod
    def init_checkpoint(config):
        """ init checkpoint """
        if config.vision_config.checkpoint_name_or_path:
            vision_checkpoint = config.vision_config.checkpoint_name_or_path
        else:
            vision_checkpoint = 'not configured'
        if config.text_config.checkpoint_name_or_path:
            text_checkpoint = config.text_config.checkpoint_name_or_path
        else:
            text_checkpoint = 'not configured'
        if config.qformer_config.checkpoint_name_or_path:
            qformer_checkpoint = config.qformer_config.checkpoint_name_or_path
        else:
            qformer_checkpoint = 'not configured'
        logger.info("training blip2 second stage, pretrained vision model: %s, pretrained llm model: %s, "
                    "pretrained qformer: %s", vision_checkpoint, text_checkpoint, qformer_checkpoint)

    def forward_qformer_and_proj(self, image: ms.Tensor):
        """forward the image tensor to the qformer, then project the output to adapt the dimension"""
        image_embeds = self.visual_encoder(image)
        image_embeds = self.ln_vision(image_embeds)  # [batch_size, vit_seq_length, vit_encoder_hidden_width]
        image_atts = self.ones(image_embeds.shape[:-1], mstype.float32)  # [batch_size, vit_seq_length]

        query_tokens = self.broadcast_to(self.query_tokens)  # [batch_size, query_size, qformer_hidden_size]

        query_output = self.qformer.bert(query_embeds=query_tokens,
                                         encoder_hidden_states=image_embeds,
                                         encoder_attention_mask=image_atts,
                                         use_cache=True)

        # [batch_size, query_size, qformer_hidden_size] -> [batch_size, query_size, llm_hidden_size]
        return self.llm_proj(query_output[0])


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class VisualGLMImageToTextGeneration(VisualGLMModel):
    """
    VisualGLMImageToTextGeneration rely on Blip2Llm, used for image to text genearation.

    Args:
        config (VisualGLMConfig): The config of Blip2ImageToTextGeneration.

    Examples:
        >>> from mindformers import Blip2ImageToTextGeneration
        >>> model = Blip2ImageToTextGeneration.from_pretrained('itt_blip2_stage2_vit_g_llama_7b')
        >>> type(model)
        <class 'mindformers.models.blip2.blip2_llama.Blip2ImageToTextGeneration'>
    """

    def __init__(self, config: VisualGLMConfig, **kwargs):
        super(VisualGLMImageToTextGeneration, self).__init__(config, **kwargs)

        self.llm_model.set_train(False)
        self.one_prefix = ops.Ones()
        self.expand_dims = P.ExpandDims()

        self.query_length = self.config.qformer_config.query_length

    def construct(self, image: ms.Tensor, pre_input_ids: ms.Tensor, post_input_ids: ms.Tensor):
        """ VisualGLMImageToTextGeneration model network """
        if len(pre_input_ids.shape) == 1:
            pre_input_ids = self.expand_dims(pre_input_ids, 0)

        if len(post_input_ids.shape) == 1:
            post_input_ids = self.expand_dims(post_input_ids, 0)

        batch_size = image.shape[0]
        prefix_ones = self.one_prefix((batch_size, self.query_length), mstype.int32)

        extend_text_input_ids = self.concat_2d([pre_input_ids, prefix_ones, post_input_ids])
        projected_qformer_output = self.forward_qformer_and_proj(image)
        return extend_text_input_ids, projected_qformer_output, pre_input_ids, post_input_ids

    def generate_text_for_image(self, image: ms.Tensor, pre_input_ids: ms.Tensor, post_input_ids: ms.Tensor, **kwargs):
        """generate text for image by calling llm generate"""
        text_input_ids, projected_qformer_output, pre_input_ids, post_input_ids = self(image, pre_input_ids,
                                                                                       post_input_ids)
        output_ids = self.llm_model.generate(input_ids=text_input_ids.asnumpy(),
                                             image_embeds=projected_qformer_output,
                                             pre_input_ids=pre_input_ids.asnumpy(),
                                             post_input_ids=post_input_ids.asnumpy(),
                                             **kwargs)
        return output_ids


class VisualglmWithLora(VisualGLMModel):
    """ visualglm net for lora finetune"""

    def __init__(self, config):
        super(VisualglmWithLora, self).__init__(config)
        num_layers = config.text_config.num_layers
        pet_config = config.text_config.pet_config
        if not isinstance(pet_config.layer_range, list):
            pet_config.layer_range = [i for i in range(int(pet_config.layer_range))]
        exclude = [str(i) + "$" for i in range(num_layers) if i not in pet_config.layer_range]
        if exclude:
            pet_config.exclude_layers += exclude
        logger.info(f"pet_config: {pet_config}")
        pet_config.target_modules = r"query_key_value$|dense"
        self.llm_model = LoraAdapter.get_pet_model(self.llm_model, pet_config)
        self.batch_size = config.batch_size
        freeze_delta(self, config.text_config.pet_config.pet_type,
                     exclude=[r"*tk_delta_lora*"])
        self.one_prefix = ops.Ones()
        self.expand_dims = P.ExpandDims()
        self.query_length = self.config.qformer_config.query_length

        register_trainer_task()

    def construct(self, image: ms.Tensor, input_ids: ms.tensor, labels: ms.Tensor, position_id,
                  attention_mask):
        """ model network """
        qformer_output = self.forward_qformer_and_proj(image)
        batch_size = input_ids.shape[0]
        seq_length = input_ids.shape[1]

        # [batch_size, max_txt_length, llm_hidden_size]
        pre_input_id = self.slice(input_ids, (0, 0), (batch_size, 3), (1, 1))
        post_input_id = self.slice(input_ids, (0, 3 + 32), (batch_size, seq_length), (1, 1))
        pre_inputs_embedding = self.llm_model.to_text_embeddings(pre_input_id)
        post_inputs_embedding = self.llm_model.to_text_embeddings(post_input_id)
        pre_inputs_embedding = self.cast(pre_inputs_embedding, mstype.float32)
        post_inputs_embedding = self.cast(post_inputs_embedding, mstype.float32)
        llm_inputs_embeds = self.concat_3d([pre_inputs_embedding, qformer_output, post_inputs_embedding])
        llm_attention_mask = self.cast(attention_mask, mstype.int32)
        loss = self.llm_model(llm_inputs_embeds, input_ids, labels.astype(mstype.int32), position_id,
                              llm_attention_mask)
        return loss

    def forward(self, image: ms.Tensor, pre_input_ids: ms.Tensor, post_input_ids: ms.Tensor):
        """ forward by vit and qformer """
        if len(pre_input_ids.shape) == 1:
            pre_input_ids = self.expand_dims(pre_input_ids, 0)

        if len(post_input_ids.shape) == 1:
            post_input_ids = self.expand_dims(post_input_ids, 0)

        batch_size = image.shape[0]
        prefix_ones = self.one_prefix((batch_size, self.query_length), mstype.int32)

        extend_text_input_ids = self.concat_2d([pre_input_ids, prefix_ones, post_input_ids])
        projected_qformer_output = self.forward_qformer_and_proj(image)
        return extend_text_input_ids, projected_qformer_output, pre_input_ids, post_input_ids

    def generate_text_for_image(self, image: ms.Tensor, pre_input_ids: ms.Tensor, post_input_ids: ms.Tensor, **kwargs):
        """generate text for image by calling llm generate"""
        text_input_ids, projected_qformer_output, pre_input_ids, post_input_ids = self.forward(image, pre_input_ids,
                                                                                               post_input_ids)
        output_ids = self.llm_model.generate(input_ids=text_input_ids.asnumpy(),
                                             image_embeds=projected_qformer_output,
                                             pre_input_ids=pre_input_ids.asnumpy(),
                                             post_input_ids=post_input_ids.asnumpy(),
                                             **kwargs)
        return output_ids
