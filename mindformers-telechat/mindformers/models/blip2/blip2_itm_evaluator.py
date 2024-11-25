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
# This file was refer to project:
# https://github.com/salesforce/LAVIS/tree/main/lavis/models/blip2_models
# ============================================================================
"""
Blip2 Qformer Evaluator, used for itm-score computing.
supports data-parallel computing.
"""
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P

from mindformers.mindformer_book import MindFormerBook
from mindformers.models.blip2.blip2 import Blip2PreTrainedModel
from mindformers.models.blip2.blip2_config import Blip2Config
from mindformers.models.blip2.blip2_qformer import Blip2Qformer
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.tools.logger import logger


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class Blip2ItmEvaluator(Blip2PreTrainedModel):
    """
    The evaluator class for BLIP2 first-stage trained model.
    Args:
        config (Blip2Config): The config of Blip2Qformer.
        blip2_qformer (Blip2Qformer): The trained Blip2Qformer, default is None.

    Returns:
        sims_matrix, vit_outputs, text_ids

    Examples:
        >>> from mindformers.models.blip2.blip2_itm_evaluator import Blip2ItmEvaluator
        >>> model = Blip2ItmEvaluator.from_pretrained("blip2_stage1_evaluator")
        >>> type(model)
        <class 'mindformers.models.blip2.blip2_itm_evaluator.Blip2ItmEvaluator'>
    """
    _support_list = MindFormerBook.get_model_support_list()['blip2']['stage1']

    def __init__(self, config: Blip2Config, blip2_qformer: Blip2Qformer = None, **kwargs):
        super(Blip2ItmEvaluator, self).__init__(config, **kwargs)
        if not blip2_qformer:
            blip2_qformer = Blip2Qformer(config, **kwargs)
            blip2_qformer.load_checkpoint(config)
        self.blip2_qformer = blip2_qformer
        self.text_forwarder = Blip2TextForwarder(self.blip2_qformer)
        self.image_forwarder = Blip2ImageForwarder(self.blip2_qformer)
        self.itm_computer = Blip2ItmComputer(self.blip2_qformer)
        self.group_size = blip2_qformer.group_size
        self.rank = blip2_qformer.rank
        self.score_reducer = ScoreReducer(self.group_size)
        self.topk = P.TopK()

        if self.group_size > 1:
            self.all_gather = ops.AllGather()

    def load_checkpoint(self, config):
        logger.info("For Blip2ItmEvaluator, the checkpoint is already loaded in the __init__() process.")

    def construct(self, image_feats, text_feats, vit_outputs=None, text_ids=None, add_extra_itm_score: bool = False):
        """
        compute image-text matching scores, in matrix format.

        Args:
            image_feats: image features
            text_feats: text features
            vit_outputs: outputs of Vision Transformer
            text_ids: input text_ids
            add_extra_itm_score (bool, optional): whether to add extra scores (model decides), Defaults to False.

        Returns:
            sims_matrix: similarity matrix
            vit_outputs: outputs of Vision Transformer
            text_ids: input text_ids
        """
        if self.group_size > 1:
            text_feats = self.all_gather(text_feats)
        text_feats = text_feats.T

        sims_matrix = []
        for image_feat in image_feats:
            sim_q2t = ops.matmul(image_feat, text_feats)
            sim_i2t = sim_q2t.max(0)
            sims_matrix.append(sim_i2t)
        sims_matrix = ops.stack(sims_matrix, axis=0)

        if self.group_size > 1:
            sims_matrix = self.all_gather(sims_matrix)
            if add_extra_itm_score:
                vit_outputs = self.all_gather(vit_outputs)
                text_ids = self.all_gather(text_ids)

        return sims_matrix, vit_outputs, text_ids


class Blip2TextForwarder(nn.Cell):
    """
    TextForwarder, same function as blip2_qformer.forward_text.
    """
    def __init__(self, blip2_qformer: Blip2Qformer, **kwargs):
        super(Blip2TextForwarder, self).__init__(**kwargs)
        self.qformer = blip2_qformer.qformer
        self.text_proj = blip2_qformer.text_proj
        self.pad_token_id = blip2_qformer.pad_token_id
        self.not_equal = P.NotEqual()
        self.cast = P.Cast()
        self.normalize = ops.L2Normalize(axis=-1, epsilon=1e-7)

    def construct(self, text_input_ids):
        """ forawrd text_ids through the bert model.

        Args:
            text_input_ids (Tensor): input text_ids

        Returns:
            text embeddings and text feat
        """
        attention_mask = self.cast(self.not_equal(
            text_input_ids, self.pad_token_id), mstype.float32)
        text_output = self.qformer.bert(
            text_input_ids,
            attention_mask=attention_mask
        )
        text_embed = text_output[0]
        text_feat = self.normalize(self.text_proj(text_embed[:, 0, :]))
        # text embeddings and features
        return text_embed, text_feat


class Blip2ImageForwarder(nn.Cell):
    """
    ImageForwarder, same function as blip2_qformer.forward_image.
    """
    def __init__(self, blip2_qformer: Blip2Qformer, **kwargs):
        super(Blip2ImageForwarder, self).__init__(**kwargs)
        self.qformer = blip2_qformer.qformer
        self.ln_vision = blip2_qformer.ln_vision
        self.visual_encoder = blip2_qformer.visual_encoder
        self.vision_proj = blip2_qformer.vision_proj
        self.query_tokens = blip2_qformer.query_tokens
        self.ones = ops.ones
        self.broadcast_to = ops.broadcast_to
        self.normalize = ops.L2Normalize(axis=-1, epsilon=1e-7)

    def construct(self, image):
        """ forawrd image through vit and the bert model.

        Args:
            image (Tensor): input image

        Returns:
            image_feat, image_embeds_frozen
        """
        image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
        image_atts = self.ones(
            image_embeds_frozen.shape[:-1], mstype.float32)
        query_tokens = self.broadcast_to(
            self.query_tokens, (image_embeds_frozen.shape[0], -1, -1))

        query_output = self.qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds_frozen,
            encoder_attention_mask=image_atts,
            use_cache=False,
        )

        image_embed = query_output[0]
        image_embed = self.vision_proj(image_embed)
        image_feat = self.normalize(image_embed)

        return image_feat, image_embeds_frozen


class Blip2ItmComputer(nn.Cell):
    """
    ItmComputer, same function as blip2_qformer.compute_itm.
    """
    def __init__(self, blip2_qformer: Blip2Qformer, **kwargs):
        super(Blip2ItmComputer, self).__init__(**kwargs)
        self.qformer = blip2_qformer.qformer
        self.itm_head = blip2_qformer.itm_head
        self.pad_token_id = blip2_qformer.pad_token_id
        self.query_tokens = blip2_qformer.query_tokens
        self.not_equal = P.NotEqual()
        self.cast = P.Cast()
        self.ones = ops.ones
        self.broadcast_to = ops.broadcast_to
        self.concat = ops.concat

    def construct(self, image_inputs, text_ids):
        """ compute image-text matching scores for the model.
        Args:
            image_inputs (Tensor): input image or image embeds (computed)
            text_ids (Tensor): input text_ids

        Returns:
            itm_logit
        """
        text_atts = self.cast(self.not_equal(
            text_ids, self.pad_token_id), mstype.float32)
        image_atts = self.ones(
            image_inputs.shape[:-1], mstype.float32)
        query_tokens = self.broadcast_to(
            self.query_tokens, (image_inputs.shape[0], -1, -1))
        query_atts = self.ones(query_tokens.shape[:-1], mstype.float32)
        attention_mask = self.concat([query_atts, text_atts], axis=1)

        output_itm = self.qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_inputs,
            encoder_attention_mask=image_atts
        )
        # multimodal embeddings
        vl_embeddings = output_itm[0][:, : query_tokens.shape[1], :]
        itm_logit = self.itm_head(vl_embeddings)
        itm_logit = itm_logit[:, :, 1].mean(axis=1)
        return itm_logit


class ScoreReducer(nn.Cell):
    """
    A reducer layer to collect results from multiple cards.
    """
    def __init__(self, group_size, **kwargs):
        super(ScoreReducer, self).__init__(**kwargs)
        self.group_size = group_size
        if self.group_size > 1:
            self.all_reduce = ops.AllReduce(ops.ReduceOp.SUM)

    def construct(self, score_matrix_i2t, score_matrix_t2i):
        # allReduce results for DP
        if self.group_size > 1:
            score_matrix_i2t = self.all_reduce(score_matrix_i2t)
            score_matrix_t2i = self.all_reduce(score_matrix_t2i)

        return score_matrix_i2t, score_matrix_t2i
