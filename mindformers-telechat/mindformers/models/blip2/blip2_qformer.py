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
Blip2 Qformer, link to ViT.
the main model for image-text pretraining.
"""
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.dtype as mstype
from mindspore import Tensor
import mindspore.numpy as np
from mindspore.ops import operations as P

from mindformers.mindformer_book import MindFormerBook
from mindformers.modules.layers import Linear
from mindformers.models.blip2.blip2 import Blip2Base
from mindformers.models.blip2.blip2_config import Blip2Config
from mindformers.models.blip2.qformer import CrossEntropyLoss
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.tools.logger import logger
from mindformers.tools.utils import get_real_rank, get_real_group_size


def choose_idx_with_prob(weight: Tensor):
    """
    choose idx depend on probability, replace torch.multinomial
    """
    weight_acc = ops.cumsum(weight, -1)
    rand_x = np.rand([1], dtype=weight_acc.dtype) * weight_acc[-1]
    idx = np.argmax(weight_acc > rand_x)
    return idx


class AllGatherWithGrad(nn.Cell):
    """
    AllGather Layer which does not cut gradients.
    """
    def __init__(self):
        super(AllGatherWithGrad, self).__init__()
        self.all_gather = ops.AllGather()
        self.reduce_scatter = ops.ReduceScatter(ops.ReduceOp.SUM)

    def construct(self, x):
        return self.all_gather(x)

    def bprop(self, x, out, dout):
        x = x
        out = out
        return (self.reduce_scatter(dout),)


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class Blip2Qformer(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Args:
        config (Blip2Config): The config of Blip2Qformer.

    Returns:
        Tensor, loss, logits.

    Examples:
        >>> from mindformers.models.blip2 import Blip2Qformer
        >>> model = Blip2Qformer.from_pretrained("blip2_stage1_vit_g")
        >>> type(model)
        <class 'mindformers.models.blip2.blip2_qformer.Blip2Qformer'>
    """

    _support_list = MindFormerBook.get_model_support_list()['blip2']['stage1']

    def __init__(self, config: Blip2Config, **kwargs):
        super(Blip2Qformer, self).__init__(config, **kwargs)
        self.config = config if config is not None else Blip2Config()
        self.group_size = get_real_group_size()
        self.rank = get_real_rank()
        self.visual_encoder, self.ln_vision = self.init_vision_encoder()
        if config.freeze_vision:
            for _, cell in self.visual_encoder.cells_and_names():
                params = cell.get_parameters()
                for param in params:
                    param.requires_grad = False
            self.visual_encoder.set_train(False)
            logger.info("freeze vision encoder")

        qformer_config = self.config.qformer_config

        # note on Atlas 800T A2, function resize_token_embeddings() is not supported,
        # thus in this case, a resized weight will be loaded, i.e:
        # 1. vocab_size: vocab_size + special_token_nums,
        # 2. special_token_nums: 0
        if not qformer_config.resize_token_embeddings:
            qformer_config.vocab_size = qformer_config.vocab_size + qformer_config.special_token_nums
            qformer_config.special_token_nums = 0

        # init qformer
        self.qformer, self.query_tokens = self.init_qformer()

        if qformer_config.resize_token_embeddings:
            # note special token added: bos_token -> [DEC]
            self.qformer.resize_token_embeddings(qformer_config.vocab_size + qformer_config.special_token_nums)

        params = self.qformer.get_parameters()
        # modify layer names
        for param in params:
            if "_query" in param.name:
                key_orig = param.name.replace("_query", "")
                param.set_data(self.qformer.parameters_dict().get(key_orig))

        # parallel settings
        if config.parallel_config:
            dp = config.parallel_config.data_parallel
            mp = config.parallel_config.model_parallel
        else:
            dp = mp = 1

        self.vision_proj = Linear(in_channels=qformer_config.hidden_size,
                                  out_channels=qformer_config.head_embed_dim,
                                  param_init_type=config.dtype,
                                  compute_dtype=config.compute_dtype)
        self.vision_proj.shard(strategy_matmul=((dp, mp), (1, mp)))

        self.text_proj = Linear(in_channels=qformer_config.hidden_size,
                                out_channels=qformer_config.head_embed_dim,
                                param_init_type=config.dtype,
                                compute_dtype=config.compute_dtype)
        self.text_proj.shard(strategy_matmul=((dp, mp), (1, mp)))

        self.itm_head = Linear(in_channels=qformer_config.hidden_size,
                               out_channels=2,
                               param_init_type=config.dtype,
                               compute_dtype=config.compute_dtype)
        self.itm_head.shard(strategy_matmul=((dp, mp), (1, mp)))

        self.gather = P.Gather()
        self.matmul = P.BatchMatMul()
        self.matmul.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
        self.concat = ops.concat
        self.expand_dims = ops.expand_dims
        self.transpose = ops.transpose
        self.zeros = ops.zeros
        self.ones = ops.ones
        self.linspace = ops.linspace
        self.floor = ops.floor
        self.softmax = nn.Softmax(axis=1)
        self.softmax.softmax.shard(((dp, mp, 1),))
        self.eye = ops.eye
        self.masked_fill = ops.masked_fill
        self.stack = ops.stack
        self.broadcast_to = ops.broadcast_to
        self.tile = P.Tile()
        self.tile.shard(((dp, mp, 1, 1),))
        self.normalize = ops.L2Normalize(axis=-1, epsilon=1e-12)

        self.temp = ms.Parameter(
            Tensor(0.07, dtype=config.compute_dtype), requires_grad=True)

        self.max_txt_len = config.max_txt_len
        self.bos_token_id = qformer_config.bos_token_id
        self.pad_token_id = qformer_config.pad_token_id

        if self.group_size > 1:
            self.all_gather = ops.AllGather()
            self.all_gather_with_grad = AllGatherWithGrad()
        self.not_equal = P.NotEqual()
        self.cast = P.Cast()

        self.itc_loss = CrossEntropyLoss(label_smoothing=0.1)
        self.itm_loss = CrossEntropyLoss()

    def construct(self, image: ms.Tensor, text_input_ids: ms.Tensor, return_tuple: bool = False):
        """
        forwarding image and text, compute itc, itm and lm losses.

        Args:
            image (Tensor):
                The indices of images.
            text_input_ids (Tensor):
                The indices of input sequence tokens in the vocabulary.
            return_tuple (bool, defaults to False):
                Whether to return the output separately. If set to True,
                the loss, loss_itc, loss_itm and loss_lm will be returned as a tuple,
                otherwise only the loss will be returned.

        Returns:
            loss (Tensor) or loss_tuple (tuple):
                if return_tuple is False, directly return the loss.
                otherwise, loss, loss_itc, loss_itm and loss_lm will be
                returned as a tuple.
        """
        image_feats, image_embeds, past_key_values = self.forward_image(
            image, use_cache=True)
        image_feats = self.normalize(self.vision_proj(image_feats))

        text_embeds, text_attention_mask = self.forward_text(text_input_ids)
        text_feat = self.normalize(self.text_proj(text_embeds[:, 0, :]))

        image_feats = self.cast(image_feats, mstype.float16)
        text_feat = self.cast(text_feat, mstype.float16)

        ### ============== Image-text Contrastive ===================###
        # if/else branch: distribute setting
        if self.group_size > 1:
            # [batch_size*num_gpu, num_query_tokens, embed_dim]
            image_feats_all = self.all_gather(image_feats)
            # [batch_size*num_gpu, embed_dim]
            text_feat_all = self.all_gather(text_feat)
        else:
            image_feats_all = image_feats
            text_feat_all = text_feat

        batch_size = image.shape[0]
        sim_q2t = []
        for i in range(self.group_size):
            text_feat_part = text_feat_all[i * batch_size: (i + 1) * batch_size]
            sim_temp = self.matmul(self.expand_dims(image_feats, 1), self.expand_dims(
                self.expand_dims(text_feat_part, -1), 0)).squeeze(-1)
            sim_q2t.append(sim_temp.max(-1))
        # query-text similarity: [batch_size, batch_size*num_gpu]
        sim_q2t = self.concat(sim_q2t, axis=1)

        # image-text similarity: aggregate across all query tokens
        sim_i2t = sim_q2t / self.temp

        sim_t2q = []
        # align with ops.matmul, x1 -> [batch_size, batch_size, 1, embed_dim]
        text_feat = self.tile(self.expand_dims(
            self.expand_dims(text_feat, 1), 1), (1, batch_size, 1, 1))
        for i in range(self.group_size):
            image_feats_part = image_feats_all[i * batch_size: (i + 1) * batch_size]
            # align with ops.matmul, x2 -> [batch_size, batch_size, embed_dim, num_query_tokens]
            image_feats_part = self.tile(self.expand_dims(
                self.transpose(image_feats_part, (0, 2, 1)), 0), (batch_size, 1, 1, 1))
            # compute similarity same as ops.matmul
            sim_temp = self.matmul(text_feat, image_feats_part).squeeze(2)
            sim_t2q.append(sim_temp.max(-1))
        # text-query similarity: [batch_size, batch_size*num_gpu]
        sim_t2q = self.concat(sim_t2q, axis=1)

        # text-image similarity: aggregate across all query tokens
        sim_t2i = sim_t2q / self.temp

        targets = self.floor(self.linspace(ms.Tensor(self.rank * batch_size, mstype.float32),
                                           ms.Tensor(self.rank * batch_size + batch_size - 1, mstype.float32),
                                           batch_size)).astype(mstype.int32)

        sim_i2t = self.cast(sim_i2t, mstype.float32)
        sim_t2i = self.cast(sim_t2i, mstype.float32)
        loss_itc = (self.itc_loss(sim_i2t, targets) +
                    self.itc_loss(sim_t2i, targets)) / 2

        # ============== Image-text Matching ===================
        # mask text-image similarity as weights
        weights_t2i, weights_i2t = self.fill_masked_weight(sim_t2i, sim_i2t, batch_size)

        # choose negative image/text for each text/image
        image_embeds_neg, text_ids_neg = self.choose_negative_targets(weights_t2i,
                                                                      weights_i2t,
                                                                      batch_size,
                                                                      image_embeds,
                                                                      text_input_ids)

        text_ids_all = self.concat(
            [text_input_ids, text_input_ids, text_ids_neg], axis=0)  # pos, pos, neg

        image_embeds_all = self.concat(
            [image_embeds, image_embeds_neg, image_embeds], axis=0)  # pos, neg, pos

        vl_embeddings = self.forward_text_and_image(
            image_embeds_all, text_ids_all, vit_computed=True)
        vl_output = self.itm_head(vl_embeddings)
        logits = vl_output.mean(axis=1)

        itm_labels = self.concat(
            [self.ones(batch_size, mstype.int32), self.zeros(2 * batch_size, mstype.int32)],
            axis=0
        )
        loss_itm = self.itm_loss(logits, itm_labels)

        # ================= Image Captioning ========================
        decoder_input_ids = text_input_ids.copy().astype(mstype.float32)
        decoder_input_ids[:, 0] = self.bos_token_id
        decoder_input_ids = decoder_input_ids.astype(mstype.int32)
        labels = decoder_input_ids.masked_fill(
            decoder_input_ids == self.pad_token_id, Tensor(-100, decoder_input_ids.dtype)
        )

        query_tokens = self.broadcast_to(self.query_tokens, (image_embeds.shape[0], -1, -1))
        query_atts = self.ones(query_tokens.shape[:-1], mstype.float32)

        attention_mask = self.concat(
            [query_atts, text_attention_mask.astype(mstype.float32)], axis=1)
        lm_output = self.qformer(
            decoder_input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            labels=labels,
        )

        loss_lm = lm_output[0]
        loss = loss_itc + loss_itm + loss_lm

        if return_tuple:
            return (
                loss,
                loss_itc,
                loss_itm,
                loss_lm
            )
        return loss

    def forward_image(self, image, use_cache=False):
        """ forawrd image through vit and the bert model.

        Args:
            image (Tensor): input image
            use_cache (bool, optional): whether to return past_key_values.

        Returns:
            hidden_states, image_embeds_frozen, past_key_values (optional)
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
            use_cache=use_cache,
        )
        if use_cache:
            return query_output[0], image_embeds_frozen, query_output[1]
        return query_output[0], image_embeds_frozen

    def forward_text(self, text_input_ids):
        """ forawrd text_ids through the bert model.

        Args:
            text_input_ids (Tensor): input text_ids

        Returns:
            text embeddings and mask
        """
        attention_mask = self.cast(self.not_equal(
            text_input_ids, self.pad_token_id), mstype.float32)
        text_output = self.qformer.bert(
            text_input_ids,
            attention_mask=attention_mask
        )
        # text embeddings and mask
        return text_output[0], attention_mask

    def forward_text_and_image(self, image_inputs, text_ids, vit_computed=False):
        """ forward text and image at the same time to the bert model.

        Args:
            image_inputs(Tensor): input image or image embeds (computed)
            text_input_ids (Tensor): input text_ids
            vit_computed (bool, optional): whether image embeds is computed

        Returns:
            multimodal embeddings
        """
        if not vit_computed:
            image_embeds_frozen = self.ln_vision(
                self.visual_encoder(image_inputs))
        else:
            image_embeds_frozen = image_inputs
        text_atts = self.cast(self.not_equal(
            text_ids, self.pad_token_id), mstype.float32)
        image_atts = self.ones(
            image_embeds_frozen.shape[:-1], mstype.float32)
        query_tokens = self.broadcast_to(
            self.query_tokens, (image_embeds_frozen.shape[0], -1, -1))
        query_atts = self.ones(query_tokens.shape[:-1], mstype.float32)
        attention_mask = self.concat([query_atts, text_atts], axis=1)

        output_itm = self.qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds_frozen,
            encoder_attention_mask=image_atts
        )
        # multimodal embeddings
        multimodal_embeds = output_itm[0][:, : query_tokens.shape[1], :]
        return multimodal_embeds

    def fill_masked_weight(self, sim_t2i, sim_i2t, batch_size):
        """return masked weights based on similarity

        Args:
            sim_t2i (Tensor): text-to-image similarity
            sim_i2t (Tensor): image-to-text similarity
            batch_size (int): current batch size
        """
        weights_t2i = self.softmax(sim_t2i) + 1e-4
        diag_fill_mask_t2i = self.eye(weights_t2i.shape[0], batch_size, mstype.bool_)
        filled_weights_t2i = self.masked_fill(
            weights_t2i[:, self.rank * batch_size: self.rank * batch_size + batch_size],
            diag_fill_mask_t2i, Tensor(0, weights_t2i.dtype))
        weights_t2i[:, self.rank * batch_size: self.rank *
                    batch_size + batch_size] = filled_weights_t2i

        weights_i2t = self.softmax(sim_i2t) + 1e-4
        diag_fill_mask_i2t = self.eye(weights_i2t.shape[0], batch_size, mstype.bool_)
        filled_weights_i2t = self.masked_fill(
            weights_i2t[:, self.rank * batch_size: self.rank * batch_size + batch_size],
            diag_fill_mask_i2t, Tensor(0, weights_i2t.dtype))
        weights_i2t[:, self.rank * batch_size: self.rank *
                    batch_size + batch_size] = filled_weights_i2t

        return weights_t2i, weights_i2t

    def choose_negative_targets(self,
                                weights_t2i,
                                weights_i2t,
                                batch_size,
                                image_embeds,
                                text_input_ids):
        """choose negative targets for each image/text.

        Args:
            weights_t2i (Tensor): masked text-to-image weights
            weights_i2t (Tensor): masked image-to-text weights
            batch_size (int): current batch size
            image_embeds (Tensor): image embeddings
            text_input_ids (Tensor): text ids

        Returns:
            image_embeds_neg (Tensor): negative image_embeds
            text_ids_neg (Tensor): negative text ids
        """
        if self.group_size > 1:
            # do all_gather with grads, align with torch impl.
            image_embeds_gathered = self.all_gather_with_grad(image_embeds)
            text_ids_gathered = self.all_gather(text_input_ids)
        else:
            image_embeds_gathered = image_embeds
            text_ids_gathered = text_input_ids

        # select a negative image for each text
        image_embeds_neg_idx = self.zeros(batch_size, mstype.int32)
        for i in range(batch_size):
            image_embeds_neg_idx[i] = choose_idx_with_prob(weights_t2i[i])
        image_embeds_neg = self.gather(image_embeds_gathered, image_embeds_neg_idx, 0)

        # select a negative text for each image
        text_ids_neg_idx = self.zeros(batch_size, mstype.int32)
        for i in range(batch_size):
            text_ids_neg_idx[i] = choose_idx_with_prob(weights_i2t[i])
        text_ids_neg = self.gather(text_ids_gathered, text_ids_neg_idx, 0)

        return image_embeds_neg, text_ids_neg

    def compute_itm(self, image_inputs, text_ids, vit_computed=False):
        """ compute image-text matching scores for the model.
        Args:
            image_inputs (Tensor): input image or image embeds (computed)
            text_ids (Tensor): input text_ids
            vit_computed (bool, optional): whether image embeds is computed

        Returns:
            itm_logit
        """
        vl_embeddings = self.forward_text_and_image(
            image_inputs, text_ids, vit_computed)
        itm_logit = self.itm_head(vl_embeddings)
        itm_logit = itm_logit[:, :, 1].mean(axis=1)
        return itm_logit

    def get_image_feature(self, image, output_past_keys=False):
        """extract image feature"""
        forward_image_outputs = self.forward_image(image, output_past_keys)
        image_features = ops.L2Normalize(
            axis=-1, epsilon=1e-12)(self.vision_proj(forward_image_outputs[0]))
        return image_features

    def get_text_feature(self, input_ids):
        """extract text feature"""
        forward_text_outputs = self.forward_text(input_ids)
        text_features = ops.L2Normalize(
            axis=-1, epsilon=1e-12)(self.text_proj(forward_text_outputs[0]))
        return text_features

    def extract_features(self, samples, mode="multimodal"):
        """ extract feature as well as embeds by given mode,

        Args:
            samples (tuple of Tensors): image/text input
            mode (str): [image, text, multimodal]
        """
        image = samples.get("image")
        text_ids = samples.get("text_input")

        # assert mode is one of "image", "text", "multimodal"
        if mode not in ["image", "text", "multimodal"]:
            raise ValueError(f"mode must be one of 'image', 'text', 'multimodal', but got {mode}.")

        # initialize output
        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None

        if mode == "image":  # return query features
            if image is None:
                raise ValueError(f"Image is not provided for mode 'image' or 'multimodal'.")
            forward_image_outputs = self.forward_image(image, use_cache=False)
            image_embeds = forward_image_outputs[0]
            image_features = ops.L2Normalize(
                axis=-1, epsilon=1e-12)(self.vision_proj(image_embeds))

        elif mode == "text":
            if text_ids is None:
                raise ValueError(f"text input is None for mode 'text' or 'multimodal'.")
            forward_text_outputs = self.forward_text(text_ids)
            text_embeds = forward_text_outputs[0]
            text_features = ops.L2Normalize(
                axis=-1, epsilon=1e-12)(self.text_proj(text_embeds))

        elif mode == "multimodal":  # return multimodal query features
            multimodal_embeds = self.forward_text_and_image(image, text_ids, vit_computed=False)
        return (image_embeds,
                image_features,
                text_embeds,
                text_features,
                multimodal_embeds)


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class Blip2Classifier(Blip2Qformer):
    """
    Blip2Classifier rely on Blip2Qformer, used for zero-shot classification.

    Args:
        config (Blip2Config): The config of Blip2Qformer.

    Examples:
        >>> from mindformers import Blip2Classifier
        >>> model_type = 'blip2_stage1_classification'
        >>> model = Blip2Classifier.from_pretrained(model_type)
        >>> type(model)
        <class 'mindformers.models.blip2.blip2_qformer.Blip2Classifier'>
    """

    def __init__(self, config: Blip2Config, **kwargs):
        super(Blip2Classifier, self).__init__(config, **kwargs)
        self.load_checkpoint(config)

    def construct(self, image: ms.Tensor, text_input_ids: ms.Tensor, return_tuple: bool = False):
        image_features = self.get_image_feature(image)[:, 0]
        text_features = self.get_text_feature(text_input_ids)[:, 0]
        sims = ops.matmul(image_features, text_features.T) / self.temp
        return sims, sims.T  # no label as input (compare to CLIP)
