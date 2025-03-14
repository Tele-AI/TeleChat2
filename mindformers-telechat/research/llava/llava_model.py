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
"""
Llava Model
"""
from typing import Optional
import numpy as np

import mindspore as ms
from mindspore import Parameter, Tensor
from mindspore import dtype as mstype
from mindspore import nn
from mindspore.context import ParallelMode
from mindspore.common.initializer import Normal
from mindspore.ops import operations as P
from mindspore.parallel._utils import _get_parallel_mode, _is_sharding_propagation

from mindformers import BaseXModalToTextModel
from mindformers.models.build_model import build_network
from mindformers import MindFormerRegister, MindFormerModuleType
from mindformers.models.utils import lazy_inline
from mindformers.models.configuration_utils import PretrainedConfig
from mindformers.modules import Linear
from mindformers.tools import logger

from llava_config import LlavaConfig


class LlavaAdapter(nn.Cell):
    r"""
    Module to resize the image dimension into the llm model dimension
    Args:
        config (LlavaConfig): The config of Llava model.
    """

    def __init__(self, config: LlavaConfig):
        super(LlavaAdapter, self).__init__(config)

        self.adapter = Linear(
            in_channels=config.vision_config.model_config.hidden_size,
            out_channels=config.text_config.model_config.hidden_size,
            compute_dtype=config.compute_dtype,
            param_init_type=config.param_init_type,
            skip_redistribution=config.is_dynamic
        )
        self.activation_func = P.GeLU()
        self.adapter_2 = Linear(
            in_channels=config.text_config.model_config.hidden_size,
            out_channels=config.text_config.model_config.hidden_size,
            compute_dtype=config.compute_dtype,
            param_init_type=config.param_init_type,
            skip_redistribution=config.is_dynamic
        )

    def construct(self, x):
        """adapter forward method"""
        output = self.adapter(x)
        output = self.activation_func(output)
        output = self.adapter_2(output)
        return output

    def shard(self, parallel_config):
        """distributed method"""
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        self.adapter.shard(((dp, 1), (mp, 1)), strategy_bias=((dp, mp), (mp,)))
        self.adapter_2.shard(((dp, mp), (1, mp)), strategy_bias=((dp, 1), (1,)))
        self.activation_func.shard(((dp, 1, mp),))


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class LlavaVlm(BaseXModalToTextModel):
    """
    Provide Llava1.5 training loss or logits through network.

    Args:
        config (LlavaConfig): The config of Llava model.
    """

    @lazy_inline
    def __init__(self, config: LlavaConfig, **kwargs):
        super(LlavaVlm, self).__init__(config, **kwargs)
        self.config = config if config is not None else LlavaConfig()

        self.language_model = build_network(self.config.text_config)
        self.vision_encoder = build_network(self.config.vision_config)
        self.adapter = LlavaAdapter(self.config)

        self.vision_encoder.pipeline_stage = 0
        self.adapter.pipeline_stage = 0

        self.is_first_iteration = True
        self.use_past = config.use_past
        self.pad_token_id = config.pad_token_id
        self.eos_token_id = config.eos_token_id
        self.image_size = self.config.vision_config.model_config.image_size
        self.num_queries = self.config.vision_config.model_config.num_queries
        self.ignore_token_id = Tensor(config.ignore_token_id, mstype.int32)

        if config.is_dynamic:
            self.reshape.add_prim_attr("skip_redistribution", True)
        self.slice = P.StridedSlice()
        self.ones = P.Ones()

        dp = config.parallel_config.data_parallel
        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            self.adapter.shard(config.parallel_config)
            self.slice.shard(((dp, 1),))

        self.freeze_component()

    def freeze_component(self):
        """freeze parameters of model according to different command"""
        if self.config.freeze_vision:
            logger.info("freeze vision encoder")
            for param in self.vision_encoder.trainable_params():
                param.requires_grad = False

        if self.config.freeze_llm:
            logger.info("freeze llm model")
            for param in self.language_model.trainable_params():
                param.requires_grad = False

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """prepare inputs for generation in inference"""
        if self.config.is_dynamic and "origin_inputs" in kwargs:
            input_ids = kwargs.get("origin_inputs")
        images = kwargs.pop("images")
        slot_mapping = kwargs.pop("slot_mapping")
        image_context_pos = kwargs.pop("image_context_pos", None)
        if image_context_pos is not None and not isinstance(image_context_pos, ms.Tensor):
            image_context_pos = ms.Tensor(image_context_pos, mstype.int32)

        model_inputs = {
            "input_ids": ms.Tensor(input_ids, mstype.int32),
            "images": images,
            "image_context_pos": image_context_pos,
            "slot_mapping": Tensor.from_numpy(slot_mapping)
        }
        if self.is_first_iteration:
            batch_valid_length = kwargs.get("valid_length_each_example")
            model_inputs = self._prepare_inputs_for_prefill_flatten(
                input_ids, batch_valid_length, slot_mapping, model_inputs
            )

        return model_inputs

    def _prepare_inputs_for_prefill_flatten(self, input_ids, batch_valid_length, slot_mapping, model_inputs):
        """prepare inputs ids for prefill flatten"""
        batch_valid_length_bs = batch_valid_length.shape[0]  # [bs,]
        input_ids_list = []
        for i in range(batch_valid_length_bs):
            context_len = batch_valid_length[i]
            input_ids_list.append(input_ids[i][:context_len])
        input_ids = np.concatenate(input_ids_list, 0)
        input_ids = input_ids.reshape((1, -1))
        slot_mapping = np.delete(slot_mapping, np.where(slot_mapping == -1))
        model_inputs["input_ids"] = Tensor.from_numpy(input_ids.astype(np.int32))
        model_inputs["slot_mapping"] = Tensor.from_numpy(slot_mapping)
        return model_inputs

    def prepare_inputs_for_predict_layout(self, input_ids, **kwargs):
        """prepare inputs for predict layout"""
        input_ids = Tensor(input_ids, mstype.int32)
        bs, seq = input_ids.shape
        if "images" in kwargs:
            images = Tensor(kwargs.get("images"))
        else:
            images = Tensor(np.random.random((bs, 3, self.image_size, self.image_size)), ms.float32)

        if "image_context_pos" in kwargs:
            image_context_pos = Tensor(kwargs.get("image_context_pos"))
        else:
            image_context_pos = Tensor(np.random.randint(0, self.num_queries, (bs, self.num_queries, 2)), ms.int32)

        slot_mapping = Tensor(np.ones(shape=tuple([bs * seq])), mstype.int32)
        return input_ids, images, image_context_pos, None, None, None, None, None, None, None, None, None, slot_mapping

    def set_dynamic_inputs(self):
        """set inputs when is_dynamic=True"""
        dynamic_input_ids = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_images = Tensor(shape=[None, None, None, None], dtype=mstype.float32)
        dynamic_img_pos = Tensor(shape=[None, self.num_queries, 2], dtype=mstype.int32)
        dynamic_batch_valid_length = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_block_tables = Tensor(shape=[None, None], dtype=mstype.int32)
        dynamic_slot_mapping = Tensor(shape=[None], dtype=mstype.int32)
        self.set_inputs(dynamic_input_ids, dynamic_images, dynamic_img_pos, None, None, None,
                        None, None, dynamic_batch_valid_length, None, None,
                        dynamic_block_tables, dynamic_slot_mapping)

        logger.info("Set dynamic inputs for LLava-1.5")

    def construct(self, input_ids, images, image_context_pos: Tensor = None, labels=None, input_position=None,
                  position_ids=None, attention_mask=None, init_reset=None, batch_valid_length=None, batch_index=None,
                  zactivate_len=None, block_tables=None, slot_mapping=None):
        r"""
        LlavaVlm forward.

        Args:
            input_ids(Tensor): the tokenized inputs with datatype int32, Tensor of shape :math:`(batch, seq\_length)`.
            images(Tensor): the image tensor with datatype float32, Tensor of shape :math:
            `(batch, 3, image_resolution, image_resolution)`
            image_context_pos(Tensor): the position index of the image in final input embedding. Tensor of shape :math
            `(batch, num_queries, 2)`
            labels(Tensor): the tokenized labels with datatype int32, Tensor of shape :math:`(batch, seq\_length)`.
            input_position(Tensor): current position, used by model.predict.
            position_ids(Tensor): Reserved param, not used.
            attention_mask(Tensor): Reserved param, not used.
            input_embeds(Tensor): the input embedding Tensor of shape :math:`(batch, seq\_length, hidden_size)`.
                Default None.
            init_reset(bool, optional): A bool tensor with shape [1], used to clear the past key parameter and
                past value parameter used in the incremental prediction. Default True.
            batch_valid_length(Tensor): the past calculated the index with datatype int32, used for incremental
                prediction. Tensor of shape :math:`(batch_size,)`. Default None.
            block_tables (Tensor[int64]): Store mapping tables for each sequence.
            slot_mapping (Tensor[int32]): Store token cache physical slot index.
        Returns:
            Tensor: The loss or (logits, tokens, input_mask) of the network.
        """
        bs, seq_len = self.shape(input_ids)
        if self.training:
            tokens = self.slice(input_ids, (0, 0), (bs, seq_len - 1), (1, 1))
            labels = self.slice(labels, (0, 1), (bs, seq_len), (1, 1))
        else:
            tokens = input_ids
        input_embeds = self.language_model.to_embeddings(tokens)

        if self.is_first_iteration:
            if images is not None and images.ndim == 5:
                images_shape = self.shape(images)
                new_shape = (images_shape[0] * images_shape[1], images_shape[2], images_shape[3], images_shape[4])
                images = self.reshape(images, new_shape)
            image_embeds = self.vision_encoder(images)

            image_embeds = self.adapter(image_embeds)  # 1 576 4096
            input_embeds = self.update_modal_to_text(image_embeds, input_embeds, image_context_pos)

        return self.language_model(
            input_ids=tokens,
            input_embeds=input_embeds,
            labels=labels,
            batch_valid_length=batch_valid_length,
            block_tables=block_tables,
            slot_mapping=slot_mapping,
            input_position=input_position,
            init_reset=init_reset,
            position_ids=position_ids,
            attention_mask=attention_mask,
            batch_index=batch_index,
            zactivate_len=zactivate_len
        )

    def kvcache(self, layer_idx):
        key_cache = self.language_model.layers[layer_idx].attention.infer_attention.paged_attention_mgr.key_cache
        value_cache = self.language_model.layers[layer_idx].attention.infer_attention.paged_attention_mgr.value_cache
        return key_cache, value_cache


class LayerNorm(nn.LayerNorm):
    r"""Implementation That Supports Fp16 Inputs But Fp32 Gains Biases.

    Args:
        x (ms.Tensor): Input tensor.
            The detailed function could refer to mindspore.nn.LayerNorm.

    Return:
        y (ms.Tensor): Normalized tensor.
    """
    # pylint: disable=C0111
    def construct(self, x: ms.Tensor):
        y = super().construct(P.Cast()(x, ms.float32))
        return P.Cast()(y, x.dtype)


class MultiheadAttention(nn.Cell):
    r"""MultiheadAttention, With Layers As Input For Initialization

    Args:
        d_model (int): The feature dimension
        n_head (int): The number of attention heads
        layers (int): The number of transformers, used for weight initialization
        dtype (mstype): The type of calculation, [mstype.float32, mstype.float16].
    """

    def __init__(self, d_model: int, n_head: int, layers: int, compute_dtype: mstype, param_init_type):
        super(MultiheadAttention, self).__init__()
        self.num_heads = n_head
        self.head_dim = d_model // n_head
        proj_std = (d_model ** -0.5) * ((2 * layers) ** -0.5)
        attn_std = d_model ** -0.5

        # self.softmax = nn.Softmax(-1)
        self.scaling = self.head_dim ** -0.5
        self.reshape = P.Reshape()
        self.slice = P.StridedSlice()
        self.batch_matmul_q_k = P.BatchMatMul(transpose_b=True)
        self.mul = P.Mul()
        self.softmax = P.Softmax()
        self.shape = P.Shape()
        self.batch_matmul = P.BatchMatMul()
        self.merger_head_transpose = P.Transpose()
        self.add = P.Add()
        self.transpose = P.Transpose()
        self.out_proj = Linear(d_model, d_model, weight_init=Normal(mean=0.0, sigma=proj_std),
                               compute_dtype=compute_dtype, param_init_type=param_init_type)
        self.in_proj = Linear(d_model, 3 * d_model, weight_init=Normal(mean=0.0, sigma=attn_std),
                              compute_dtype=compute_dtype, param_init_type=param_init_type)

        self.softmax_dtype = mstype.float32

        self.cast_attn = P.Cast()
        self.cast = P.Cast()
        self.split_qkv = ms.ops.auto_generate.SplitWithSize()
        self.split_qkv.add_prim_attr("skip_redistribution", True)
        self.dtype = compute_dtype

    def construct(self, query: ms.Tensor, attn_mask: Optional[ms.Tensor] = None):
        r"""Construct

        Args:
            query (ms.Tensor): query of attention.
            attn_mask (Optional[ms.Tensor]): attention mask.

        Returns:
            attn_output (ms.Tensor): attention output.
        """
        batch_size, len_tgt, width = query.shape
        qkv = self.in_proj(query)
        query, key, value = self.split_qkv(qkv, (width, width, width), 2)

        query = self.cast(self.transpose(self.reshape(query, (batch_size, len_tgt, self.num_heads, self.head_dim)),
                                         (0, 2, 1, 3)), self.dtype)
        key = self.cast(self.transpose(self.reshape(key, (batch_size, len_tgt, self.num_heads, self.head_dim)),
                                       (0, 2, 1, 3)), self.dtype)
        value = self.cast(self.transpose(self.reshape(value, (batch_size, len_tgt, self.num_heads, self.head_dim)),
                                         (0, 2, 1, 3)), self.dtype)

        attn = self._attn(query, key, value, attn_mask)
        return self.out_proj(attn)

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
        score = self.batch_matmul_q_k(query, key)

        score = self.mul(score, self.scaling)
        if mask is not None:
            score = self.add(mask, score)

        attention_probs = self.softmax(self.cast_attn(score, self.softmax_dtype))
        weighted_values = self.batch_matmul(self.cast(attention_probs, self.dtype), value)
        return self._merge_heads(weighted_values)

    def _merge_heads(self, x):
        """
        convert a 4d input to a 3d output

        Inputs:
            x: input tensor

        Output:
            x_merge: the 2d output
        """
        # [bs, n_head, seq/1, head_dim]
        x = self.merger_head_transpose(x, (0, 2, 1, 3))  # dp,mp,1,1 -> dp,1,mp,1
        bs, seq_len, n_head, head_dim = self.shape(x)
        new_shape = (bs, seq_len, n_head * head_dim)
        return self.reshape(x, new_shape)

    # pylint: disable=C0111
    def shard(self, parallel_config):
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        self.in_proj.shard(strategy_matmul=((dp, 1), (mp, 1)), strategy_bias=((dp, mp), (mp,)))
        self.split_qkv.shard(((dp, 1, 1),))
        self.transpose.shard(((dp, 1, mp, 1),))
        self.batch_matmul_q_k.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
        self.mul.shard(((dp, mp, 1, 1), ()))
        self.softmax.shard(((dp, mp, 1, 1),))
        self.batch_matmul.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
        self.merger_head_transpose.shard(((dp, mp, 1, 1),))
        self.out_proj.shard(strategy_matmul=((dp, mp), (1, mp)), out_strategy_matmul=((dp, 1),),
                            strategy_bias=((dp, 1), (1,)))


class QuickGELU(nn.Cell):
    r"""QuickGELU of CLIP"""

    def __init__(self, ratio: Optional[int] = 1.702):
        super(QuickGELU, self).__init__()
        self.ratio = ratio
        self.sigmoid = P.Sigmoid()
        self.mul = P.Mul()
        self.mul2 = P.Mul()

    # pylint: disable=C0111
    def construct(self, input_x: ms.Tensor):
        return self.mul(input_x, self.sigmoid(self.mul2(input_x, self.ratio)))

    # pylint: disable=C0111
    def shard(self, parallel_config):
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        self.mul2.shard(((dp, 1, mp), ()))
        self.sigmoid.shard(((dp, 1, mp),))
        self.mul.shard(((dp, 1, mp), (dp, 1, mp)))


class MLP(nn.Cell):
    """
    A multilayer perceptron for ViT
    """

    def __init__(self, layers: int, input_channel_dim: int, output_channel_dim: int, compute_dtype, param_init_type):
        super().__init__()

        proj_std = (input_channel_dim ** -0.5) * ((2 * layers) ** -0.5)
        fc_std = (2 * input_channel_dim) ** -0.5
        self.c_fc = Linear(input_channel_dim, output_channel_dim, weight_init=Normal(mean=0.0, sigma=fc_std),
                           compute_dtype=compute_dtype, param_init_type=param_init_type)

        self.c_proj = Linear(output_channel_dim, input_channel_dim, weight_init=Normal(mean=0.0, sigma=proj_std),
                             compute_dtype=compute_dtype,
                             param_init_type=param_init_type)

        self.gelu = QuickGELU()
        self.cast = P.Cast()
        self.dtype = compute_dtype

    # pylint: disable=C0111
    def construct(self, x):
        ori_dtype = x.dtype
        x = self.cast(x, self.dtype)
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.cast(x, ori_dtype)

    # pylint: disable=C0111
    def shard(self, parallel_config):
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        self.c_fc.shard(strategy_matmul=((dp, 1), (mp, 1)), strategy_bias=((dp, mp), (mp,)))
        self.gelu.shard(parallel_config)
        self.c_proj.shard(strategy_matmul=((dp, mp), (1, mp)), strategy_bias=((dp, 1), (1,)))


class ResidualAttentionBlock(nn.Cell):
    r"""
    ResidualAttentionBlock of CLIP

    Args:
        d_model (int): The dimension of features.
        n_head (int): The number of attention heads.
        layers (int): The number of transformer layers for weight initialization.
        dtype (mstype): The type of calculation, [mstype.float32, mstype.float16].
        attn_mask (Optional[ms.Tensor]): attention mask.
    """

    def __init__(self, d_model: int, n_head: int, layers: int,
                 dtype: mstype, attn_mask: Optional[ms.Tensor] = None, **kwargs):
        super(ResidualAttentionBlock, self).__init__()
        param_init_type = kwargs.get("param_init_type", mstype.float16)
        self.dtype = dtype
        self.attn = MultiheadAttention(d_model, n_head, layers, self.dtype, param_init_type)
        self.ln_1 = LayerNorm([d_model], epsilon=1e-5)

        self.mlp = MLP(layers, d_model, d_model * 4, self.dtype, param_init_type)
        self.ln_2 = LayerNorm([d_model], epsilon=1e-5)

        self.attn_mask = attn_mask
        self.add = P.Add()

    # pylint: disable=C0111
    def construct(self, input_x: ms.Tensor):
        ln_1 = self.ln_1(input_x)
        attn_tensor = self.attention(ln_1)
        input_x = self.add(input_x, attn_tensor)
        ln_2 = self.ln_2(input_x)
        mlp_2 = self.mlp(ln_2)
        return self.add(input_x, mlp_2)

    # pylint: disable=C0111
    def attention(self, input_x: ms.Tensor):
        return self.attn(input_x, self.attn_mask)

    # pylint: disable=C0111
    def shard(self, parallel_config):
        dp = parallel_config.data_parallel
        self.ln_1.layer_norm.shard(((dp, 1, 1), (1,), (1,)))
        self.ln_2.layer_norm.shard(((dp, 1, 1), (1,), (1,)))
        self.add.shard(((dp, 1, 1), (dp, 1, 1)))
        self.mlp.shard(parallel_config)
        self.attn.shard(parallel_config)


class Transformer(nn.Cell):
    r"""
    Text Transformer of CLIP

    Args:
        width (int): The dimension of input features.
        layers (int): The number of transformer layers.
        heads (int): The number of attention heads.
        attn_mask (ms.Tensor):  Attention mask.
        dtype (mstype): The type of calculation, [mstype.float32, mstype.float16].
    """

    def __init__(self, width, layers, heads, dtype, attn_mask=None, **kwargs):
        super(Transformer, self).__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.SequentialCell(
            *[ResidualAttentionBlock(width, heads, layers, dtype, attn_mask, **kwargs) for _ in range(layers)]
        )

    # pylint: disable=C0111
    def construct(self, input_x):
        return self.resblocks(input_x)

    # pylint: disable=C0111
    def shard(self, parallel_config):
        for layer in self.resblocks:
            layer.shard(parallel_config)


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class LlavaVisionEncoder(nn.Cell):
    r"""VisionTransformer Of CLIPModel

    Args: config: Llavaconfig for Llava model
    """

    def __init__(self, config: PretrainedConfig, **kwargs):
        super(LlavaVisionEncoder, self).__init__(config, **kwargs)
        self.config = config
        input_resolution = config.image_size
        patch_size = config.patch_size
        width = config.hidden_size
        layers = config.num_hidden_layers + config.vision_feature_layer + 1
        if layers <= 0:
            raise ValueError("num of layers is invalid, please set number of layers larger than 0, at least 1.")
        self.vision_feature_select_strategy = config.vision_feature_select_strategy

        heads = config.num_attention_heads
        self.dtype = config.compute_dtype
        parallel_config = config.parallel_config
        param_init_type = config.param_init_type
        self.conv1 = \
            nn.Conv2d(
                in_channels=3, out_channels=width, kernel_size=patch_size,
                stride=patch_size, has_bias=False, pad_mode="pad").to_float(param_init_type)

        scale = width ** -0.5
        self.class_embedding = \
            Parameter(scale * Tensor(np.random.normal(0, 1, size=(width))).astype(param_init_type))
        self.positional_embedding = \
            Parameter(scale * Tensor(
                np.random.normal(0, 1, size=(
                    (input_resolution // patch_size) ** 2 + 1, width))).astype(param_init_type),
                      parallel_optimizer=False)
        self.ln_pre = LayerNorm([width], epsilon=1e-5)
        self.transformer = Transformer(width, layers, heads, self.dtype, param_init_type=param_init_type,
                                       is_dynamic=config.is_dynamic)
        self.ln_post = LayerNorm([width], epsilon=1e-5)
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        if config.is_dynamic:
            self.reshape.add_prim_attr("skip_redistribution", True)

        self.position_shape = self.positional_embedding.shape
        self.expand_dims = P.ExpandDims()
        self.shape = P.Shape()
        self.cat = P.Concat(1)
        self.tile = P.Tile()
        self.add = P.Add()
        self.transpose = P.Transpose()
        self.slice = P.StridedSlice()

        if not (_get_parallel_mode() in (ParallelMode.AUTO_PARALLEL,) and _is_sharding_propagation()):
            dp = parallel_config.data_parallel
            self.conv1.conv2d.shard(((dp, 1, 1, 1), (1, 1, 1, 1)))
            self.conv1.bias_add.shard(((dp, 1, 1, 1), (1,)))
            self.tile.shard(((1, 1, 1),))
            self.transpose.shard(((dp, 1, 1),))
            self.cat.shard(((dp, 1, 1), (dp, 1, 1)))
            self.add.shard(((dp, 1, 1), (1, 1, 1)))
            self.ln_pre.layer_norm.shard(((dp, 1, 1), (1,), (1,)))
            self.slice.shard(((dp, 1, 1),))
            self.transformer.shard(parallel_config)
            self.expand_dims.shard(((1, 1),))

    def construct(self, input_x: ms.Tensor):
        r"""Construct

        Args:
            input_x (ms.Tensor): Input tensor.

        Returns:
            input_x (ms.Tensor): Output tensor.
        """

        input_x = self.conv1(input_x)
        bs, dim, seq1, seq2 = self.shape(input_x)
        input_x = self.reshape(input_x, (bs, dim, seq1 * seq2))

        input_x = self.transpose(input_x, (0, 2, 1))  #

        class_embedding = self.cast(self.tile(self.class_embedding, (input_x.shape[0], 1, 1)), self.dtype)
        input_x = self.cat([class_embedding, input_x])
        positional_embedding = self.expand_dims(self.positional_embedding, 0)
        input_x = self.add(input_x, positional_embedding)
        input_x = self.ln_pre(input_x)
        input_x = self.transformer(input_x)
        if self.vision_feature_select_strategy == 'default':
            bs, seq_length, dim = input_x.shape
            output = self.slice(input_x, (0, 1, 0), (bs, seq_length, dim), (1, 1, 1))
        elif self.vision_feature_select_strategy == "full":
            output = input_x
        else:
            raise ValueError("Please select valuable vision feature select strategy in ['full' and 'default']!")
        return output
