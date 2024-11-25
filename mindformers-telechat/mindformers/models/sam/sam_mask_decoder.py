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
"""SAM Mask Decoder Model"""
import math
from typing import Tuple, Type

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.modules.layers import Linear, LayerNorm

from .sam_layers import MLPBlock, LayerNorm2d
from .sam_config import MaskDecoderConfig


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class SamMaskDecoder(PreTrainedModel):
    """
    A class to predict masks given image and prompt embeddings using a transformer architecture.
    """
    config_class = MaskDecoderConfig
    base_model_prefix = "sam_mask_decoder"

    def __init__(self, config) -> None:
        super().__init__(config)
        self.transformer_dim = config.transformer_dim
        self.decoder_depth = config.decoder_depth
        self.decoder_embed_dim = config.decoder_embed_dim
        self.decoder_mlp_dim = config.decoder_mlp_dim
        self.decoder_num_heads = config.decoder_num_heads
        self.num_multimask_outputs = config.num_multimask_outputs
        self.iou_head_depth = config.iou_head_depth
        self.iou_head_hidden_dim = config.iou_head_hidden_dim
        self.layer_norm_eps = config.layer_norm_eps
        self.activation = nn.GELU

        self.compute_dtype = config.compute_dtype
        self.layernorm_compute_type = config.layernorm_compute_type
        self.softmax_compute_type = config.softmax_compute_type
        self.param_init_type = config.param_init_type

        self.transformer = TwoWayTransformer(
            depth=self.decoder_depth,
            embedding_dim=self.decoder_embed_dim,
            mlp_dim=self.decoder_mlp_dim,
            num_heads=self.decoder_num_heads,
            layer_norm_eps=self.layer_norm_eps,
            compute_dtype=self.compute_dtype,
            layernorm_compute_type=self.layernorm_compute_type,
            softmax_compute_type=self.softmax_compute_type,
            param_init_type=self.param_init_type
        )

        self.iou_token = nn.Embedding(1, self.transformer_dim)
        self.num_mask_tokens = self.num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, self.transformer_dim)

        self.output_upscaling = nn.SequentialCell(
            nn.Conv2dTranspose(self.transformer_dim,
                               self.transformer_dim // 4,
                               kernel_size=2,
                               stride=2,
                               has_bias=True),
            LayerNorm2d(self.transformer_dim // 4),
            self.activation(),
            nn.Conv2dTranspose(self.transformer_dim // 4,
                               self.transformer_dim // 8,
                               kernel_size=2,
                               stride=2,
                               has_bias=True),
            self.activation(),
        )
        self.output_hypernetworks_mlps = nn.CellList(
            [
                MLP(self.transformer_dim, self.transformer_dim, self.transformer_dim // 8, 3,
                    compute_dtype=self.compute_dtype, param_init_type=self.param_init_type)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            self.transformer_dim, self.iou_head_hidden_dim, self.num_mask_tokens, self.iou_head_depth,
            compute_dtype=self.compute_dtype, param_init_type=self.param_init_type
        )

        self.concat_0 = P.Concat(axis=0)
        self.concat_1 = P.Concat(axis=1)
        self.expand_dims = P.ExpandDims()
        self.stack = P.Stack(axis=1)

    def construct(self,
                  image_embeddings: ms.Tensor,
                  image_pe: ms.Tensor,
                  sparse_prompt_embeddings: ms.Tensor,
                  dense_prompt_embeddings: ms.Tensor,
                  multimask_output: bool) -> Tuple[ms.Tensor, ms.Tensor]:
        """
        Args:
            image_embeddings (ms.Tensor): Embeddings from the image encoder.
            image_pe (ms.Tensor): Positional encoding with the shape of image_embeddings.
            sparse_prompt_embeddings (ms.Tensor): Embeddings of the points and boxes.
            dense_prompt_embeddings (ms.Tensor): Embeddings of the mask inputs.
            multimask_output (bool): Whether to return multiple masks or a single mask.

        Returns:
            Tuple[ms.Tensor, ms.Tensor]: Batched predicted masks and batched predictions of mask quality.
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(self,
                      image_embeddings: ms.Tensor,
                      image_pe: ms.Tensor,
                      sparse_prompt_embeddings: ms.Tensor,
                      dense_prompt_embeddings: ms.Tensor) -> Tuple[ms.Tensor, ms.Tensor]:
        """
        Args:
            image_embeddings (ms.Tensor): Embeddings from the image encoder.
            image_pe (ms.Tensor): Positional encoding with the shape of image_embeddings.
            sparse_prompt_embeddings (ms.Tensor): Embeddings of the points and boxes.
            dense_prompt_embeddings (ms.Tensor): Embeddings of the mask inputs.

        Returns:
            Tuple[ms.Tensor, ms.Tensor]: Batched predicted masks and batched predictions of mask quality.
        """
        bs = sparse_prompt_embeddings.shape[0]
        # Concatenate output tokens
        output_tokens = self.concat_0([self.iou_token.embedding_table, self.mask_tokens.embedding_table])
        output_tokens = ops.broadcast_to(self.expand_dims(output_tokens, 0), (bs, -1, -1))

        if ops.any(sparse_prompt_embeddings):
            tokens = self.concat_1([output_tokens, sparse_prompt_embeddings])
        else:
            tokens = output_tokens

        # Expand per-image data in batch direction to be per-mask
        src = image_embeddings.repeat_interleave(tokens.shape[0], 0)
        src = src + dense_prompt_embeddings
        pos_src = image_pe.repeat_interleave(tokens.shape[0], 0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(0, 2, 1).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = self.stack(hyper_in_list)
        b, c, h, w = upscaled_embedding.shape
        masks = ops.matmul(hyper_in, upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


class TwoWayTransformer(nn.Cell):
    """
        Two-way transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.
    """
    def __init__(self,
                 depth: int,
                 embedding_dim: int,
                 num_heads: int,
                 mlp_dim: int,
                 activation: Type[nn.Cell] = nn.ReLU,
                 attention_downsample_rate: int = 2,
                 layer_norm_eps: float = 1.e-12,
                 compute_dtype=mstype.float16,
                 layernorm_compute_type=mstype.float32,
                 softmax_compute_type=mstype.float32,
                 param_init_type=mstype.float32) -> None:
        super().__init__()
        self.depth = depth
        self.embeding_dim = embedding_dim
        self.num_heads = num_heads
        self.mpl_dim = mlp_dim

        self.compute_dtype = compute_dtype
        self.layernorm_compute_type = layernorm_compute_type
        self.softmax_compute_type = softmax_compute_type
        self.param_init_type = param_init_type

        self.layers = nn.CellList()
        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                    layer_norm_eps=layer_norm_eps,
                    compute_dtype=self.compute_dtype,
                    layernorm_compute_type=self.layernorm_compute_type,
                    softmax_compute_type=self.softmax_compute_type,
                    param_init_type=self.param_init_type
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim,
            num_heads,
            downsample_rate=attention_downsample_rate,
            compute_dtype=self.compute_dtype,
            layernorm_compute_type=self.layernorm_compute_type,
            softmax_compute_type=self.softmax_compute_type,
            param_init_type=self.param_init_type
        )
        self.norm_final_attn = LayerNorm((embedding_dim,), eps=layer_norm_eps)

    def construct(self,
                  image_embedding: ms.Tensor,
                  image_pe: ms.Tensor,
                  point_embedding: ms.Tensor) -> Tuple[ms.Tensor, ms.Tensor]:
        """
        Apply the two-way transformer.

        Args:
            image_embedding (ms.Tensor): Image to attend to. Should have shape
                B x embedding_dim x h x w for any h and w.
            image_pe (ms.Tensor): The positional encoding to add to the image.
                Must have the same shape as image_embedding.
            point_embedding (ms.Tensor): The embedding to add to the query points.
                Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
            ms.Tensor: The processed point_embedding.
            ms.Tensor: The processed image_embedding.
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.reshape(bs, c, h * w).permute(0, 2, 1)
        image_pe = image_pe.reshape(bs, c, h * w).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayAttentionBlock(nn.Cell):
    """
        Two-way attention block consisting of self-attention and cross-attention layers.
    """
    def __init__(self,
                 embedding_dim: int,
                 num_heads: int,
                 mlp_dim: int = 2048,
                 activation: Type[nn.Cell] = nn.ReLU,
                 attention_downsample_rate: int = 2,
                 skip_first_layer_pe: bool = False,
                 layer_norm_eps: float = 1.e-12,
                 compute_dtype=mstype.float16,
                 layernorm_compute_type=mstype.float32,
                 softmax_compute_type=mstype.float32,
                 param_init_type=mstype.float32) -> None:
        super().__init__()
        self.self_attn = Attention(embedding_dim,
                                   num_heads,
                                   compute_dtype=compute_dtype,
                                   layernorm_compute_type=layernorm_compute_type,
                                   softmax_compute_type=softmax_compute_type,
                                   param_init_type=param_init_type)
        self.norm1 = LayerNorm((embedding_dim,), eps=layer_norm_eps)

        self.cross_attn_token_to_image = Attention(embedding_dim,
                                                   num_heads,
                                                   downsample_rate=attention_downsample_rate,
                                                   compute_dtype=compute_dtype,
                                                   layernorm_compute_type=layernorm_compute_type,
                                                   softmax_compute_type=softmax_compute_type,
                                                   param_init_type=param_init_type)
        self.norm2 = LayerNorm((embedding_dim,), eps=layer_norm_eps)

        self.mlp = MLPBlock(embedding_dim,
                            mlp_dim,
                            activation,
                            compute_dtype=compute_dtype,
                            param_init_type=param_init_type)
        self.norm3 = LayerNorm((embedding_dim,), eps=layer_norm_eps)

        self.norm4 = LayerNorm((embedding_dim,), eps=layer_norm_eps)
        self.cross_attn_image_to_token = Attention(embedding_dim,
                                                   num_heads,
                                                   downsample_rate=attention_downsample_rate,
                                                   compute_dtype=compute_dtype,
                                                   layernorm_compute_type=layernorm_compute_type,
                                                   softmax_compute_type=softmax_compute_type,
                                                   param_init_type=param_init_type)

        self.skip_first_layer_pe = skip_first_layer_pe

    def construct(self,
                  queries: ms.Tensor,
                  keys: ms.Tensor,
                  query_pe: ms.Tensor,
                  key_pe: ms.Tensor) -> Tuple[ms.Tensor, ms.Tensor]:
        """
        Apply the two-way attention block.

        Args:
            queries (ms.Tensor): Queries tensor.
            keys (ms.Tensor): Keys tensor.
            query_pe (ms.Tensor): Query positional encoding.
            key_pe (ms.Tensor): Key positional encoding.

        Returns:
            ms.Tensor: Processed queries tensor.
            ms.Tensor: Processed keys tensor.
        """
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class Attention(nn.Cell):
    """
    Attention layer that supports downscaling the embedding size after projection to queries, keys, and values.
    """

    def __init__(self,
                 embedding_dim: int,
                 num_heads: int,
                 downsample_rate: int = 1,
                 compute_dtype=mstype.float16,
                 layernorm_compute_type=mstype.float32,
                 softmax_compute_type=mstype.float32,
                 param_init_type=mstype.float32) -> None:
        """
        Initialize the Attention layer.

        Args:
            embedding_dim (int): The channel dimension of the embeddings.
            num_heads (int): The number of heads in the attention layer.
            downsample_rate (int): Downsample rate for the attention.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads

        self.compute_dtype = compute_dtype
        self.layernorm_compute_type = layernorm_compute_type
        self.softmax_compute_type = softmax_compute_type
        self.param_init_type = param_init_type
        if self.internal_dim % num_heads != 0:
            raise ValueError(f"num_heads must divide embedding_dim, but internal_dim got {self.internal_dim},"
                             f"num_heads got {num_heads}.")

        self.inv_norm_factor = ms.Tensor(1.0 / math.sqrt(self.internal_dim // self.num_heads))

        self.q_proj = Linear(in_channels=embedding_dim,
                             out_channels=self.internal_dim,
                             compute_dtype=compute_dtype,
                             param_init_type=param_init_type)
        self.k_proj = Linear(in_channels=embedding_dim,
                             out_channels=self.internal_dim,
                             compute_dtype=compute_dtype,
                             param_init_type=param_init_type)
        self.v_proj = Linear(in_channels=embedding_dim,
                             out_channels=self.internal_dim,
                             compute_dtype=compute_dtype,
                             param_init_type=param_init_type)
        self.out_proj = Linear(in_channels=self.internal_dim,
                               out_channels=embedding_dim,
                               compute_dtype=compute_dtype,
                               param_init_type=param_init_type)

        self.softmax = P.Softmax(axis=-1)
        self.batchmatmul_trans_b = P.BatchMatMul(transpose_b=True)
        self.batchmatmul = P.BatchMatMul()
        self.mul = P.Mul()

    def _separate_heads(self, x: ms.Tensor, num_heads: int) -> ms.Tensor:
        """
        Separate the input tensor into multiple heads.

        Args:
            x (ms.Tensor): Input tensor.
            num_heads (int): Number of heads to separate into.

        Returns:
            ms.Tensor: Tensor with dimensions reshaped for separate heads.
        """
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(0, 2, 1, 3)

    def _recombine_heads(self, x: ms.Tensor) -> ms.Tensor:
        """
        Recombine the tensor with multiple heads into a single tensor.

        Args:
            x (ms.Tensor): Input tensor with separate heads.

        Returns:
            ms.Tensor: Recombined tensor.
        """
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(b, n_tokens, n_heads * c_per_head)

    def construct(self,
                  q: ms.Tensor,
                  k: ms.Tensor,
                  v: ms.Tensor) -> ms.Tensor:
        """
        Apply the attention mechanism.

        Args:
            q (ms.Tensor): Queries tensor.
            k (ms.Tensor): Keys tensor.
            v (ms.Tensor): Values tensor.

        Returns:
            ms.Tensor: Output tensor after attention.
        """
        ori_type = k.dtype
        # Input projections
        q = self.q_proj(q).astype(self.compute_dtype)
        k = self.k_proj(k).astype(self.compute_dtype)
        v = self.v_proj(v).astype(self.compute_dtype)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        attn = self.batchmatmul_trans_b(q, k)
        attn = self.mul(attn, self.inv_norm_factor)
        attn = self.softmax(attn.astype(self.softmax_compute_type))

        # Get output
        out = self.batchmatmul(attn.astype(self.compute_dtype), v)
        out = self._recombine_heads(out)
        out = self.out_proj(out).astype(ori_type)

        return out


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Cell):
    """
        Multi-Layer Perceptron (MLP) class definition.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int,
                 sigmoid_output: bool = False,
                 compute_dtype=mstype.float16,
                 param_init_type=mstype.float32) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.CellList([
            Linear(in_channels=n,
                   out_channels=k,
                   compute_dtype=compute_dtype,
                   param_init_type=param_init_type)
            for n, k in zip([input_dim] + h, h + [output_dim])
        ])
        self.sigmoid_output = sigmoid_output

        self.relu = P.ReLU()
        self.sigmoid = P.Sigmoid()

    def construct(self, x):
        """
        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor of the MLP.
        """
        for i, layer in enumerate(self.layers):
            x = self.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = self.sigmoid(x)
        return x
