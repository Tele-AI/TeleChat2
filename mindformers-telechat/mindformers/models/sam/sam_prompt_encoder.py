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
"""SAM Prompt Encoder"""
from typing import Optional, Tuple

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as np
import mindspore.common.dtype as mstype
from mindspore import Parameter
from mindspore.ops import operations as P

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.models.modeling_utils import PreTrainedModel

from .sam_layers import LayerNorm2d
from .sam_config import PromptEncoderConfig


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class SamPromptEncoder(PreTrainedModel):
    """
    Encodes prompts for input to SAM's mask decoder.
    """
    config_class = PromptEncoderConfig
    base_model_prefix = "sam_prompt_encoder"

    def __init__(self, config) -> None:
        super().__init__(config)

        self.embed_dim = config.prompt_embed_dim
        self.input_image_size = config.input_image_size
        self.image_embedding_size = config.image_embedding_size
        self.mask_in_chans = config.mask_in_chans
        self.activation = nn.GELU
        self.pe_layer = PositionEmbeddingRandom(self.embed_dim // 2)

        self.num_point_embeddings = 4  # pos/neg point + 2 box corners
        point_embeddings = [nn.Embedding(1, self.embed_dim) for i in range(self.num_point_embeddings)]
        self.point_embeddings = nn.CellList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, self.embed_dim)

        self.mask_input_size = (4 * self.image_embedding_size[0], 4 * self.image_embedding_size[1])
        self.mask_downscaling = nn.SequentialCell(
            nn.Conv2d(1, self.mask_in_chans // 4, kernel_size=2, stride=2, has_bias=True),
            LayerNorm2d(self.mask_in_chans // 4),
            self.activation(),
            nn.Conv2d(self.mask_in_chans // 4, self.mask_in_chans, kernel_size=2, stride=2, has_bias=True),
            LayerNorm2d(self.mask_in_chans),
            self.activation(),
            nn.Conv2d(self.mask_in_chans, self.embed_dim, kernel_size=1, has_bias=True),
        )
        self.no_mask_embed = nn.Embedding(1, self.embed_dim)

        self.zeros = P.Zeros()
        self.ones = P.Ones()
        self.expand_dims = P.ExpandDims()
        self.concat = P.Concat(axis=1)
        self.tile = P.Tile()

    def get_dense_pe(self) -> ms.Tensor:
        """
        Get the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
            ms.Tensor: Positional encoding with shape
                1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.expand_dims(self.pe_layer(self.image_embedding_size), 0)

    def _embed_points(self, points: ms.Tensor, labels: ms.Tensor, pad: bool) -> ms.Tensor:
        """Embeds point prompts."""
        bs = points.shape[0]
        points = points + 0.5 # Shift to center of pixel
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        if pad:
            not_a_point_embed = self.tile(self.expand_dims(self.not_a_point_embed.embedding_table, 0), (bs, 1, 1))
            point_embedding = self.concat([point_embedding, not_a_point_embed])
            labels = self.concat([labels, -self.ones((bs, 1), mstype.int32)])

        mask_neg = (labels == 0).reshape(bs, -1, 1)
        point_embedding += np.where(mask_neg, self.point_embeddings[0].embedding_table,\
                                    np.zeros_like(self.point_embeddings[0].embedding_table))

        mask_pos = (labels == 1).reshape(bs, -1, 1)
        point_embedding += np.where(mask_pos, self.point_embeddings[1].embedding_table,\
                                    np.zeros_like(self.point_embeddings[1].embedding_table))
        return point_embedding

    def _embed_boxes(self, boxes: ms.Tensor) -> ms.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5 # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].embedding_table
        corner_embedding[:, 1, :] += self.point_embeddings[3].embedding_table
        return corner_embedding

    def _embed_masks(self, masks: ms.Tensor) -> ms.Tensor:
        """Embeds mask inputs."""
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _get_batch_size(self,
                        points: Optional[ms.Tensor],
                        boxes: Optional[ms.Tensor],
                        masks: Optional[ms.Tensor]) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points is not None:
            return points.shape[0]
        if boxes is not None:
            return boxes.shape[0]
        if masks is not None:
            return masks.shape[0]
        return 1

    def construct(self,
                  point_coords: Optional[ms.Tensor],
                  point_labels: Optional[ms.Tensor],
                  boxes: Optional[ms.Tensor],
                  mask_inputs: Optional[ms.Tensor]) -> Tuple[ms.Tensor, ms.Tensor]:
        """
        Args:
            point_coords (ms.Tensor or none): Point coordinates to embed.
            point_labels (ms.Tensor or none): Point labels to embed.
            boxes (ms.Tensor or none): Boxes to embed.
            mask_inputs (ms.Tensor or none): Masks to embed.

        Returns:
            Tuple[ms.Tensor, ms.Tensor]: Sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points and boxes.
            Dense embeddings for the masks, in the shape Bx(embed_dim)x(embed_H)x(embed_W).
        """
        bs = self._get_batch_size(point_coords, boxes, mask_inputs)
        sparse_embeddings = np.empty((bs, 1, self.embed_dim))
        if point_coords is not None:
            point_embeddings = self._embed_points(point_coords, point_labels, pad=(boxes is None))
            sparse_embeddings = point_embeddings
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = box_embeddings
        if mask_inputs is not None:
            dense_embeddings = self._embed_masks(mask_inputs)
        else:
            dense_embeddings = ops.broadcast_to(self.no_mask_embed.embedding_table.reshape(1, -1, 1, 1),\
                                                (bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]))
        return sparse_embeddings, dense_embeddings


class PositionEmbeddingRandom(nn.Cell):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self,
                 num_pos_feats: int = 64,
                 scale: Optional[float] = None,
                 compute_dtype=mstype.float16) -> None:
        """
        Initialize the PositionEmbeddingRandom.

        Args:
            num_pos_feats (int): Number of positional encoding features.
            scale (float): Scale factor for the positional encoding Gaussian matrix.
        """
        super().__init__()
        self.compute_dtype = compute_dtype
        if scale is None or scale <= 0.0:
            scale = 1.0
        pe_gaussian_matrix = scale * P.StandardNormal()((2, num_pos_feats)).astype(self.compute_dtype)
        self.positional_encoding_gaussian_matrix = Parameter(pe_gaussian_matrix,\
                                                             name="positional_encoding_gaussian_matrix")

        self.cast = P.Cast()
        self.ones = P.Ones()
        self.sin = P.Sin()
        self.cos = P.Cos()
        self.concat = P.Concat(axis=-1)
        self.stack = P.Stack(axis=-1)

    def _pe_encoding(self, coords: ms.Tensor) -> ms.Tensor:
        """
        Positionally encode points that are normalized to [0,1].

        Args:
            coords (ms.Tensor): Coordinates normalized to [0, 1].

        Returns:
            ms.Tensor: Positional encoding for the given coordinates.
        """
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        ori_type = coords.dtype
        # here must use ops.matmul, because 'x', 'y' have different dimension
        coords = ops.matmul(coords.astype(self.compute_dtype), self.positional_encoding_gaussian_matrix)
        coords = self.cast(coords, ori_type)
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return self.concat([self.sin(coords), self.cos(coords)])

    def construct(self, size: Tuple[int, int]) -> ms.Tensor:
        """
        Generate positional encoding for a grid of the specified size.

        Args:
            size (Tuple[int, int]): Grid size (H, W).

        Returns:
            ms.Tensor: Positional encoding for the grid.
        """
        h, w = size
        grid = self.ones((h, w), ms.float32)
        y_embed = grid.cumsum(axis=0) - 0.5
        x_embed = grid.cumsum(axis=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(self.stack([x_embed, y_embed]))
        return pe.transpose(2, 0, 1) # C x H x W

    def forward_with_coords(self, coords_input: ms.Tensor, image_size: Tuple[int, int]) -> ms.Tensor:
        """
        Positionally encode points that are not normalized to [0,1].

        Args:
            coords_input (ms.Tensor): Coordinates not normalized to [0, 1].
            image_size (Tuple[int, int]): Image size (H, W).

        Returns:
            ms.Tensor: Positional encoding for the given coordinates.
        """
        coords = coords_input
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords)
