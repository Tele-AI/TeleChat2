# Copyright 2022 Huawei Technologies Co., Ltd
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
CLIPModel
"""
from typing import Optional, Union

import numpy as np
import mindspore as ms

from mindspore import nn
from mindspore.ops import functional as F
from mindspore.common.initializer import Normal, initializer
from mindspore import Parameter, Tensor
import mindspore.ops as ops

from mindformers.version_control import get_norm
from mindformers.models.modeling_utils import PreTrainedModel
from ...mindformer_book import MindFormerBook
from .clip_modules import VisionTransformer, Transformer, LayerNorm
from .clip_config import CLIPConfig
from ...tools.register import MindFormerRegister, MindFormerModuleType


class ClipPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = CLIPConfig
    base_model_prefix = "clip"


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class CLIPModel(ClipPreTrainedModel):
    r"""CLIPModel.
    The supported model name could be selected from CLIPModel.show_support_list().

    Args:
        config (CLIPConfig): The config of clip model, which could be obtained by CLIPConfig class.

    Examples:
        >>> from mindformers import CLIPModel
        >>> model = CLIPModel.from_pretrained('clip_vit_b_32')
        >>> type(model)
        <class 'mindformers.models.clip.clip.CLIPModel'>
    """
    _support_list = MindFormerBook.get_model_support_list()['clip']

    def __init__(self, config: CLIPConfig):
        super(CLIPModel, self).__init__(config)
        self.dtype = self.get_dtype(config.dtype)
        self.cross_entropy = nn.SoftmaxCrossEntropyWithLogits(reduction="mean", sparse=True)

        self.max_position_embeddings = config.text_config.max_position_embeddings
        self.visual = VisionTransformer(
            input_resolution=config.vision_config.image_size,
            patch_size=config.vision_config.patch_size,
            width=config.vision_config.hidden_size,
            layers=config.vision_config.num_hidden_layers,
            heads=config.vision_config.num_attention_heads,
            output_dim=config.projection_dim,
            dtype=self.dtype
        )

        self.transformer = Transformer(
            width=config.text_config.hidden_size,
            layers=config.text_config.num_hidden_layers,
            heads=config.text_config.num_attention_heads,
            dtype=self.dtype,
            attn_mask=self.build_attention_mask()
        )

        self.token_embedding = \
            nn.Embedding(config.text_config.vocab_size, config.text_config.hidden_size,
                         embedding_table=Normal(mean=0.0, sigma=0.02))
        self.positional_embedding = Parameter(initializer(
            Normal(mean=0.0, sigma=0.01), [config.text_config.max_position_embeddings,
                                           config.text_config.hidden_size]))
        self.ln_final = LayerNorm([config.text_config.hidden_size])

        self.text_projection = Parameter(initializer(
            Normal(mean=0.0, sigma=config.text_config.hidden_size ** -0.5),
            [config.text_config.hidden_size, config.projection_dim], ms.float32))
        self.logit_scale = Parameter(Tensor(np.log(1 / 0.07)).astype(ms.float32))
        self.exp = ops.Exp()
        self.norm = get_norm()
        self.load_checkpoint(config)

    def get_dtype(self, dtype: str):
        """Get_dtype"""
        if dtype == "float16":
            return ms.float16
        if dtype == "float32":
            return ms.float32
        raise TypeError("unsupported data type.")

    def construct(self, image: ms.Tensor, text: ms.Tensor,
                  label: Optional[Union[ms.Tensor, np.ndarray]] = None,
                  input_ids: Optional[ms.Tensor] = None,
                  pixel_values: Optional[ms.Tensor] = None):
        r"""Construct

        Args:
            image (Tensor): A image tensor processed by image_processor.
            text (Tensor): A text id tensor processed by tokenizer.
            input_ids (Optional[ms.Tensor]): Equal to "text",
                if "input_ids" is set, "text" is useless.
            pixel_values (Optional[ms.Tensor]): Equal to "image",
                if "pixel_values" is set, "image" is useless.
            label (Optional[Union[ms.Tensor, np.ndarray]]): The classification label.

        Returns:
            if not self.trainining:
                if label is None:
                    logits_per_image: Similarity between image and text.
                    logits_per_text: Similarity between text and image.
                else:
                    logits_per_image: Similarity between image and text.
                    label: The classification label.
            else:
                loss: Constructive language image pretraining loss.
        """
        if pixel_values is not None:
            image = pixel_values

        if input_ids is not None:
            text = input_ids

        if len(text.shape) == 3:
            text = text[0].squeeze()

        image_features = self.get_image_features(image)
        text_features = self. get_text_features(text)

        image_features = image_features / self.norm(image_features, dim=1, keepdim=True)
        text_features = text_features / self.norm(text_features, dim=1, keepdim=True)

        logit_scale = self.exp(self.logit_scale)

        if not self.training:
            if label is None:
                logits_per_image = ops.matmul(logit_scale * image_features, text_features.T)
                logits_per_text = logits_per_image.T
                return logits_per_image, logits_per_text

            logits_per_image = ops.matmul(logit_scale * image_features, text_features.T)
            return logits_per_image, label

        logits = ops.matmul(logit_scale * image_features, text_features.T)
        batch_size, _ = F.shape(logits)

        labels = ms.Tensor(np.arange(batch_size))

        images_loss = self.cross_entropy(logits, labels)
        texts_loss = self.cross_entropy(logits.T, labels)
        loss = (images_loss + texts_loss) / 2
        return loss

    def build_attention_mask(self):
        """Build_attention_mask"""
        mask = np.ones((self.max_position_embeddings, self.max_position_embeddings))
        mask = np.triu(mask * float("-inf"), k=1)
        return Tensor(mask).astype(self.dtype)

    def get_image_features(self, image: ms.Tensor, pixel_values: Optional[ms.Tensor] = None):
        r"""Get_image_features

        Args:
            image (ms.Tensor): A image tensor processed by image_processor.
            pixel_values (Optional[ms.Tensor]): Equal to "image",
                if "pixel_values" is set, "image" is useless.

        Returns:
            Image feature.

        Examples:
            >>> import numpy as np
            >>> from mindformers import CLIPModel, CLIPProcessor
            >>> processor = CLIPProcessor.from_pretrained('clip_vit_b_32')
            >>> model = CLIPModel.from_pretrained('clip_vit_b_32')
            >>> fake_image_batch = np.random.random((5, 3, 578, 213))
            >>> model.get_image_features(processor.image_processor(fake_image_batch))
                Tensor(shape=[5, 512], dtype=Float32, value=
                [[-1.50102973e-001, -2.63687313e-001, -5.65953791e-001 ... -2.93511450e-001],
                 [-1.50103331e-001, -2.63622820e-001, -5.65623760e-001 ... -2.93337226e-001],
                 [-1.50102973e-001, -2.63687313e-001, -5.65953791e-001 ... -2.93511450e-001],
                 [-1.49712294e-001, -2.64100820e-001, -5.65740824e-001 ... -2.93599486e-001],
                 [-1.50102973e-001, -2.63687313e-001, -5.65953791e-001 ... -2.93511450e-001]])
        """
        if pixel_values is not None:
            image = pixel_values

        image = image.astype(self.dtype)
        return self.visual(image)

    def get_text_features(self, text: ms.Tensor, input_ids: Optional[ms.Tensor] = None):
        r"""Get_text_features

        Args:
            text (ms.Tensor): A text id tensor processed by tokenizer.
            input_ids (Optional[ms.Tensor]): Equal to "text",
                if "input_ids" is set, "text" is useless.

        Returns:
            Text feature.

        Examples:
            >>> from mindformers import CLIPModel, CLIPProcessor
            >>> processor = CLIPProcessor.from_pretrained('clip_vit_b_32')
            >>> model = CLIPModel.from_pretrained('clip_vit_b_32')
            >>> fake_text_batch = ["a boy", "a girl", "a women", "a men"]
            >>> text = processor.tokenizer(
            ...    fake_text_batch, max_length=77, padding="max_length", return_tensors="ms"
            ...    )["input_ids"]
            >>> model.get_text_features(text)
                Tensor(shape=[4, 512], dtype=Float32, value=
                [[6.03631809e-002, 1.79528534e-001, ... -2.23753393e-001, 1.42413378e-002],
                [1.28974199e-001, 7.46373609e-002, ...  -3.68579805e-001, 1.53980583e-001],
                [9.89909172e-002, 2.01410800e-002, ...  -2.54495114e-001, 7.68117979e-002],
                [3.16975415e-002, 2.26992741e-001, ... -5.22942394e-002, 1.98922127e-001]])
        """
        if input_ids is not None:
            text = input_ids

        text_ = self.token_embedding(text).astype(self.dtype)
        text_ = ops.Add()(text_, self.positional_embedding).astype(self.dtype)
        text_ = text_.transpose(1, 0, 2)
        text_ = self.transformer(text_)
        text_ = text_.transpose(1, 0, 2)
        text_ = self.ln_final(text_).astype(ms.float32)

        text_ = ops.MatMul()(
            text_[ms.numpy.arange(text_.shape[0]), text.argmax(-1)], self.text_projection)
        return text_
