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
visualglm Base Model
"""
import mindspore as ms
from mindspore.common.initializer import initializer, Normal

from mindformers.models.modeling_utils import PreTrainedModel
from mindformers.models.glm import GLMConfig
from mindformers.modules.activation import GELU
from mindformers.modules.layers import LayerNorm
from qformer import BertLMHeadModel
from visualglm_glm import GLMForPreTrainingForBlip2
from visualglm_vit import ViTModelForBlip2
from visualglm_config import VisualGLMConfig


class VisualGLMPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = VisualGLMConfig
    base_model_prefix = "visualglm"


class VisualGLMBase(VisualGLMPreTrainedModel):
    """
    VisualGLM base model, all VisualGLM models inherit this class.
    """

    def init_qformer(self):
        """
        Init qformer for VisualGLM model

        Raises:
            ValueError: qformer config wrong

        Returns:
            qformer, query_tokens
        """
        qformer_config = self.config.qformer_config
        qformer_config.parallel_config = self.config.parallel_config
        qformer = BertLMHeadModel(qformer_config)
        if qformer is None:
            raise ValueError("qformer configuration is wrong. \
            please check 'qformer_config' is set in Blip2Config")
        query_tokens = ms.Parameter(initializer(
            Normal(mean=0.0, sigma=qformer_config.initializer_range),
            [1, qformer_config.query_length, qformer_config.hidden_size]))
        return qformer, query_tokens

    def init_vision_encoder(self):
        """
        init vision encoder for VisualGLM model

        Raises:
            ValueError: vit config wrong

        Returns:
            visual_encoder, ln_vision
        """
        vision_config = self.config.vision_config
        visual_encoder = None
        if vision_config is not None:
            visual_encoder = ViTModelForBlip2(vision_config)
        if visual_encoder is None:
            raise ValueError("visual_encoder configuration is wrong. \
            please check 'vision_config' is set in Blip2Config")
        for block in visual_encoder.blocks:
            mapping = block.output.mapping
            if mapping.activation_flag and isinstance(mapping.activation, GELU):
                mapping.activation = GELU(approximate=False)

        ln_vision = LayerNorm(visual_encoder.config.embed_dim)
        return visual_encoder, ln_vision

    def init_llm(self):
        """"
        init llm model for VisualGLM model

        Raises:
            ValueError: text config is wrong

        Returns:
            llm model

        """
        llm_config = self.config.text_config
        if not llm_config:
            raise ValueError("llm configuration is wrong. \
                        please check 'text_config' is set in Blip2Config")

        if isinstance(llm_config, GLMConfig):
            llm_model = GLMForPreTrainingForBlip2(llm_config)
        else:
            raise ValueError("the glm-arch is support by the blip2")
        return llm_model
