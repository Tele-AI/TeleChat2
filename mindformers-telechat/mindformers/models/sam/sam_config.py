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
"""SAM Config API."""

from typing import Union, Tuple

from mindspore._checkparam import args_type_check
import mindspore.common.dtype as mstype

from mindformers.modules.transformer.transformer import TransformerOpParallelConfig, default_transformer_config, \
    default_moe_config
from mindformers.modules.transformer.moe import MoEConfig
from mindformers.mindformer_book import MindFormerBook
from mindformers.models.configuration_utils import PretrainedConfig
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class SamConfig(PretrainedConfig):
    """
    Configuration class for the SAM model.
    """

    model_type = "sam"
    _support_list = MindFormerBook.get_config_support_list()['sam']

    def __init__(self,
                 image_encoder,
                 prompt_config,
                 decoder_config,
                 checkpoint_name_or_path: str = '',
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.image_encoder = image_encoder
        self.prompt_config = prompt_config
        self.decoder_config = decoder_config
        self.checkpoint_name_or_path = checkpoint_name_or_path


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class ImageEncoderConfig(PretrainedConfig):
    """
    Configuration class for the image encoder of the SAM model.
    """

    @args_type_check(parallel_config=(dict, TransformerOpParallelConfig),
                     moe_config=(dict, MoEConfig))
    def __init__(self,
                 img_size: int = 1024,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: int = 4,
                 out_chans: int = 256,
                 qkv_bias: bool = True,
                 use_abs_pos: bool = True,
                 use_rel_pos: bool = True,  # 仅支持默认False
                 rel_pos_zero_init: bool = True,
                 window_size: int = 14,
                 global_attn_indexes: tuple = (2, 5, 8, 11),
                 checkpoint_name_or_path: str = '',
                 compute_dtype: str = "float16",
                 layernorm_compute_type: str = "float32",
                 softmax_compute_type: str = "float32",
                 param_init_type: str = "float32",
                 parallel_config: Union[dict, TransformerOpParallelConfig] = default_transformer_config,
                 moe_config: Union[dict, MoEConfig] = default_moe_config,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if isinstance(parallel_config, dict):
            parallel_config = TransformerOpParallelConfig(**parallel_config)
        if isinstance(moe_config, dict):
            moe_config = MoEConfig(**moe_config)
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.out_chans = out_chans
        self.qkv_bias = qkv_bias
        self.use_abs_pos = use_abs_pos
        self.use_rel_pos = use_rel_pos  # 仅支持默认False
        self.rel_pos_zero_init = rel_pos_zero_init
        self.window_size = window_size
        self.global_attn_indexes = global_attn_indexes

        self.compute_dtype = mstype.float32 if compute_dtype == "float32" else mstype.float16
        self.layernorm_compute_type = mstype.float32 if layernorm_compute_type == "float32" else mstype.float16
        self.softmax_compute_type = mstype.float32 if softmax_compute_type == "float32" else mstype.float16
        self.param_init_type = mstype.float32 if param_init_type == "float32" else mstype.float16
        self.parallel_config = parallel_config
        self.moe_config = moe_config

        self.checkpoint_name_or_path = checkpoint_name_or_path


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class PromptEncoderConfig(PretrainedConfig):
    """
    Configuration for the prompt encoder of the SAM model.
    """

    @args_type_check(parallel_config=(dict, TransformerOpParallelConfig),
                     moe_config=(dict, MoEConfig))
    def __init__(self,
                 prompt_embed_dim: int = 256,
                 image_embedding_size: Union[int, Tuple[int]] = (64, 64),
                 input_image_size: Union[int, Tuple[int]] = (1024, 1024),
                 mask_in_chans: int = 16,
                 checkpoint_name_or_path: str = '',
                 compute_dtype: str = "float16",
                 parallel_config: Union[dict, TransformerOpParallelConfig] = default_transformer_config,
                 moe_config: Union[dict, MoEConfig] = default_moe_config,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if isinstance(parallel_config, dict):
            parallel_config = TransformerOpParallelConfig(**parallel_config)
        if isinstance(moe_config, dict):
            moe_config = MoEConfig(**moe_config)
        self.prompt_embed_dim = prompt_embed_dim
        self.image_embedding_size = image_embedding_size
        self.input_image_size = input_image_size
        self.mask_in_chans = mask_in_chans

        self.compute_dtype = mstype.float32 if compute_dtype == "float32" else mstype.float16
        self.parallel_config = parallel_config
        self.moe_config = moe_config

        self.checkpoint_name_or_path = checkpoint_name_or_path


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class MaskDecoderConfig(PretrainedConfig):
    """
    Configuration for the mask decoder of the SAM model.
    """

    @args_type_check(parallel_config=(dict, TransformerOpParallelConfig),
                     moe_config=(dict, MoEConfig))
    def __init__(self,
                 num_multimask_outputs: int = 3,
                 decoder_depth: int = 2,
                 decoder_embed_dim: int = 256,
                 decoder_mlp_dim: int = 2048,
                 decoder_num_heads: int = 8,
                 transformer_dim: int = 256,
                 iou_head_depth: int = 3,
                 iou_head_hidden_dim: int = 256,
                 checkpoint_name_or_path: str = '',
                 compute_dtype: str = "float16",
                 layernorm_compute_type: str = "float32",
                 softmax_compute_type: str = "float32",
                 param_init_type: str = "float32",
                 parallel_config: Union[dict, TransformerOpParallelConfig] = default_transformer_config,
                 moe_config: Union[dict, MoEConfig] = default_moe_config,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if isinstance(parallel_config, dict):
            parallel_config = TransformerOpParallelConfig(**parallel_config)
        if isinstance(moe_config, dict):
            moe_config = MoEConfig(**moe_config)
        self.num_multimask_outputs = num_multimask_outputs
        # transformer:
        self.decoder_depth = decoder_depth
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_mlp_dim = decoder_mlp_dim
        self.decoder_num_heads = decoder_num_heads
        self.transformer_dim = transformer_dim
        self.iou_head_depth = iou_head_depth
        self.iou_head_hidden_dim = iou_head_hidden_dim

        self.compute_dtype = mstype.float32 if compute_dtype == "float32" else mstype.float16
        self.layernorm_compute_type = mstype.float32 if layernorm_compute_type == "float32" else mstype.float16
        self.softmax_compute_type = mstype.float32 if softmax_compute_type == "float32" else mstype.float16
        self.param_init_type = mstype.float32 if param_init_type == "float32" else mstype.float16
        self.parallel_config = parallel_config
        self.moe_config = moe_config

        self.checkpoint_name_or_path = checkpoint_name_or_path
