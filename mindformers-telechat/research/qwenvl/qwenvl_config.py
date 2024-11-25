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
"""QwenVL Config API"""

from mindformers import TransformerOpParallelConfig, MindFormerRegister, \
    MindFormerModuleType, PretrainedConfig
from mindformers.core.parallel_config import default_parallel_config
from mindformers.models.utils import convert_mstype


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class QwenVLVisionConfig(PretrainedConfig):
    r"""
    Config For QwenVL Vision Module

    Args:
        hidden_size (int): Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (int): Dimensionality of the "intermediate"
            (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (int): Number of hidden layers in the Transformer encoder.
        num_attention_heads (int): Number of attention heads for
            each attention layer in the Transformer encoder.
        image_size (int): The size (resolution) of each image.
        patch_size (int): The size (resolution) of each patch.
        hidden_act (str): The non-linear activation function
            (function or string) in the encoder and pooler. Only "quick_gelu" supported currently.
        dropout (float): The dropout probability for all fully connected
            layers in the embeddings, encoder, and pooler.
        attention_dropout (float): The dropout ratio for the attention probabilities.
        initializer_range (float): The standard deviation of the
            truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (float): A factor for initializing all weight matrices
            (should be kept to 1, used internally for initialization testing).
        parallel_config(TransformerOpParallelConfig): The parallel configure. Default `default_transformer_config`,
            an instance of `TransformerOpParallelConfig` with default args.
        output_dim(int): The output hidden dim of vision transformer
        use_flash_attention(bool): Whether enable flash attention ops, default False.
        enable_fa_opt(bool): Whether to enable 128-alignment of q, k, and v dimensions during flash attention
            calculation, default False.
    Returns:
        Class, QwenVLVisionConfig
    """

    def __init__(self, hidden_size: int = 1664,
                 intermediate_size: int = 8192,
                 num_hidden_layers: int = 48,
                 num_attention_heads: int = 16,
                 image_size: int = 448,
                 patch_size: int = 14,
                 dropout: float = 0.0,
                 attention_dropout: float = 0.0,
                 initializer_range: float = 0.02,
                 initializer_factor: float = 1.0,
                 parallel_config: TransformerOpParallelConfig = default_parallel_config,
                 output_dim: int = 4096,
                 use_flash_attention: bool = False,
                 enable_fa_opt: bool = False,
                 dtype: str = "float32",
                 compute_dtype: str = "float16",
                 softmax_compute_type: str = "float32",
                 param_init_type: str = "float16",
                 gelu_dtype: str = "float32",
                 **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.image_size = image_size
        self.patch_size = patch_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.parallel_config = parallel_config
        self.output_dim = output_dim
        self.use_flash_attention = use_flash_attention
        self.enable_fa_opt = enable_fa_opt
        self.dtype = convert_mstype(dtype)
        self.compute_dtype = convert_mstype(compute_dtype)
        self.softmax_compute_type = convert_mstype(softmax_compute_type)
        self.param_init_type = convert_mstype(param_init_type)
        self.gelu_dtype = convert_mstype(gelu_dtype)


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class QwenVLConfig(PretrainedConfig):
    r"""
    Config For QwenVL Vision Module

    Args:
        vision_model (dict): vision model config.
        llm_model (dict): llm model config.
        num_queries (int): num of query tokens.
        proj_output_dim (int): the output dim after projection in visual model.
        image_start_id (int): token id of image_start.
        image_pad_id (int): token id of image_pad.
        freeze_vision (bool): Whether to freeze visual model.
        freeze_llm (bool): Whether to freeze llm model.
        checkpoint_name_or_path (str): checkpoint path or name used to load to the network.
        use_past (bool): Whether the model should use the past last key/values attentions
            (if applicable to the model) to speed up decoding.
        parallel_config(TransformerOpParallelConfig):
            The parallel configure. Default `default_transformer_config`,
            an instance of `TransformerOpParallelConfig` with default args.

    Returns:
        Class, QwenVLConfig
    """
    def __init__(self, vision_model: dict,
                 llm_model: dict,
                 num_queries: int = 256,
                 proj_output_dim: int = 4096,
                 image_start_id: int = 151857,
                 image_pad_id: int = 151859,
                 freeze_vision: bool = False,
                 freeze_resampler: bool = False,
                 freeze_llm: bool = False,
                 checkpoint_name_or_path: str = None,
                 use_past: bool = False,
                 compute_dtype: str = None,
                 softmax_compute_type: str = None,
                 param_init_type: str = None,
                 parallel_config: TransformerOpParallelConfig = default_parallel_config,
                 **kwargs):
        super().__init__(**kwargs)

        self.vision_model = vision_model
        self.llm_model = llm_model

        self.num_queries = num_queries
        self.proj_output_dim = proj_output_dim
        self.image_start_id = image_start_id
        self.image_pad_id = image_pad_id

        self.freeze_vision = freeze_vision
        self.freeze_resampler = freeze_resampler
        self.freeze_llm = freeze_llm
        self.checkpoint_name_or_path = checkpoint_name_or_path
        self.use_past = use_past

        self.parallel_config = parallel_config

        if compute_dtype is not None:
            self.vision_model.model_config.compute_dtype = compute_dtype
            self.llm_model.model_config.compute_dtype = compute_dtype

        if softmax_compute_type is not None:
            self.vision_model.model_config.softmax_compute_type = softmax_compute_type
            self.llm_model.model_config.softmax_compute_type = softmax_compute_type

        if param_init_type is not None:
            self.vision_model.model_config.param_init_type = param_init_type
            self.llm_model.model_config.param_init_type = param_init_type

        self.vision_model.model_config.parallel_config = parallel_config
        self.llm_model.model_config.parallel_config = parallel_config

        llm_model_config = llm_model["model_config"]
        self.pad_token_id = llm_model_config.pad_token_id
        self.eos_token_id = llm_model_config.eos_token_id
        self.ignore_token_id = llm_model_config.ignore_token_id

        self.vocab_size = llm_model_config.vocab_size
        self.seq_length = llm_model_config.seq_length
        self.repetition_penalty = llm_model_config.repetition_penalty
        self.max_decode_length = llm_model_config.max_decode_length
        self.top_k = llm_model_config.top_k
        self.top_p = llm_model_config.top_p
        self.do_sample = llm_model_config.do_sample
