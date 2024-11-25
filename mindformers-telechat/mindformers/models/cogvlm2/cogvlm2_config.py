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
"""CogVLM2 Config API."""
from mindformers.models.configuration_utils import PretrainedConfig
from mindformers.modules.transformer import TransformerOpParallelConfig
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.core.parallel_config import default_parallel_config

__all__ = ['CogVLM2Config']


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class CogVLM2Config(PretrainedConfig):
    r"""
    Config For CogVLM2 Vision Module

    Args:
        vision_model (PretrainedConfig): vision model config.
        llm_model (PretrainedConfig): llm model config.
        is_video (bool): Whether input is the video.
        use_past (bool): Whether the model should use the past last key/values attentions
            (if applicable to the model) to speed up decoding.
        is_dynamic (bool): Whether the model should use dynamic inputs.
        num_queries (int): num of visual query tokens.
        proj_output_dim (int): the output dim after projection in visual model.
        image_start_id (int): token id of image_start.
        image_pad_id (int): token id of image_pad.
        parallel_config(TransformerOpParallelConfig):
            The parallel configure. Default `default_transformer_config`,
            an instance of `TransformerOpParallelConfig` with default args.

    Returns:
        Class, CogVLM2Config
    """
    def __init__(self,
                 vision_model: PretrainedConfig,
                 llm_model: PretrainedConfig,
                 freeze_vision: bool = False,
                 freeze_adapter: bool = False,
                 freeze_llm: bool = False,
                 is_video: bool = False,
                 use_past: bool = False,
                 is_dynamic: bool = False,
                 num_queries: int = 66,
                 proj_output_dim: int = 4096,
                 image_start_id: int = 151857,
                 image_pad_id: int = 151859,
                 video_downsample: int = 1,
                 batch_size: int = 1,
                 parallel_config: TransformerOpParallelConfig = default_parallel_config,
                 **kwargs):
        super().__init__(**kwargs)

        if isinstance(parallel_config, dict):
            parallel_config = TransformerOpParallelConfig(**parallel_config)
        self.batch_size = batch_size

        self.vision_model = vision_model
        self.llm_model = llm_model

        self.freeze_vision = freeze_vision
        self.freeze_adapter = freeze_adapter
        self.freeze_llm = freeze_llm

        self.num_queries = num_queries
        self.proj_output_dim = proj_output_dim
        self.image_start_id = image_start_id
        self.image_pad_id = image_pad_id

        self.is_video = is_video
        self.video_downsample = video_downsample

        self.use_past = use_past
        self.is_dynamic = is_dynamic
        self.llm_model.model_config.use_past = use_past
        self.llm_model.model_config.is_dynamic = is_dynamic

        self.parallel_config = parallel_config
        self.vision_model.model_config.parallel_config = parallel_config
        self.llm_model.model_config.parallel_config = parallel_config

        llm_model_config = llm_model.get("model_config")
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
