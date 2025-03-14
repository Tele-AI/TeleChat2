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
from mindformers.tools.utils import calculate_pipeline_stage
from mindformers.tools.register.config import DictConfig
from mindformers.models.llama import LlamaConfig
from mindformers.models.eva02 import EVA02Config

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
                 vision_model: DictConfig = None,
                 llm_model: DictConfig = None,
                 layers_per_stage: list = None,
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

        if vision_model is None:
            model_config = EVA02Config()
            model_config.type = model_config.__class__.__name__
            vision_model = DictConfig(arch='EVAModel', model_config=model_config)
        if llm_model is None:
            model_config = LlamaConfig()
            model_config.type = model_config.__class__.__name__
            llm_model = DictConfig(arch='CogVLM2VideoLM', model_config=model_config)

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
        self.layers_per_stage = layers_per_stage
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

        if self.layers_per_stage is not None:
            if len(self.layers_per_stage) != parallel_config.pipeline_stage:
                raise ValueError("The length of layers_per_stage must be equal to pipeline_stage.")
            model_layers = []
            model_layers.append(self.vision_model.model_config.num_hidden_layers)
            model_layers.append(llm_model_config.num_layers)

            pipeline_stage = calculate_pipeline_stage(self.layers_per_stage, model_layers)
            stage_index = 0
            self.vision_model.model_config.pipeline_stage = pipeline_stage[stage_index]
            self.vision_model.model_config.start_stage = pipeline_stage[stage_index]['start_stage']
            self.vision_model.model_config.stage_num = pipeline_stage[stage_index]['stage_num']
            self.vision_model.model_config.offset = pipeline_stage[stage_index]['offset']
            stage_index += 1
            self.llm_model.model_config.pipeline_stage = pipeline_stage[stage_index]
            self.llm_model.model_config.start_stage = pipeline_stage[stage_index]['start_stage']
            self.llm_model.model_config.stage_num = pipeline_stage[stage_index]['stage_num']
            self.llm_model.model_config.offset = pipeline_stage[stage_index]['offset']
        else:
            self.vision_model.model_config.start_stage = 0
            self.vision_model.model_config.stage_num = 1
            self.llm_model.model_config.pipeline_stage = {}
            self.llm_model.model_config.pipeline_stage['start_stage'] = 1
            self.llm_model.model_config.pipeline_stage['stage_num'] = parallel_config.pipeline_stage - 1
            self.llm_model.model_config.pipeline_stage['offset'] = 0
            self.llm_model.model_config.start_stage = 1
            self.llm_model.model_config.stage_num = parallel_config.pipeline_stage - 1
