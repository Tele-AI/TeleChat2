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


"""Blip2 Config API"""
from typing import Optional, Union

from mindspore._checkparam import args_type_check
import mindspore.common.dtype as mstype

from mindformers.models.blip2.qformer_config import QFormerConfig
from mindformers.models.llama import LlamaConfig
from mindformers.models.vit import ViTConfig
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.models.configuration_utils import PretrainedConfig
from mindformers.mindformer_book import MindFormerBook
from mindformers.modules.transformer.transformer import default_transformer_config, TransformerOpParallelConfig

__all__ = ['Blip2Config']


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class Blip2Config(PretrainedConfig):
    r"""
    Config For BLIP2 Module

    Args:
        model_type (Optional[int]):
            model type for blip2 model, default is 'blip2'.
        batch_size (Optional[int]):
            batch size for input data, use in predict.
        freeze_vision (Optional[bool]):
            whether to freeze vit weights, default is True.
        freeze_text (Optional[bool]):
            whether to freeze LLM weights, default is True.
        max_txt_len (Optional[int]):
            max text length for llama model.
        prompt (Optional[str]):
            prompt for llama model.
        prompt_length (Optional[int]):
            prompt length for llama model.
        checkpoint_name_or_path (Optional[str]):
            checkpoint path or name used to load to the network.
        dtype (Optional[str]):
            layer digital type, default is "float32".
        compute_dtype (Optional[str]):
            Linear layer compute dtype, default is "float16".
        layernorm_compute_type (Optional[str]):
            layernorm compute dtype, default is "float32".
        softmax_compute_type (Optional[str]):
            softmax compute dtype, default is "float32".
        vision_config (Optional[ViTConfig]):
            config for ViTModel.
        qformer_config (Optional[QFormerConfig]):
            config for qformer.
        text_config (Optional[LlamaConfig]):
            config for LLM model, like llama.
        parallel_config(TransformerOpParallelConfig):
            The parallel configure. Default `default_transformer_config`,
            an instance of `TransformerOpParallelConfig` with default args.
        is_training (Optional[bool]): whether the model is in training state.
    Returns:
        Class, Blip2Config.
    """

    model_type = "blip2"
    _support_list = MindFormerBook.get_config_support_list()['blip2']

    @args_type_check(parallel_config=(dict, TransformerOpParallelConfig),
                     qformer_config=(dict, QFormerConfig))
    def __init__(self,
                 batch_size: int = 8,
                 freeze_vision: bool = True,
                 freeze_text: bool = True,
                 max_txt_len: int = 32,
                 prompt: bool = False,
                 prompt_length: int = 0,
                 checkpoint_name_or_path: str = None,
                 dtype: str = "float32",
                 compute_dtype: str = "float16",
                 layernorm_compute_type: str = "float32",
                 softmax_compute_type: str = "float32",
                 vision_config: Union[dict, ViTConfig] = ViTConfig(),
                 qformer_config: Union[dict, QFormerConfig] = QFormerConfig(),
                 text_config: Optional[LlamaConfig] = None,
                 parallel_config: Union[dict, TransformerOpParallelConfig] = default_transformer_config,
                 is_training: bool = True,
                 micro_batch_interleave_num=1,
                 use_past=False,
                 **kwargs):
        super(Blip2Config, self).__init__(**kwargs)
        if isinstance(parallel_config, dict):
            parallel_config = TransformerOpParallelConfig(**parallel_config)
        if isinstance(qformer_config, dict):
            qformer_config = QFormerConfig(**qformer_config)
        if isinstance(vision_config, dict):
            vision_config = ViTConfig(**vision_config)
        if isinstance(text_config, dict):
            text_config = LlamaConfig(**text_config)
        self.batch_size = batch_size
        self.freeze_vision = freeze_vision
        self.freeze_text = freeze_text
        self.max_txt_len = max_txt_len
        self.checkpoint_name_or_path = checkpoint_name_or_path
        self.prompt = prompt
        self.prompt_length = prompt_length

        self.parallel_config = parallel_config
        self.compute_dtype = mstype.float32 if compute_dtype == "float32" else mstype.float16
        self.layernorm_compute_type = mstype.float32 if layernorm_compute_type == "float32" else mstype.float16
        self.softmax_compute_type = mstype.float32 if softmax_compute_type == "float32" else mstype.float16
        self.dtype = mstype.float32 if dtype == "float32" else mstype.float16
        self.is_training = is_training
        self.micro_batch_interleave_num = micro_batch_interleave_num

        self.vision_config = vision_config
        self.qformer_config = qformer_config

        self.text_config = text_config

        # first stage is without text config
        if self.text_config is not None:
            self.text_config.parallel_config = parallel_config
            self.text_config.compute_dtype = self.compute_dtype
            self.text_config.layernorm_compute_type = self.layernorm_compute_type
            self.text_config.softmax_compute_type = self.softmax_compute_type
            self.text_config.dtype = self.dtype

        parallel_config.pipeline_stage = 1

        # pass configs to submodule config
        self.qformer_config.encoder_width = self.vision_config.hidden_size
        self.qformer_config.parallel_config = parallel_config
        self.qformer_config.compute_dtype = self.compute_dtype
        self.qformer_config.layernorm_dtype = self.layernorm_compute_type
        self.qformer_config.softmax_dtype = self.softmax_compute_type
        self.qformer_config.dtype = self.dtype

        self.vision_config.parallel_config = parallel_config
        self.vision_config.compute_dtype = self.compute_dtype
        self.vision_config.layernorm_compute_type = self.layernorm_compute_type
        self.vision_config.softmax_compute_type = self.softmax_compute_type
        self.vision_config.dtype = self.dtype
        self.use_past = use_past
