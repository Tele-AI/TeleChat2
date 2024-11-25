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
"""Whisper Config"""

from mindformers.models.configuration_utils import PretrainedConfig
from mindformers.models.utils import convert_mstype
from mindformers.modules.transformer.transformer import default_transformer_config, \
    TransformerOpParallelConfig
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class WhisperConfig(PretrainedConfig):
    """Whisper Model Config"""
    model_type = "whisper"

    def __init__(
            self,
            vocab_size=51865,
            num_mel_bins=80,
            encoder_layers=4,
            encoder_attention_heads=6,
            decoder_layers=4,
            decoder_attention_heads=6,
            decoder_ffn_dim=1536,
            encoder_ffn_dim=1536,
            encoder_layerdrop=0.0,
            decoder_layerdrop=0.0,
            decoder_start_token_id=50257,
            use_cache=True,
            is_encoder_decoder=True,
            activation_function="gelu",
            d_model=384,
            dropout=0.0,
            attention_dropout=0.0,
            activation_dropout=0.0,
            max_source_positions=1500,
            max_target_positions=448,
            pad_token_id=50256,
            bos_token_id=50256,
            eos_token_id=50256,
            apply_spec_augment=False,
            compute_dtype: str = "float16",
            layernorm_compute_dtype: str = "float32",
            softmax_compute_type: str = "float32",
            param_init_type: str = "float16",
            embedding_init_type: str = "float16",
            is_dynamic: bool = False,
            use_flash_attention=False,
            num_layers_of_each_stage=None,
            parallel_config=default_transformer_config,
            **kwargs,
    ):
        if isinstance(parallel_config, dict):
            parallel_config = TransformerOpParallelConfig(**parallel_config)
        self.parallel_config = parallel_config
        self.vocab_size = vocab_size
        self.num_layers_of_each_stage = num_layers_of_each_stage
        self.num_mel_bins = num_mel_bins
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.encoder_ffn_dim = encoder_ffn_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions

        self.apply_spec_augment = apply_spec_augment

        self.layernorm_compute_dtype = convert_mstype(layernorm_compute_dtype)
        self.softmax_compute_type = convert_mstype(softmax_compute_type)
        self.compute_dtype = convert_mstype(compute_dtype)
        self.param_init_type = convert_mstype(param_init_type)
        self.embedding_init_type = convert_mstype(embedding_init_type)

        self.is_dynamic = is_dynamic
        self.use_flash_attention = use_flash_attention
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            **kwargs,
        )
