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

"""Bloom Config API"""

from typing import Union

from mindspore._checkparam import args_type_check

from mindformers.modules.transformer.moe import MoEConfig
from mindformers.modules.transformer.transformer import default_transformer_config, default_moe_config, \
    TransformerOpParallelConfig
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.models.utils import convert_mstype
from mindformers.models.configuration_utils import PretrainedConfig
from mindformers.mindformer_book import MindFormerBook

__all__ = ['BloomConfig']


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class BloomConfig(PretrainedConfig):
    """
    Bloom config class which defines the model size.

    Args:
        vocab_size (`int`, *optional*, defaults to 250880):
            Vocabulary size of the Bloom model. Defines the maximum number of different tokens that can be represented
            by the `inputs_ids` passed when calling [`BloomLMHeadModel`].
        batch_size (`int`, *optional*, defaults to 1):
            batch size for input data, use in predict.
        seq_length (`int`, *optional*, defaults to 1024):
            The sequence length of input_ids.
        hidden_size (`int`, *optional*, defaults to 64):
            Dimension of the embeddings and hidden states.
        num_layers (`int`, *optional*, defaults to 2):
            Number of hidden layers in the Transformer encoder.
        num_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer decoder.
        hidden_dropout_rate (`float`, *optional*, defaults to 0.1):
            Dropout rate of the dropout function on the bias dropout.
        attention_dropout_rate (`float`, *optional*, defaults to 0.1):
            Dropout rate applied to the attention probs.
        embedding_dropout_prob (`float`, *optional*, defaults to 0.0):
            Dropout rate applied to the embedding probs.
        hidden_act (`str` or `Callable`, *optional*, defaults to "gelu"):
            The non-linear activation function (function or string) in the encoder and pooler.
        param_init_type (Optional[str]):
            Network parameter initialization type, default is "float32".
        embedding_init_type (Optional[str]):
            Embedding compute dtype, default is "float32".
        compute_dtype (Optional[str]):
            Linear layer compute dtype, default is "float16".
        layernorm_compute_type (Optional[str]):
            layernorm compute dtype, default is "float32".
        softmax_compute_type (Optional[str]):
            softmax compute dtype, default is "float32".
        moe_config(MoEConfig):
            The configuration of MoE (Mixture of Expert). Default is an instance of MoEConfig
            with default values. Please see `MoEConfig`.
        parallel_config(TransformerOpParallelConfig):
            The parallel configure. Default `default_transformer_config`,
            an instance of `TransformerOpParallelConfig` with default args.
        checkpoint_name_or_path (Optional[str]):
            checkpoint path or name used to load to the network.
        use_past (`bool`, *optional*, defaults to `False`):
            Whether or not the model should use the past last key/values attentions
            (if applicable to the model) to speed up decoding.
        bos_token_id (`int`, *optional*, defaults to 1):
            A special token representing the beginning of a sentence.
        eos_token_id (`int`, *optional*, defaults to 2):
            A special token representing the end of a sentence.
        unk_token_id (`int`, *optional*, defaults to 0):
            A special token representing an out-of-vocabulary token.
        pad_token_id (`int`, *optional*, defaults to 3):
            A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by
            attention mechanisms or loss computation.
        repetition_penalty (`float`, *optional*, defaults to 1.0):
            The parameter for repetition penalty. 1.0 means no penalty. See [this
            paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
        max_decode_length (`int`, *optional*, defaults to 1024):
            The maximum length the generated tokens can have.
        top_k (`int`, *optional*, defaults to 5):
            The number of the highest probability vocabulary tokens to keep for top-k-filtering.
        top_p (`float`, *optional*, defaults to 1.0):
            If set to float < 1, only the smallest set of most probable tokens with probabilities
            that add up to `top_p` or higher are kept for generation.
        do_sample (`bool`, *optional*, defaults to `False`):
            Whether to use sampling ; use greedy decoding otherwise.
        is_sample_acceleration(`bool`, *optional*, defaults to `False`):
            When it is used for network inference, the sampling process is completed in construct.

    Returns:
        Class, BloomConfig.
    """

    model_type = "bloom"
    _support_list = MindFormerBook.get_config_support_list()['bloom']

    @args_type_check(parallel_config=(dict, TransformerOpParallelConfig),
                     moe_config=(dict, MoEConfig))
    def __init__(self,
                 embedding_dropout_prob: float = 0.0,
                 batch_size: int = 1,
                 seq_length: int = 1024,
                 vocab_size: int = 250880,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 num_heads: int = 8,
                 expand_ratio: int = 4,
                 hidden_dropout_rate: float = 0.1,
                 attention_dropout_rate: float = 0.1,
                 param_init_type: str = "float32",
                 embedding_init_type: str = "float32",
                 layernorm_compute_type: str = "float32",
                 softmax_compute_type: str = "float32",
                 compute_dtype: str = "float16",
                 hidden_act: str = 'gelu',
                 parallel_config: Union[dict, TransformerOpParallelConfig] = default_transformer_config,
                 checkpoint_name_or_path: str = "",
                 moe_config: Union[dict, MoEConfig] = default_moe_config,
                 use_seq_parallel: bool = False,
                 use_select_recompute: bool = False,
                 use_past: bool = False,
                 unk_token_id: int = 0,
                 bos_token_id: int = 1,
                 eos_token_id: int = 2,
                 pad_token_id: int = 3,
                 repetition_penalty: int = 1,
                 max_decode_length: int = 1024,
                 top_k: int = 5,
                 top_p: int = 1,
                 do_sample: bool = True,
                 is_sample_acceleration: bool = False,
                 use_flash_attention: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        if isinstance(parallel_config, dict):
            parallel_config = TransformerOpParallelConfig(**parallel_config)
        if isinstance(moe_config, dict):
            moe_config = MoEConfig(**moe_config)
        self.embedding_dropout_prob = embedding_dropout_prob
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.expand_ratio = expand_ratio
        self.hidden_dropout_rate = hidden_dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.param_init_type = convert_mstype(param_init_type)
        self.embedding_init_type = convert_mstype(embedding_init_type)
        self.layernorm_compute_type = convert_mstype(layernorm_compute_type)
        self.softmax_compute_type = convert_mstype(softmax_compute_type)
        self.compute_dtype = convert_mstype(compute_dtype)
        self.parallel_config = parallel_config
        self.checkpoint_name_or_path = checkpoint_name_or_path
        self.moe_config = moe_config
        self.use_past = use_past
        self.unk_token_id = unk_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.hidden_act = hidden_act
        self.use_seq_parallel = use_seq_parallel
        self.use_select_recompute = use_select_recompute
        self.repetition_penalty = repetition_penalty
        self.max_decode_length = max_decode_length
        self.top_k = top_k
        self.top_p = top_p
        self.do_sample = do_sample
        self.is_sample_acceleration = is_sample_acceleration
        self.use_flash_attention = use_flash_attention
        if self.batch_size is None:
            self.use_past = False  # currently require batch_size = 1
            self.is_sample_acceleration = False  # currently require batch_size = 1
