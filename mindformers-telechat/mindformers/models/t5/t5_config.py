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
"""T5 Configuration"""
from typing import Union

from mindspore._checkparam import args_type_check

from mindformers.modules.transformer.moe import MoEConfig
from mindformers.modules.transformer.transformer import default_transformer_config, default_moe_config, \
    TransformerOpParallelConfig
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.models.utils import convert_mstype
from mindformers.models.configuration_utils import PretrainedConfig
from mindformers.mindformer_book import MindFormerBook

__all__ = ['T5Config']


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class T5Config(PretrainedConfig):
    """
    T5 config class which defines the model size

    Args:
        vocab_size (`int`, *optional*, defaults to 32128):
            Vocabulary size of the T5 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`T5ForConditionalGeneration`].
        hidden_size (`int`, *optional*, defaults to 512):
            Dimension of the embeddings and hidden states.
        d_kv (`int`, *optional*, defaults to 64):
            Size of the key, query, value projections per attention head. The `inner_dim` of the projection layer will
            be defined as `num_heads * d_kv`.
        d_ff (`int`, *optional*, defaults to 2048):
            Size of the intermediate feed forward layer in each `T5Block`.
        num_layers (`int`, *optional*, defaults to 6):
            Number of hidden layers in the Transformer encoder.
        num_decoder_layers (`int`, *optional*):
            Number of hidden layers in the Transformer decoder. Will use the same value as `num_layers` if not set.
        num_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        relative_attention_num_buckets (`int`, *optional*, defaults to 32):
            The number of buckets to use for each attention layer.
        hidden_dropout_rate (`float`, *optional*, defaults to 0.1):
            Dropout rate of the dropout function on the bias dropout.
        attention_dropout_rate (`float`, *optional*, defaults to 0.1):
            Dropout rate applied to the attention probs.
        embedding_dropout_prob (`float`, *optional*, defaults to 0.0):
            Dropout rate applied to the embedding probs.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-6):
            The epsilon of layer norm in Transformer.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        is_encoder_decoder (`bool`, *optional*, defaults to `False`):
            Whether the model if Transformer encoder-decoder structure.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether the model should return the last key/values attentions (not used by all models).
        pad_token_id (`int`, *optional*, defaults to 0):
            A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by
            attention mechanisms or loss computation.
        start_token_id (`int`, *optional*, defaults to 1):
            A special token representing the beginning of a sentence.
        eos_token_id (`int`, *optional*, defaults to 2):
            A special token representing the end of a sentence.
        batch_size (`int`, *optional*, defaults to 1):
            Batch size for input data, use in train/finetune/evaluate/predict.
        seq_length (`int`, *optional*, defaults to 1024):
            The sequence length of input_ids, defaults is 1024.
        max_position_embeddings (`int`, *optional*, defaults to 1024):
            Maximum length of sequences used in this model.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Initialization value of TruncatedNormal in embedding layers.
        max_decode_length (`int`, *optional*, defaults to 128):
            The maximum length the generated tokens can have.
        compute_dtype (`str`, *optional*, defaults to "float32):
            Linear layer compute dtype.
        has_relative_bias (`bool`, *optional*, defaults to `True`):
            Whether to add bias to the result of query multiply value matrix.
        scale_output (`bool`, *optional*, defaults to `True`):
            Whether to scale the output of decoder.
        parallel_config (TransformerOpParallelConfig, defaults to default_transformer_config):
            The parallel configure. Default `default_transformer_config`,
            an instance of `TransformerOpParallelConfig` with default args.
        checkpoint_name_or_path (`str`, *optional*, defaults to None):
            checkpoint path or name used to load to the network.
        top_p (`float`, *optional*, defaults to 0.95):
            If set to float < 1, only the smallest set of most probable tokens with probabilities
            that add up to `top_p` or higher are kept for generation.
        top_k (`int`, *optional*, defaults to 1):
            The number of the highest probability vocabulary tokens to keep for top-k-filtering.
        repetition_penalty (`float`, *optional*, defaults to 1.0):
            The parameter for repetition penalty. 1.0 means no penalty. See [this
            paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
        max_length (`int`, *optional*, defaults to 20):
            The maximum length the generated tokens can have.
        do_sample (`bool`, *optional*, defaults to `False`):
            Whether to use sampling ; use greedy decoding otherwise.
        param_init_type (`float`, *optional*, defaults to "float32"):
            The type of parameters initializer.
        layernorm_compute_type (`str`, *optional*, defaults to "float32"):
            layernorm compute dtype.
        softmax_compute_type (`str`, *optional*, defaults to "float32"):
            softmax compute dtype.
        hidden_act (`str` or `Callable`, *optional*, defaults to "relu"):
            The non-linear activation function (function or string) in the encoder and pooler.
        post_layernorm_residual (`bool`, *optional*, defaults to `False`):
            Whether to use post layernorm in Transformer.
        offset (`int`, *optional*, defaults to 1):
            The offset value of the layer_index in pipeline parallel.
        use_past (`bool`, *optional*, defaults to `False`):
            Whether the model should use the past last key/values attentions
        moe_config (MoEConfig):
            The configuration of MoE (Mixture of Expert). Default is an instance of MoEConfig
            with default values. Please see `MoEConfig`.

    Returns:
        Class, T5Config.
    """

    model_type = "t5"
    _support_list = MindFormerBook.get_config_support_list()['t5']

    @args_type_check(parallel_config=(dict, TransformerOpParallelConfig),
                     moe_config=(dict, MoEConfig))
    def __init__(self,
                 vocab_size: int = 32128,
                 hidden_size: int = 512,
                 d_kv: int = 64,
                 d_ff: int = 2048,
                 num_layers: int = 6,
                 num_decoder_layers: int = None,
                 num_heads: int = 8,
                 relative_attention_num_buckets: int = 32,
                 hidden_dropout_rate: float = 0.1,
                 attention_dropout_rate: float = 0.1,
                 embedding_dropout_prob: float = 0.1,
                 layer_norm_epsilon: float = 1e-6,
                 initializer_factor: float = 1.0,
                 is_encoder_decoder: bool = True,
                 use_cache: bool = True,
                 pad_token_id: int = 0,
                 start_token_id: int = 0,
                 eos_token_id: int = 1,
                 batch_size: int = 1,
                 seq_length: int = 1024,
                 max_position_embeddings: int = 1024,
                 initializer_range: float = 0.02,
                 max_decode_length: int = 128,
                 length_penalty_weight: float = 1.0,
                 compute_dtype: str = "float32",
                 has_relative_bias: bool = True,
                 scale_output: bool = True,
                 parallel_config: Union[dict, TransformerOpParallelConfig] = default_transformer_config,
                 checkpoint_name_or_path: str = None,
                 top_p: float = 0.95,
                 top_k: int = 1,
                 repetition_penalty: float = 1.0,
                 max_length: int = 20,
                 do_sample: bool = False,
                 param_init_type: str = "float32",
                 layernorm_compute_type: str = "float32",
                 softmax_compute_type: str = "float32",
                 hidden_act: str = 'relu',
                 post_layernorm_residual: bool = False,
                 offset: int = 0,
                 use_past: bool = False,
                 moe_config: Union[dict, MoEConfig] = default_moe_config,
                 dtype=None,
                 **kwargs):
        super(T5Config, self).__init__(**kwargs)
        if isinstance(parallel_config, dict):
            parallel_config = TransformerOpParallelConfig(**parallel_config)
        if isinstance(moe_config, dict):
            moe_config = MoEConfig(**moe_config)
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.hidden_act = hidden_act
        self.kv_size = d_kv
        self.hidden_dropout_rate = hidden_dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.embedding_dropout_prob = embedding_dropout_prob
        self.initializer_factor = initializer_factor
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.max_decode_length = max_decode_length
        self.length_penalty_weight = length_penalty_weight
        self.has_relative_bias = has_relative_bias
        self.scale_output = scale_output
        self.parallel_config = parallel_config
        self.num_decoder_layers = num_decoder_layers
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.layer_norm_epsilon = layer_norm_epsilon
        self.use_cache = use_cache
        self.checkpoint_name_or_path = checkpoint_name_or_path
        self.pad_token_id = pad_token_id
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.max_length = max_length
        self.start_token_id = start_token_id
        self.eos_token_id = eos_token_id
        self.is_encoder_decoder = is_encoder_decoder
        self.do_sample = do_sample
        self.layernorm_compute_type = convert_mstype(layernorm_compute_type)
        self.softmax_compute_type = convert_mstype(softmax_compute_type)
        self.compute_dtype = convert_mstype(compute_dtype)
        self.use_past = use_past
        self.post_layernorm_residual = post_layernorm_residual
        self.offset = offset
        self.moe_config = moe_config
        self.param_init_type = convert_mstype(param_init_type)
        self.dtype = dtype
