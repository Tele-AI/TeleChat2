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
"""GLM config"""

from typing import Union

from mindspore._checkparam import args_type_check

from mindformers.modules.transformer.moe import MoEConfig
from mindformers.modules.transformer.transformer import default_transformer_config, default_moe_config, \
    TransformerOpParallelConfig, OpParallelConfig, EmbeddingOpParallelConfig, default_embedding_parallel_config
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.models.utils import convert_mstype
from mindformers.models.configuration_utils import PretrainedConfig
from mindformers.mindformer_book import MindFormerBook

default_dpmp_config = OpParallelConfig()

__all__ = ['GLMConfig']


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class GLMConfig(PretrainedConfig):
    """
    GLM config class which defines the model size
    Args:
        batch_size (`int`, *optional*, defaults to 1):
            batch size for input data, use in predict.
        vocab_size (`int`, *optional*, defaults to 130528):
            Vocabulary size of the GLM model. Defines the maximum number of different tokens that
            can be represented by the `inputs_ids` passed when calling [`GLMModel`].
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the embeddings and hidden states.
        num_layers (`int`, *optional*, defaults to 28):
            Number of hidden layers in the Transformer encoder.
        num_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        inner_hidden_size (`int`, *optional*, defaults to 16384):
            Dimensionality of hidden states in FeedForward.
        seq_length (`int`, *optional*, defaults to 512):
            The sequence length of input_ids, default is 512.
        embedding_dropout_prob (`float`, *optional*, defaults to 0.0):
            Dropout rate applied to the embedding probs.
        attention_dropout_rate (`float`, *optional*, defaults to 0.1):
            Dropout rate applied to the attention probs.
        hidden_size_per_attention_head (`int`, *optional*, defaults to None):
            hidden size per attention head. default "None" means hidden-size/num-attention-heads.
        layernorm_order (`str`, *optional*, defaults to `post`):
            define where is the layernorm added in transformer layers,
            support "pre" "post" "sandwich", default is "post".
        layernorm_epsilon (`float`, *optional*, defaults to 1.0e-5):
            epsilon value in layernorm, default is 1.0e-5.
        use_final_layernorm (`bool`, *optional*, defaults to True):
            whether to use final layernorm or not after all layers, default is True.
        embed_parallel_config(EmbeddingOpParallelConfig):
            The parallel configure. Default `default_embedding_parallel_config`,
            an instance of `TransformerOpParallelConfig` with default args.
        parallel_config(TransformerOpParallelConfig):
            The parallel configure. Default `default_transformer_config`,
            an instance of `TransformerOpParallelConfig` with default args.
        moe_config(MoEConfig):
            The configuration of MoE (Mixture of Expert). Default is an instance of MoEConfig
            with default values. Please see `MoEConfig`.
        use_past (`bool`, *optional*, defaults to `False`):
            Whether or not the model should use the past last key/values attentions
            (if applicable to the model) to speed up decoding. only available for generation.
        activation_func (`str`, *optional*, defaults to `GELU`):
            The activate function used in Linear, default is GELU.
        position_encoding_2d (`bool`, *optional*, defaults to `True`):
            Whether to use 2d format of position encoding for GLM model, default is True.
        param_init_type (`str`, *optional*, defaults to  = "float16"):
            Network parameter initialization type, default is "float16".
        layernorm_compute_type (`str`, *optional*, defaults to  = "floa32"):
            compute dtype for layernorm, default is "float32".
        softmax_compute_type (`str`, *optional*, defaults to  = "floa32"):
            compute dtype for softmax, default is "float32".
        compute_dtype (`str`, *optional*, defaults to  = "floa16"):
            compute dtype for network, default is "float16".
        bos_token_id (`int`, *optional*, defaults to 130004):
            A special token representing the beginning of a sentence.
        eos_token_id (`int`, *optional*, defaults to 130005):
            A special token representing the end of a sentence.
        mask_token_id (`int`, *optional*, defaults to 130000):
            A special token representing an mask token.
        gmask_token_id (`int`, *optional*, defaults to 130000):
            A special token representing an gmask token.
        pad_token_id (`int`, *optional*, defaults to 3):
            A special token used to make arrays of tokens the same size for batching purpose.
            Will then be ignored by attention mechanisms or loss computation.
        is_enhanced_encoder (`bool`, *optional*, defaults to `True`):
            glm specified branch control, deprecated.
        is_sample_acceleration (`bool`, *optional*, defaults to `False`):
            Whether to do sample in construct to accelerate generation.
            This can accelerate post process a bit during generation, but will lose the
            flexibility of generation config, not commended. Default to False.
        checkpoint_name_or_path (`str`, *optional*, defaults to "")
            checkpoint path or name used to load to the network.
        max_decode_length (`int`, *optional*, defaults to 2048):
            The maximum length the generated tokens can have.
        top_k (`int`, *optional*, defaults to 5):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        top_p (`float`, *optional*, defaults to 1.0):
            If set to float < 1, only the smallest set of most probable tokens with probabilities
            that add up to `top_p` or higher are kept for generation.
        repetition_penalty (`float`, *optional*, defaults to 1.0):
            The parameter for repetition penalty. 1.0 means no penalty. See [this
            paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
        do_sample (`bool`, *optional*, defaults to `False`):
            Whether or not to use sampling; use greedy decoding otherwise.
        ignore_index (`int`, *optional*, defaults to -100):
            index that will be ignored in input_ids and labels for training.
    """

    model_type = "glm"
    _support_list = MindFormerBook.get_config_support_list()['glm']

    @args_type_check(parallel_config=(dict, TransformerOpParallelConfig),
                     moe_config=(dict, MoEConfig))
    def __init__(self,
                 batch_size: int = 1,
                 vocab_size: int = 130528,
                 hidden_size: int = 4096,
                 num_layers: int = 28,
                 num_heads: int = 32,
                 inner_hidden_size: int = 16384,
                 seq_length: int = 512,
                 embedding_dropout_prob: float = 0.0,
                 attention_dropout_rate: float = 0.0,
                 hidden_dropout_rate: float = 0.0,
                 hidden_size_per_attention_head: int = None,
                 layernorm_order: str = "post",
                 layernorm_epsilon: float = 1.0e-5,
                 use_final_layernorm: bool = True,
                 op_parallel_config: Union[dict, OpParallelConfig] = default_dpmp_config,
                 embed_parallel_config: Union[dict, EmbeddingOpParallelConfig] = default_embedding_parallel_config,
                 parallel_config: Union[dict, TransformerOpParallelConfig] = default_transformer_config,
                 moe_config: Union[dict, MoEConfig] = default_moe_config,
                 use_past: bool = False,
                 activation_func: str = 'GELU',
                 position_encoding_2d: bool = True,
                 param_init_type: str = "float16",
                 layernorm_compute_type: str = "float32",
                 softmax_compute_type: str = "float32",
                 compute_dtype: str = "float16",
                 bos_token_id: int = 130004,
                 eos_token_id: int = 130005,
                 mask_token_id: int = 130000,
                 gmask_token_id: int = 130001,
                 pad_token_id: int = 3,
                 is_enhanced_encoder: bool = True,
                 is_sample_acceleration: bool = False,
                 checkpoint_name_or_path: str = "",
                 max_decode_length: int = 2048,
                 top_k: int = 1,
                 top_p: float = 1,
                 repetition_penalty: float = 1.0,
                 do_sample: bool = True,
                 ignore_index: int = -100,
                 ignore_token_id=None,
                 **kwargs):
        super().__init__(**kwargs)
        if isinstance(op_parallel_config, dict):
            op_parallel_config = OpParallelConfig(**op_parallel_config)
        if isinstance(embed_parallel_config, dict):
            embed_parallel_config = EmbeddingOpParallelConfig(**embed_parallel_config)
        if isinstance(parallel_config, dict):
            parallel_config = TransformerOpParallelConfig(**parallel_config)
        if isinstance(moe_config, dict):
            moe_config = MoEConfig(**moe_config)
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding_dropout_prob = embedding_dropout_prob
        self.hidden_dropout_rate = hidden_dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.hidden_size_per_attention_head = hidden_size_per_attention_head
        self.layernorm_order = layernorm_order
        self.layernorm_epsilon = layernorm_epsilon
        self.use_final_layernorm = use_final_layernorm
        self.op_parallel_config = op_parallel_config
        self.embed_parallel_config = embed_parallel_config
        self.moe_config = moe_config
        self.use_past = use_past
        self.parallel_config = parallel_config
        self.activation_func = activation_func
        self.inner_hidden_size = inner_hidden_size
        self.position_encoding_2d = position_encoding_2d
        self.param_init_type = convert_mstype(param_init_type)
        self.layernorm_compute_type = convert_mstype(layernorm_compute_type)
        self.softmax_compute_type = convert_mstype(softmax_compute_type)
        self.compute_dtype = convert_mstype(compute_dtype)
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.mask_token_id = mask_token_id
        self.gmask_token_id = gmask_token_id
        self.pad_token_id = pad_token_id
        self.max_decode_length = max_decode_length
        self.seq_length = seq_length
        self.is_enhanced_encoder = is_enhanced_encoder
        self.is_sample_acceleration = is_sample_acceleration
        self.checkpoint_name_or_path = checkpoint_name_or_path
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.do_sample = do_sample
        self.ignore_index = ignore_index
        self.ignore_token_id = ignore_token_id
