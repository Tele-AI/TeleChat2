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
"""Gpt Config API."""

from typing import Union

from mindspore._checkparam import args_type_check

from mindformers.modules.transformer.moe import MoEConfig
from mindformers.modules.transformer.transformer import default_transformer_config, default_moe_config, \
    TransformerOpParallelConfig
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.models.utils import convert_mstype
from mindformers.models.configuration_utils import PretrainedConfig
from mindformers.mindformer_book import MindFormerBook

__all__ = ['GPT2Config']


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class GPT2Config(PretrainedConfig):
    """
    Gpt config class which defines the model size

    Args:
        batch_size (Optional[int]): batch size for input data, use in predict.
        eos_token_id (Optional[int]): The id of the *end-of-sequence* token.
        pad_token_id (Optional[int]): The id of the *padding* token.
        bos_token_id (Optional[int]): The id of the *beginning-of-sequence* token.
        unk_token_id (Optional[int]): The id of the *unknown* token.
        seq_length (Optional[int]): The sequence length of input_ids, default is 1024.
        vocab_size (`int`, *optional*, defaults to 50257):
            Vocabulary size of the BERT model.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_labels (Optional[int]): The number of label, default is 2.
        expand_ratio (Optional[int]): The expand ratio, default 4.
        embedding_dropout_prob (Optional[float]): The dropout ratio of embedding layer, default 0.1.
        hidden_dropout_rate (Optional[float]): The dropout ratio of hidden ffn layer, default 0.1.
        attention_dropout_rate (Optional[float]): The dropout ratio of attention layer, default 0.1.
        param_init_type (Optional[str]):
            parameter initial dtype, default is "float32".
        layernorm_compute_type (Optional[str]):
            layernorm compute dtype, default is "float32".
        softmax_compute_type (Optional[str]):
            softmax compute dtype, default is "float32".
        compute_dtype (Optional[str]):
            Linear layer compute dtype, default is "float16".
        hidden_act(str):
            The activation of the internal feedforward layer. Supports 'relu',
            'relu6', 'tanh', 'gelu', 'fast_gelu', 'elu', 'sigmoid', 'prelu', 'leakyrelu', 'hswish',
            'hsigmoid', 'logsigmoid' and so on. User can provide custom activition to the argument.
            If user wants to run the net in the parallel mode, the custom activation must also provide
            the `activation_shard` function. Please see the examples of the
            class:`mindformers.modules.transformer.FeedForward`. Default: gelu.
        use_past (`bool`, *optional*, defaults to `False`):
            Whether the model should use the past last key/values attentions
            (if applicable to the model) to speed up decoding.
        post_layernorm_residual(bool): Whether to use post layernorm, default False.
        offset(int): Offset of transformer layer when set pipeline stage number.
        checkpoint_name_or_path (Optional[str]):
            checkpoint path or name used to load to the network.
        parallel_config(TransformerOpParallelConfig):
            The parallel configure. Default `default_transformer_config`,
            an instance of `TransformerOpParallelConfig` with default args.
        moe_config(MoEConfig):
            The configuration of MoE (Mixture of Expert). Default is an instance of MoEConfig
            with default values. Please see `MoEConfig`.
        repetition_penalty (`float`, *optional*, defaults to 1.0):
            The parameter for repetition penalty. 1.0 means no penalty. See [this
            paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
        max_decode_length (`int`, *optional*, defaults to 1024):
            The maximum length the generated tokens can have. Corresponds to the length of the input prompt +
            `max_new_tokens`. Its effect is overridden by `max_new_tokens`, if also set.
        top_k (`int`, *optional*, defaults to 5):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        top_p (`float`, *optional*, defaults to 1.0):
            If set to float < 1, only the smallest set of most probable tokens with probabilities
            that add up to `top_p` or higher are kept for generation.
        do_sample (`bool`, *optional*, defaults to `False`):
            Whether to use sampling ; use greedy decoding otherwise.

    Returns:
        Class, GPT2Config.
    """

    model_type = 'gpt2'
    _support_list = MindFormerBook.get_config_support_list()['gpt2']

    @args_type_check(parallel_config=(dict, TransformerOpParallelConfig),
                     moe_config=(dict, MoEConfig))
    def __init__(self,
                 batch_size: int = 1,
                 eos_token_id: int = 50256,
                 pad_token_id: int = 50256,
                 bos_token_id: int = 50256,
                 unk_token_id: int = 50256,
                 seq_length: int = 1024,
                 max_position_embeddings: int = None,
                 vocab_size: int = 50257,
                 hidden_size: int = 768,
                 num_layers: int = 12,
                 num_heads: int = 12,
                 num_labels: int = 2,
                 expand_ratio: int = 4,
                 embedding_dropout_prob: float = 0.1,
                 hidden_dropout_rate: float = 0.1,
                 attention_dropout_rate: float = 0.1,
                 param_init_type: str = "float32",
                 layernorm_compute_type: str = "float32",
                 softmax_compute_type: str = "float32",
                 compute_dtype: str = "float16",
                 hidden_act: str = 'gelu',
                 use_past: bool = False,
                 post_layernorm_residual: bool = False,
                 offset: int = 0,
                 parallel_config: Union[dict, TransformerOpParallelConfig] = default_transformer_config,
                 checkpoint_name_or_path: str = "",
                 moe_config: Union[dict, MoEConfig] = default_moe_config,
                 repetition_penalty: float = 1.0,
                 max_decode_length: int = 1024,
                 top_k: int = 5,
                 top_p: float = 1.0,
                 do_sample: bool = True,
                 use_flash_attention: bool = False,
                 use_prompt_flash_attention: bool = False,
                 is_dynamic=False,
                 block_size: int = 16,
                 num_blocks: int = 512,
                 **kwargs):
        super(GPT2Config, self).__init__(**kwargs)
        if isinstance(parallel_config, dict):
            parallel_config = TransformerOpParallelConfig(**parallel_config)
        if isinstance(moe_config, dict):
            moe_config = MoEConfig(**moe_config)
        self.batch_size = batch_size
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.unk_token_id = unk_token_id
        self.seq_length = seq_length
        self.max_position_embeddings = max_position_embeddings
        if max_position_embeddings is None:
            self.max_position_embeddings = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_labels = num_labels
        self.expand_ratio = expand_ratio
        self.embedding_dropout_prob = embedding_dropout_prob
        self.hidden_dropout_rate = hidden_dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.param_init_type = convert_mstype(param_init_type)
        self.layernorm_compute_type = convert_mstype(layernorm_compute_type)
        self.softmax_compute_type = convert_mstype(softmax_compute_type)
        self.compute_dtype = convert_mstype(compute_dtype)
        self.hidden_act = hidden_act
        self.use_past = use_past
        self.post_layernorm_residual = post_layernorm_residual
        self.offset = offset
        self.parallel_config = parallel_config
        self.checkpoint_name_or_path = checkpoint_name_or_path
        self.moe_config = moe_config
        self.repetition_penalty = repetition_penalty
        self.max_decode_length = max_decode_length
        self.top_k = top_k
        self.top_p = top_p
        self.do_sample = do_sample
        self.use_flash_attention = use_flash_attention
        self.use_prompt_flash_attention = use_prompt_flash_attention
        self.is_dynamic = is_dynamic
        self.block_size = block_size
        self.num_blocks = num_blocks
