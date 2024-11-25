# Copyright 2022 Huawei Technologies Co., Ltd
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
"""Bert Config API."""

from typing import Union

from mindspore._checkparam import args_type_check
import mindspore.common.dtype as mstype

from mindformers.modules.transformer.moe import MoEConfig
from mindformers.modules.transformer.transformer import default_transformer_config, default_moe_config, \
    TransformerOpParallelConfig
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.models.configuration_utils import PretrainedConfig
from mindformers.mindformer_book import MindFormerBook

__all__ = ['BertConfig']


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class BertConfig(PretrainedConfig):
    """
    BERT config class which defines the model size.

    Args:
        model_type (Optional[str]): model type for bert model, default is 'bert'.
        batch_size (Optional[int]): batch size for input data, use in predict.
        seq_length (Optional[int]): The sequence length of input_ids, default is 128.
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the BERT model.
        use_one_hot_embeddings (Optional[bool]): whether to use One-Hot embedding, default is False.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_labels (Optional[int]): The number of label, default is 1.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (str, nn.Cell):
            The activation of the internal feedforward layer. Supports 'relu',
            'relu6', 'tanh', 'gelu', 'fast_gelu', 'elu', 'sigmoid', 'prelu', 'leakyrelu', 'hswish',
            'hsigmoid', 'logsigmoid' and so on. User can provide custom activition to the argument.
            If user wants to run the net in the parallel mode, the custom activation must also provide
            the `activation_shard` function. Please see the examples of the
            class:`mindformers.modules.transformer.FeedForward`. Default: gelu.
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids`.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        checkpoint_name_or_path (Optional[str]):
            checkpoint path or name used to load to the network.
        dtype (Optional[str]):
            layer digital type, default is "float32".
        compute_dtype (Optional[str]):
            Linear layer compute dtype, default is "float16".
        layernorm_dtype (Optional[str]):
            layernorm compute dtype, default is "float32".
        softmax_dtype (Optional[str]):
            softmax compute dtype, default is "float32".
        moe_config(MoEConfig):
            The configuration of MoE (Mixture of Expert). Default is an instance of MoEConfig
            with default values. Please see `MoEConfig`.
        parallel_config(TransformerOpParallelConfig):
            The parallel configure. Default `default_transformer_config`,
            an instance of `TransformerOpParallelConfig` with default args.

    Returns:
        Class, BertConfig.
    """

    model_type = "bert"
    _support_list = MindFormerBook.get_config_support_list()['bert']
    _support_list.extend(MindFormerBook.get_config_support_list()['tokcls']['bert'])
    _support_list.extend(MindFormerBook.get_config_support_list()['txtcls']['bert'])
    _support_list.extend(MindFormerBook.get_config_support_list()['qa']['bert'])

    @args_type_check(parallel_config=(dict, TransformerOpParallelConfig),
                     moe_config=(dict, MoEConfig))
    def __init__(self,
                 model_type: str = "bert",
                 use_one_hot_embeddings: bool = False,
                 num_labels: int = 1,
                 assessment_method: str = "",
                 dropout_prob: float = 0.1,
                 batch_size: int = 16,
                 seq_length: int = 128,
                 vocab_size: int = 30522,
                 hidden_size: int = 768,
                 num_hidden_layers: int = 12,
                 num_attention_heads: int = 12,
                 intermediate_size: int = 3072,
                 hidden_act: str = "gelu",
                 post_layernorm_residual: bool = True,
                 hidden_dropout_prob: float = 0.1,
                 attention_probs_dropout_prob: float = 0.1,
                 max_position_embeddings: int = 128,
                 type_vocab_size: int = 2,
                 initializer_range: float = 0.02,
                 use_relative_positions: bool = False,
                 dtype: str = "float32",
                 layernorm_dtype: str = "float32",
                 softmax_dtype: str = "float32",
                 compute_dtype: str = "float16",
                 use_past: bool = False,
                 parallel_config: Union[dict, TransformerOpParallelConfig] = default_transformer_config,
                 checkpoint_name_or_path: str = "",
                 moe_config: Union[dict, MoEConfig] = default_moe_config,
                 **kwargs):
        super(BertConfig, self).__init__(**kwargs)
        if isinstance(parallel_config, dict):
            parallel_config = TransformerOpParallelConfig(**parallel_config)
        if isinstance(moe_config, dict):
            moe_config = MoEConfig(**moe_config)
        self.model_type = model_type
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.num_labels = num_labels
        self.assessment_method = assessment_method
        self.dropout_prob = dropout_prob
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.post_layernorm_residual = post_layernorm_residual
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.use_relative_positions = use_relative_positions
        self.dtype = mstype.float32 if dtype == "float32" else mstype.float16
        self.layernorm_dtype = mstype.float32 if layernorm_dtype == "float32" else mstype.float16
        self.softmax_dtype = mstype.float32 if softmax_dtype == "float32" else mstype.float16
        self.compute_dtype = mstype.float32 if compute_dtype == "float32" else mstype.float16
        self.use_past = use_past
        self.checkpoint_name_or_path = checkpoint_name_or_path
        self.parallel_config = parallel_config
        self.moe_config = moe_config
