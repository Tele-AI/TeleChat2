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
"""Llama Config API."""

from typing import Optional, Union

from mindspore._checkparam import args_type_check

from mindformers.modules.transformer.transformer import (
    default_transformer_config,
    TransformerOpParallelConfig,
)
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.models.configuration_utils import PretrainedConfig
from mindformers.models.utils import convert_mstype

__all__ = ["LlmBoostConfig"]


@MindFormerRegister.register(MindFormerModuleType.CONFIG)
class LlmBoostConfig(PretrainedConfig):
    """
    Llm boost config class which defines the model size.

    Args:
        batch_size (int, optional): batch size for input data, use in predict. Default: ``1`` .
        seq_length (int, optional): The sequence length of input_ids. Default: ``2048`` .
        vocab_size (int, optional): Default: ``32000`` .
            Vocabulary size of the BERT model.
        hidden_size (int, optional):
            Dimensionality of the encoder layers and the pooler layer. Default: ``4096`` .
        num_layers (int, optional):
            Number of hidden layers in the Transformer encoder. Default: ``32`` .
        num_heads (int, optional):
            Number of attention heads for each attention layer in the Transformer encoder. Default: ``32`` .
        n_kv_heads (int, optional): Define multi group head attention heads number. Default: ``None`` .
        rms_norm_eps (float, optional): The epsilon value of the denominator. Default: ``1e-5`` .
        bos_token_id (int, optional): The id of the *beginning-of-sequence* token. Default: ``1`` .
        eos_token_id (int, optional): The id of the *end-of-sequence* token. Default: ``2`` .
        pad_token_id (int, optional): The id of the *padding* token. Default: ``0`` .
        ignore_token_id (int, optional): The id of the *ignoring* token. Default: ``-100`` .
        compute_dtype (str, optional):
            Linear layer compute dtype. Default: ``float16`` .
        rotary_dtype (str, optional):
            rope compute dtype. Default: ``float32`` .
        parallel_config(TransformerOpParallelConfig):
            The parallel configure. Default: ``default_transformer_config`` ,
            an instance of `TransformerOpParallelConfig` with default args.
        use_past (bool, optional):
            Whether the model should use the past last key/values attentions
            (if applicable to the model) to speed up decoding. Default: ``False`` .
        extend_method(str, optional): The extent method of seq length of inference. Default: ``None`` .
        repetition_penalty (float, optional):
            The parameter for repetition penalty. 1.0 means no penalty.
            See `this paper <https://arxiv.org/pdf/1909.05858.pdf>`_ for more details. Default: ``1.0`` .
        max_decode_length (int, optional):
            The maximum length the generated tokens can have. Corresponds to the length of the input prompt +
            `max_new_tokens`. Its effect is overridden by `max_new_tokens`, if also set. Default: ``1024`` .
        top_k (int, optional):
            The number of highest probability vocabulary tokens to keep for top-k-filtering. Default: ``5`` .
        top_p (float, optional):
            If set to float < 1, only the smallest set of most probable tokens with probabilities
            that add up to `top_p` or higher are kept for generation. Default: ``1.0`` .
        do_sample (bool, optional):
            Whether to use sampling; use greedy decoding otherwise. Default: ``True`` .
        block_size (int, optional):
            The maximum number of tokens in one block can have when using paged attention. Default: ``16`` .
        num_blocks (int, optional):
            The maximum number of blocks when using paged attention. Default: ``512`` .
        llm_backend (str, optional):
            Llm boost backend. Default: ``BuildIn`` .
        boost_model_name (str, optional):
            Llm boost model name. Default: ``None`` .
        communication_backend (str, optional):
            communication_backend, ``hccl`` or ``lccl``. Default: ``hccl`` .
    Returns:
        LlmBoostConfig, a LlmBoostConfig instance.
    """

    # pylint: disable=C0330
    @args_type_check(parallel_config=(dict, TransformerOpParallelConfig))
    def __init__(
        self,
        batch_size: int = 1,
        llm_backend: str = "BuildIn",
        boost_model_name: str = "",
        seq_length: int = 2048,
        hidden_size: int = 4096,
        num_layers: int = 32,
        num_heads: int = 32,
        n_kv_heads: Optional[int] = None,
        max_position_embedding: Optional[int] = None,
        vocab_size: int = 32000,
        rms_norm_eps: float = 1e-5,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        ignore_token_id: int = -100,
        theta: float = 10000.0,
        compute_dtype: str = "float16",
        rotary_dtype: str = "float16",
        parallel_config: Union[
            dict, TransformerOpParallelConfig
        ] = default_transformer_config,
        use_past: bool = False,
        scaling_factor: float = 1.0,
        extend_method: str = "None",
        is_dynamic: bool = False,
        parallel_optimizer: bool = False,
        repetition_penalty: float = 1.0,
        max_decode_length: int = 1024,
        block_size: int = 16,
        num_blocks: int = 512,
        top_k: int = 5,
        top_p: float = 1.0,
        do_sample: bool = True,
        quant_config: dict = None,
        communication_backend: str = "",
        **kwargs
    ):
        super(LlmBoostConfig, self).__init__(**kwargs)
        if isinstance(parallel_config, dict):
            parallel_config = TransformerOpParallelConfig(**parallel_config)
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_position_embedding = (
            max_position_embedding if max_position_embedding else seq_length
        )
        self.n_kv_heads = n_kv_heads
        self.rms_norm_eps = rms_norm_eps
        self.compute_dtype = convert_mstype(compute_dtype)
        self.rotary_dtype = convert_mstype(rotary_dtype)
        self.parallel_config = parallel_config
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.ignore_token_id = ignore_token_id
        self.use_past = use_past
        self.extend_method = extend_method
        self.scaling_factor = scaling_factor
        self.is_dynamic = is_dynamic
        self.parallel_optimizer = parallel_optimizer
        self.repetition_penalty = repetition_penalty
        self.max_decode_length = max_decode_length
        self.top_k = top_k
        self.top_p = top_p
        self.do_sample = do_sample
        self.theta = theta
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.quant_config = quant_config
        self.llm_backend = llm_backend
        self.boost_model_name = boost_model_name
        self.communication_backend = communication_backend
        self.parallel_decoding_params = kwargs.get("parallel_decoding_params")
