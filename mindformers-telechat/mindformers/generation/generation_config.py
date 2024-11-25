# Copyright 2022 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""generation config."""
import copy
from typing import Any, Dict

from mindformers.models.configuration_utils import PretrainedConfig
from mindformers.tools.logger import logger

__all__ = ["GenerationConfig"]


class GenerationConfig:
    r"""
    Class that holds a configuration for a generation task.

    Some parameters have specific functions, see the table below for details:

    +------------------------------------------------------------+------------------------------+
    | Functional classification                                  |  Configuration parameter     |
    +============================================================+==============================+
    | Parameters that control the length of the output           |  max_length                  |
    |                                                            +------------------------------+
    |                                                            |  max_new_tokens              |
    |                                                            +------------------------------+
    |                                                            |  min_length                  |
    |                                                            +------------------------------+
    |                                                            |  min_new_tokens              |
    +------------------------------------------------------------+------------------------------+
    | Parameters that control the generation strategy used       |  do_sample                   |
    |                                                            +------------------------------+
    |                                                            |  use_past                    |
    +------------------------------------------------------------+------------------------------+
    | Parameters for manipulation of the model output logits     |  temperature                 |
    |                                                            +------------------------------+
    |                                                            |  top_k                       |
    |                                                            +------------------------------+
    |                                                            |  top_p                       |
    |                                                            +------------------------------+
    |                                                            |  repetition_penalty          |
    |                                                            +------------------------------+
    |                                                            |  encoder_repetition_penalty  |
    |                                                            +------------------------------+
    |                                                            |  renormalize_logits          |
    +------------------------------------------------------------+------------------------------+
    | Parameters that define the output variables of `generate`  |  output_scores               |
    |                                                            +------------------------------+
    |                                                            |  output_logits               |
    |                                                            +------------------------------+
    |                                                            |  return_dict_in_generate     |
    +------------------------------------------------------------+------------------------------+
    | Special tokens that can be used at generation time         |  pad_token_id                |
    |                                                            +------------------------------+
    |                                                            |  bos_token_id                |
    |                                                            +------------------------------+
    |                                                            |  eos_token_id                |
    +------------------------------------------------------------+------------------------------+

    Args:
        max_length (int, optional): The maximum length the generated tokens can have.
            Corresponds to the length of the input prompt + `max_new_tokens`.
            If `max_new_tokens` is also set, the effect of `max_length` is overridden by `max_new_tokens`.
            Default: ``20`` .
        max_new_tokens (int, optional): The maximum numbers of tokens to generate,
            ignoring the number of tokens in the prompt. Default: ``None`` .
        min_length (int, optional): The minimum length of the sequence to be generated.
            Corresponds to the length of the input prompt + `min_new_tokens`.
            If `min_new_tokens` is also set, the effect of `min_length` is overridden by `min_new_tokens`.
            Default: ``0`` .
        min_new_tokens (int, optional): The minimum numbers of tokens to generate,
            ignoring the number of tokens in the prompt. Default: ``None`` .
        do_sample (bool, optional): Whether to use sampling ; ``True`` means using sampling encoding,
            ``False`` means using greedy decoding. Default: ``False`` .
        use_past (bool, optional): Whether the model should use the past last key/values attentions
            (if applicable to the model) to speed up decoding. Default: ``False`` .
        temperature (float, optional): The value used to modulate the next token probabilities.
            Default: ``1.0`` .
        top_k (int, optional): The number of highest probability vocabulary tokens to keep for top-k-filtering.
            Default: ``50`` .
        top_p (float, optional): If set to ``float < 1``, only the smallest set of most probable tokens with
            probabilities that add up to `top_p` or higher are kept for generation. Default: ``1.0`` .
        repetition_penalty (float, optional): The parameter for repetition penalty. 1.0 means no penalty.
            See `this paper <https://arxiv.org/pdf/1909.05858.pdf>`_ for more details. Default: ``1.0`` .
        encoder_repetition_penalty (float, optional): The parameter for encoder_repetition_penalty.
            An exponential penalty on sequences that are not in the original input.
            1.0 means no penalty. Default: ``1.0`` .
        renormalize_logits (bool, optional): Whether to renormalize the logits after applying all the logits
            processors or warpers (including the custom ones). It's highly recommended to set this flag to `True` as
            the search algorithms suppose the score logits are normalized but some logit processors or warpers break
            the normalization. Default: ``False`` .
        output_scores (bool, optional): Whether to return the prediction scores before softmax.
            Default: ``False`` .
        output_logits (bool, optional): Whether to return the unprocessed prediction logit scores.
            Default: ``False`` .
        return_dict_in_generate (bool, optional): Whether to return a dictionary output instead of a
            tuple with output_ids. Only when this is set to True, can generate other output items besides output_ids.
            Default: ``False`` .
        pad_token_id (int, optional): The id of the padding token.
        bos_token_id (int, optional): The id of the beginning-of-sequence token.
        eos_token_id (Union[int, List[int]], optional): The id of the end-of-sequence token.
            Optionally, use a list to set multiple *end-of-sequence* tokens.

    Returns:
        Instance of GenerationConfig.

    Examples:
        >>> from mindformers.generation import GenerationConfig
        >>> config = GenerationConfig()
        >>> print(config)
        {'max_length': 20, 'max_new_tokens': None, 'min_length': 0, 'min_new_tokens': None, 'num_beams': 1,
        'do_sample': False, 'use_past': False, 'temperature': 1.0, 'top_k': 50, 'top_p': 1.0, 'repetition_penalty':
        1.0, 'encoder_repetition_penalty': 1.0, 'renormalize_logits': False, 'return_dict_in_generate': False,
        'output_scores': False, 'output_logits': False, 'pad_token_id': None, 'bos_token_id': None, 'eos_token_id':
        [], '_from_model_config': False}
        >>> config = GenerationConfig(max_length=100, min_length=10, do_sample=True, top_k=5, top_p=0.8)
        >>> print(config)
        {'max_length': 100, 'max_new_tokens': None, 'min_length': 10, 'min_new_tokens': None, 'num_beams': 1,
        'do_sample': True, 'use_past': False, 'temperature': 1.0, 'top_k': 5, 'top_p': 0.8, 'repetition_penalty':
        1.0, 'encoder_repetition_penalty': 1.0, 'renormalize_logits': False, 'return_dict_in_generate': False,
        'output_scores': False, 'output_logits': False, 'pad_token_id': None, 'bos_token_id': None, 'eos_token_id':
        [], '_from_model_config': False}
    """

    def __init__(self, **kwargs):
        # max generate length
        self.max_length = kwargs.pop("max_decode_length", 20)
        self.max_length = kwargs.pop("max_length", self.max_length)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)
        self.min_length = kwargs.pop("min_length", 0)
        self.min_new_tokens = kwargs.pop("min_new_tokens", None)

        # number of beams
        self.num_beams = kwargs.pop("num_beams", 1)
        # do sample or not
        self.do_sample = kwargs.pop("do_sample", False)
        # incremental infer
        self.use_past = kwargs.pop("use_past", False)
        # logits processors
        self.temperature = kwargs.pop("temperature", 1.0)
        self.top_k = kwargs.pop("top_k", 50)
        self.top_p = kwargs.pop("top_p", 1.0)
        self.repetition_penalty = kwargs.pop("repetition_penalty", 1.0)
        self.encoder_repetition_penalty = kwargs.pop("encoder_repetition_penalty", 1.0)
        self.renormalize_logits = kwargs.pop("renormalize_logits", False)

        # dictionary output
        self.return_dict_in_generate = kwargs.pop("return_dict_in_generate", False)
        self.output_scores = kwargs.pop("output_scores", False)
        self.output_logits = kwargs.pop("output_logits", False)

        # Special tokens that can be used at generation time
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", [])

        # parallel_decoding
        self.parallel_decoding = kwargs.pop("parallel_decoding", False)
        self.window_size = kwargs.pop("window_size", 5)
        self.level = kwargs.pop("level", 5)
        self.guess_set_size = kwargs.pop("guess_set_size", 3)

        if isinstance(self.eos_token_id, int):
            self.eos_token_id = [self.eos_token_id]

        # interface.
        self._from_model_config = kwargs.pop("_from_model_config", False)
        # Additional attributes without default values
        if not self._from_model_config:
            # we don't want to copy values from the model config
            # if we're initializing a `GenerationConfig` from a
            # model's default configuration file
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error("Can't set %s with value %s for %s", key, value, self)
                    raise err

    def __str__(self) -> str:
        return str(self.__dict__)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs):
        """
        Instantiates a [`GenerationConfig`] from a Python dictionary of parameters.

        Args:
            config_dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the configuration object.
            kwargs (`Dict[str, Any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            [`GenerationConfig`], the configuration object instantiated from those parameters.
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        config = cls(**{**config_dict, **kwargs})
        unused_kwargs = config.update(**kwargs)
        logger.debug("Generate config %s", config)
        if return_unused_kwargs:
            return config, unused_kwargs
        return config

    @classmethod
    def from_model_config(cls, model_config: PretrainedConfig) -> "GenerationConfig":
        """
        Instantiates a [`GenerationConfig`] from a [`PretrainedConfig`].
        This function is useful to convert legacy [`PretrainedConfig`] objects,
        which may contain generation parameters, into a stand-alone [`GenerationConfig`].

        Args:
            model_config (`PretrainedConfig`):
                The model config that will be used to instantiate the generation config.

        Returns:
            [`GenerationConfig`], the configuration object instantiated from those parameters.
        """
        config_dict = model_config.to_dict()
        config_dict.pop("_from_model_config", None)
        config = cls.from_dict(config_dict, return_unused_kwargs=False, _from_model_config=True)

        return config

    def update(self, **kwargs):
        """
        Updates attributes of this class instance with attributes from `kwargs`
        if they match existing atributtes, returning all the unused kwargs.

        Args:
            kwargs (`Dict[str, Any]`):
                Dictionary of attributes to tentatively update this class.

        Returns:
            `Dict[str, Any]`, dictionary containing all the key-value pairs
            that were not used to update the instance.
        """
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                to_remove.append(key)
        unused_kwargs = {key: value for key, value in kwargs.items() if key not in to_remove}
        return unused_kwargs

    def to_dict(self) -> Dict[str, Any]:
        """to dict convert function."""
        output = copy.deepcopy(self.__dict__)
        return output
