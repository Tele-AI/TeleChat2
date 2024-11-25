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

"""TranslationPipeline"""
from typing import Optional, Union

import mindspore
from mindspore import Tensor, Model

from mindformers.mindformer_book import MindFormerBook
from .base_pipeline import Pipeline
from ..tools.register import MindFormerRegister, MindFormerModuleType
from ..models import PreTrainedModel, PreTrainedTokenizer

__all__ = ['TranslationPipeline']


@MindFormerRegister.register(MindFormerModuleType.PIPELINE, alias="translation")
class TranslationPipeline(Pipeline):
    """Pipeline for Translation

    Args:
        model (Union[PretrainedModel, Model]):
            The model used to perform task, the input should be a model instance inherited from PretrainedModel.
        tokenizer (Optional[PreTrainedTokenizer]):
            A tokenizer (None or PreTrainedTokenizer) for text processing. Default: None.
        **kwargs:
            Specific parametrization of `generate_config` and/or additional model-specific kwargs that will be
            forwarded to the `forward` function of the model. Supported `generate_config` keywords can be
            checked in [`GenerationConfig`]'s documentation. Mainly used Keywords are shown below:

            max_length(int): The maximum length the generated tokens can have. Corresponds to the length of
                the input prompt + `max_new_tokens`. Its effect is overridden by `max_new_tokens`, if also set.
            max_new_tokens (int): The maximum numbers of tokens to generate, ignoring the number of
                tokens in the prompt.
            do_sample(bool): Whether to do sampling on the candidate ids.
                If set True it will be enabled, and set it to be False to disable the sampling,
                equivalent to topk 1.
                If set None, it follows the setting in the configureation in the model.
            top_k(int): Determine the topK numbers token id as candidate. This should be a positive number.
                If set None, it follows the setting in the configureation in the model.
            top_p(float): The accumulation probability of the candidate token ids below the top_p
                will be select as the condaite ids. The valid value of top_p is between (0, 1]. If the value
                is larger than 1, top_K algorithm will be enabled. If set None, it follows the setting in the
                configureation in the model.
            eos_token_id(int): The end of sentence token id. If set None, it follows the setting in the
                configureation in the model.
            pad_token_id(int): The pad token id. If set None, it follows the setting in the configureation
                in the model.
            repetition_penalty(float): The penalty factor of the frequency that generated words. The If set 1,
                the repetition_penalty will not be enabled. If set None, it follows the setting in the
                configureation in the model. Default None.

    Raises:
        TypeError:
            If input model and tokenizer's types are not corrected.
        ValueError:
            If the input model is not in support list.

    Examples:
        >>> from mindformers.pipeline import TranslationPipeline
        >>> from mindformers import AutoModel, AutoTokenizer
        >>> model = AutoModel.from_pretrained("t5_small")
        >>> tokenizer = AutoTokenizer.from_pretrained("t5_small")
        >>> translator = TranslationPipeline(model=model, tokenizer=tokenizer)
        >>> output = translator("translate the English to Romanian: a good boy!")
        >>> print(output)
        [{'translation_text': ['un băiat bun!']}]
    """
    _support_list = MindFormerBook.get_model_support_list()['t5']
    return_name = 'translation'

    def __init__(self, model: Union[PreTrainedModel, Model],
                 tokenizer: Optional[PreTrainedTokenizer] = None,
                 **kwargs):

        if tokenizer is None:
            raise ValueError(f"{self.__class__.__name__}"
                             " requires for a tokenizer.")

        super().__init__(model, tokenizer=tokenizer, **kwargs)

    def _sanitize_parameters(self, **pipeline_parameters):
        """Sanitize Parameters

        Args:
            pipeline_parameters (Optional[dict]):
                The parameter dict to be parsed.
        """
        preprocess_keys = ['keys']
        preprocess_params = {}
        for item in preprocess_keys:
            if item in pipeline_parameters:
                preprocess_params[item] = pipeline_parameters.get(item)

        postprocess_params = {}

        forward_key_name = ['top_k', 'top_p', 'do_sample', 'eos_token_id', 'repetition_penalty', 'max_length']
        forward_kwargs = {}
        for item in forward_key_name:
            if item in pipeline_parameters:
                forward_kwargs[item] = pipeline_parameters.get(item)
        # always use dictionary output
        forward_kwargs["return_dict_in_generate"] = True
        return preprocess_params, forward_kwargs, postprocess_params

    def preprocess(self, inputs: Union[str, dict, Tensor],
                   **preprocess_params):
        """The Preprocess For Translation

        Args:
            inputs (Union[str, dict, Tensor]):
                The text to be classified.
            preprocess_params (dict):
                The parameter dict for preprocess.

        Return:
            Processed text.
        """
        if isinstance(inputs, dict):
            keys = preprocess_params.get('keys', None)
            default_src_language_name = 'text'
            feature_name = keys.get('src_language', default_src_language_name) if keys else default_src_language_name

            inputs = inputs[feature_name]
            if isinstance(inputs, mindspore.Tensor):
                inputs = inputs.asnumpy().tolist()
        input_ids = self.tokenizer(inputs, return_tensors=None)["input_ids"]
        return {"input_ids": input_ids}

    def _forward(self, model_inputs: dict,
                 **forward_params):
        """The Forward Process of Model

        Args:
            model_inputs (dict):
                The output of preprocess.
            forward_params (dict):
                The parameter dict for model forward.

        Return:
            Dict of output_ids.
        """
        forward_params.pop("None", None)
        input_ids = model_inputs["input_ids"]
        result = self.network.generate(input_ids, **forward_params)
        return result

    def postprocess(self, model_outputs: dict,
                    **postprocess_params):
        """Postprocess

        Args:
            model_outputs (dict):
                Outputs of forward process.

        Return:
            Translation results.
        """
        outputs = self.tokenizer.decode(model_outputs["sequences"], skip_special_tokens=True)
        return [{self.return_name + '_text': outputs}]
