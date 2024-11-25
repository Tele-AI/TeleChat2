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

"""TextGenerationPipeline"""
from typing import Union, Optional

import mindspore
from mindspore import Model, Tensor

from ..mindformer_book import MindFormerBook
from ..models import PreTrainedModel, PreTrainedTokenizer
from ..tools.register import MindFormerModuleType, MindFormerRegister
from .base_pipeline import Pipeline

__all__ = ['TextGenerationPipeline']


@MindFormerRegister.register(MindFormerModuleType.PIPELINE, alias="text_generation")
class TextGenerationPipeline(Pipeline):
    r"""Pipeline for Text Generation

    Args:
        model (Union[PretrainedModel, Model]):
            The model used to perform task, the input should be a model instance inherited from PretrainedModel.
        tokenizer (Optional[PreTrainedTokenizer]):
            A tokenizer (None or PreTrainedTokenizer) for text processing.
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
        >>> from mindformers.pipeline import TextGenerationPipeline
        >>> from mindformers import AutoModel, AutoTokenizer
        >>> model = AutoModel.from_pretrained("gpt2")
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> text_generate = TextGenerationPipeline(model=model, tokenizer=tokenizer)
        >>> output = text_generate("I love Beijing, because ")
    """
    _support_list = MindFormerBook.get_pipeline_support_task_list()['text_generation'].keys()
    return_name = 'text_generation'

    def __init__(self, model: Union[PreTrainedModel, Model],
                 tokenizer: Optional[PreTrainedTokenizer] = None,
                 **kwargs):
        if tokenizer is None:
            raise ValueError(f"{self.__class__.__name__}"
                             " requires for a tokenizer.")

        super().__init__(model, tokenizer=tokenizer, **kwargs)
        self.model_name = kwargs.get("model_name", None)
        self.use_past = False
        if hasattr(self.network.config, "use_past"):
            self.use_past = self.network.config.use_past
        # only when incremental generate, set batch size as model config bs
        if self.use_past and hasattr(self.network.config, "batch_size") and self.network.config.batch_size is not None:
            self._batch_size = self.network.config.batch_size

    def _sanitize_parameters(self, **pipeline_parameters):
        r"""Sanitize Parameters

        Args:
            pipeline_parameters (Optional[dict]):
                The parameter dict to be parsed.
        """
        preprocess_keys = ['keys', 'add_special_tokens']
        preprocess_params = {}
        for item in preprocess_keys:
            if item in pipeline_parameters:
                preprocess_params[item] = pipeline_parameters.pop(item)

        postprocess_params = {}

        # all other pipeline_parameters are passed to text generator to handle
        forward_kwargs = pipeline_parameters
        # always use dictionary output
        forward_kwargs["return_dict_in_generate"] = True

        return preprocess_params, forward_kwargs, postprocess_params

    def preprocess(self, inputs: Union[str, dict, Tensor],
                   **preprocess_params):
        r"""The Preprocess For Translation

        Args:
            inputs (Union[str, dict, Tensor]):
                The text to be classified.
            preprocess_params (dict):
                The parameter dict for preprocess.

        Return:
            Processed text.
        """
        add_special_tokens = preprocess_params.get('add_special_tokens', True)
        if isinstance(inputs, dict):
            keys = preprocess_params.get('keys', None)
            default_src_language_name = 'text'
            feature_name = keys.get('src_language', default_src_language_name) if keys else default_src_language_name

            inputs = inputs[feature_name]
            if isinstance(inputs, mindspore.Tensor):
                inputs = inputs.asnumpy().tolist()
        # for batch inputs, pad to longest
        input_ids = self.tokenizer(inputs,
                                   return_tensors=None,
                                   add_special_tokens=add_special_tokens,
                                   padding=True)["input_ids"]
        return {"input_ids": input_ids}

    def run_multi(self, inputs: Union[list, tuple],
                  batch_size: int,
                  preprocess_params: dict,
                  forward_params: dict,
                  postprocess_params: dict):
        r"""Run Multiple Method
        This function is used to run a list input for task.

        Args:
            inputs (Union[list, tuple, iterator]):
                The iterable input for pipeline.
            batch_size (int):
                Batch size of pipeline input.
            preprocess_params (dict):
                The parameter dict for preprocess.
            forward_params (dict):
                The parameter dict for model forward process.
            postprocess_params (dict):
                The parameter dict for postprocess.
        """
        if self.use_past and self._batch_size != batch_size:
            raise ValueError("When using text generation pipeline with use_past model, "
                             f"the batch size of input list {batch_size} should be consistent with "
                             f"model batch size {self._batch_size}. Please check your inputs.")
        if len(inputs) % batch_size != 0:
            raise ValueError(f"When running multi input pipeline, the length of inputs {len(inputs)}"
                             f" should be multiple of batch size {batch_size}. Please check yout inputs.")
        outputs = []
        if batch_size > 1:
            batch_inputs = [inputs[i:i + batch_size] for i in range(0, len(inputs), batch_size)]
        else:
            batch_inputs = inputs
        for item in batch_inputs:
            outputs.extend(self.run_single(item, preprocess_params,
                                           forward_params, postprocess_params))
        return outputs

    def _forward(self, model_inputs: dict,
                 **forward_params):
        r"""The Forward Process of Model

        Args:
            model_inputs (dict):
                The output of preprocess.
            forward_params (dict):
                The parameter dict for model forward.
        """
        forward_params.pop("None", None)
        input_ids = model_inputs["input_ids"]
        result = self.network.generate(input_ids, **forward_params)
        return result

    def postprocess(self, model_outputs: dict,
                    **postprocess_params):
        r"""Postprocess

        Args:
            model_outputs (dict):
                Outputs of forward process.

        Return:
            Translation results.
        """
        outputs = self.tokenizer.decode(model_outputs["sequences"], skip_special_tokens=True)
        return [{self.return_name + '_text': outputs}]
