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
"""Multi-Modal to Text Generation Pipeline API."""
from typing import Optional, Union, Dict, List

from mindspore import Model
from mindformers.mindformer_book import MindFormerBook
from mindformers.models import PreTrainedModel, BaseXModalToTextProcessor
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType

from .base_pipeline import Pipeline

__all__ = ['MultiModalToTextPipeline']


@MindFormerRegister.register(MindFormerModuleType.PIPELINE, alias="multi_modal_to_text_generation")
class MultiModalToTextPipeline(Pipeline):
    r"""Pipeline for multi-modal to text generation.

    Args:
        model (Union[PretrainedModel, Model]):
            The model used to perform task, the input should be a model instance inherited from PretrainedModel.
        processor (BaseXModalToTextProcessor, optional):
            The image_processor of model, it could be None if the model do not need image_processor. Default: ``None`` .

    Returns:
        A pipeline for MultiModalToTextPipeline.

    Raises:
        TypeError:
            If input model and image_processor's types are not corrected.
        ValueError:
            If the input model is not in support list.

    Examples:
        >>> import os
        >>> import mindspore as ms
        >>> from mindformers import build_context
        >>> from mindformers import AutoModel, AutoTokenizer, pipeline, AutoProcessor, MindFormerConfig, AutoConfig
        >>> os.environ['USE_ROPE_SELF_DEFINE'] = 'True'
        >>> inputs = [[{"image": "/path/to/example.jpg"}, {"text": "Please describe this image."}]]
        >>> # Note:
        >>> #     "image": is an image path
        >>> model_path = "/path/to/cogvlm2_mode_path"
        >>> # Note:
        >>> #     mode_path: a new folder (containing configs/cogvlm2/predict_cogvlm2_image_llama3_chat_19b.yaml)
        >>> config_path = "/path/to/cogvlm2_mode_path/predict_cogvlm2_image_llama3_chat_19b.yaml"
        >>> # Note:
        >>> #     config_path: the predict_cogvlm2_image_llama3_chat_19b.yaml path in mode_path
        >>> #     Please change the value of 'vocab_file' in predict_cogvlm2_image_llama3_chat_19b.yaml
        >>> #     to the value of 'tokenizer.model'.
        >>> config = MindFormerConfig(config_path)
        >>> build_context(config)
        >>> model_config = AutoConfig.from_pretrained(config_path)
        >>> tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
        >>> model = AutoModel.from_config(model_config)
        >>> processor = AutoProcessor.from_pretrained(config_path, trust_remote_code=True, use_fast=True)
        >>> param_dict = ms.load_checkpoint("/path/to/cogvlm2image.ckpt")
        >>> _, not_load = ms.load_param_into_net(model, param_dict)
        >>> text_generation_pipeline = pipeline(task="multi_modal_to_text_generation",
        >>>                                     model=model, processor=processor)
        >>> outputs = text_generation_pipeline(inputs, max_length=model_config.max_decode_length,
        >>>                                    do_sample=False, top_k=model_config.top_k,top_p=model_config.top_p)
        >>> for output in outputs:
        >>>     print(output)
        Question: Please describe this image. Answer:This image is an apple.
        >>> # Note:
        >>> #     The final result shall be subject to the actual input image.
    """
    _support_list = MindFormerBook.get_pipeline_support_task_list()['multi_modal_to_text_generation'].keys()

    def __init__(self, model: Union[PreTrainedModel, Model],
                 processor: Optional[BaseXModalToTextProcessor] = None,
                 **kwargs):
        if processor is None:
            raise ValueError("MultiModalToTextPipeline requires a processor.")
        self.processor = processor
        self.output_columns = self.processor.output_columns
        super().__init__(model, processor=processor, **kwargs)

    def _sanitize_parameters(self, **pipeline_parameters):
        r"""Sanitize Parameters

        Args:
            pipeline_parameters (Optional[dict]):
                The parameter dict to be parsed.
        """
        preprocess_params = {}
        postprocess_params = {}
        forward_params = {}
        return preprocess_params, forward_params, postprocess_params

    def preprocess(self, inputs: Union[List[Dict[str, str]], List[List[Dict[str, str]]]],
                   **preprocess_params):
        r"""The Preprocess For Task

        Args:
            inputs (Union[List[Dict[str, str]], List[List[Dict[str, str]]]]):
                Inputs used to generate text.
            preprocess_params (dict):
                The parameter dict for preprocess.

        Return:
            Processed image, input_ids, img_pos.
        """
        process_res = self.processor(query_list=inputs)
        if "raw_query" in process_res:
            process_res.pop("raw_query")

        return process_res

    def _forward(self, model_inputs: dict,
                 **forward_params):
        r"""The Forward Process of Model

        Args:
            model_inputs (dict):
                The output of preprocess.
            forward_params (dict):
                The parameter dict for model forward.
        """
        output_ids = self.network.generate(**model_inputs, **forward_params)
        return {"output_ids": output_ids}

    def postprocess(self, model_outputs, **postprocess_params):
        output_ids = model_outputs["output_ids"]
        return self.processor.post_process(output_ids, **postprocess_params)
