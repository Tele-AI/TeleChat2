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
"""Image to Text Generation Pipeline API."""
from typing import Optional, Union, Dict, List

from mindspore import Model
from mindformers.mindformer_book import MindFormerBook
from mindformers.models import PreTrainedModel, BaseProcessor
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType

from .base_pipeline import Pipeline

__all__ = ['ImageToTextPipeline']


@MindFormerRegister.register(MindFormerModuleType.PIPELINE, alias="image_to_text_generation")
class ImageToTextPipeline(Pipeline):
    r"""Pipeline for image to text generation

    Args:
        model (Union[PretrainedModel, Model]):
            The model used to perform task, the input should be a model instance inherited from PretrainedModel.
        processor (Optional[BaseProcessor]):
            The image_processor of model, it could be None if the model do not need image_processor.

    Raises:
        TypeError:
            If input model and image_processor's types are not corrected.
        ValueError:
            If the input model is not in support list.
    """
    _support_list = MindFormerBook.get_pipeline_support_task_list()['image_to_text_generation'].keys()

    def __init__(self, model: Union[PreTrainedModel, Model],
                 processor: Optional[BaseProcessor] = None,
                 **kwargs):
        if processor is None:
            raise ValueError("ImageToTextPipeline requires a processor.")
        self.processor = processor
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
        processed_res = self.processor(text_input=inputs)

        return {
            "query": inputs,
            "image": processed_res.get("image"),
            "input_ids": processed_res.get("input_ids"),
            "img_pos": processed_res.get("img_pos")
        }

    def _forward(self, model_inputs: dict,
                 **forward_params):
        r"""The Forward Process of Model

        Args:
            model_inputs (dict):
                The output of preprocess.
            forward_params (dict):
                The parameter dict for model forward.
        """
        query = model_inputs["query"]
        image = model_inputs["image"]
        input_ids = model_inputs["input_ids"]
        img_pos = model_inputs["img_pos"]

        output_ids_per_image = self.network.generate(input_ids=input_ids, images=image, img_pos=img_pos)
        return {"output_ids": output_ids_per_image, "query": query}

    def postprocess(self, model_outputs, **postprocess_params):
        output_ids = model_outputs["output_ids"]
        query = model_outputs["query"]
        outputs = self.tokenizer.decode(output_ids, skip_special_tokens=False)
        return self.processor.post_process(outputs, [query])
