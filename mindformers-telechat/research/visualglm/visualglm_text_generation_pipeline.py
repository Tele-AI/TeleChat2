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
"""visualglm Image to text generation Pipeline adaptor."""
import os
import re
from collections import OrderedDict
from typing import Optional, Union

import numpy as np
from PIL import Image
from mindspore import Tensor, Model

from mindformers import AutoProcessor, AutoModel
from mindformers.mindformer_book import MindFormerBook
from mindformers.models import PreTrainedModel, BaseImageProcessor
from mindformers.pipeline.base_pipeline import Pipeline
from mindformers.tools.image_tools import load_image
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType

__all__ = ['VisualGLMImageToTextGenerationPipeline', 'register_pipeline_task']


def register_pipeline_task():
    """ register pipeline task for visualglm """
    MindFormerBook.get_pipeline_support_task_list()['visualglm_image_to_text_generation'] = OrderedDict([
        ("visualglm_6b", os.path.join(
            MindFormerBook.get_project_path(), "research/visualglm/run_visualglm_6b_image_to_text_generation.yaml"))])
    MindFormerBook.get_trainer_support_task_list()['visualglm_image_to_text_generation'] = OrderedDict([
        ("visualglm_6b", os.path.join(
            MindFormerBook.get_project_path(), "research/visualglm/run_visualglm_6b_image_to_text_generation.yaml"))])


@MindFormerRegister.register(MindFormerModuleType.PIPELINE, alias="visualglm_image_to_text_generation")
class VisualGLMImageToTextGenerationPipeline(Pipeline):
    r"""Visualglm pipeline for image to text generation

    Args:
        model (Union[str, PreTrainedModel]): The model used to perform task,
            the input could be a supported model name, or a model instance
            inherited from PreTrainedModel.
        image_processor (Optional[BaseImageProcessor]): The image_processor of model,
            it could be None if the model do not need image_processor.

    Raises:
        TypeError: If input model and image_processor's types are not corrected.
        ValueError: If the input model is not in support list.
    """
    _support_list = MindFormerBook.get_pipeline_support_task_list()['image_to_text_generation'].keys()

    def __init__(self, model: Union[str, PreTrainedModel, Model],
                 image_processor: Optional[BaseImageProcessor] = None,
                 tokenizer=None,
                 **kwargs):

        if isinstance(model, str):
            if model in self._support_list:
                if image_processor is None:
                    image_processor = AutoProcessor.from_pretrained(model).image_processor
                if not isinstance(image_processor, BaseImageProcessor):
                    raise TypeError(f"image_processor should be inherited from"
                                    f" BaseImageProcessor, but got {type(image_processor)}.")
                model = AutoModel.from_pretrained(model)
            else:
                raise ValueError(f"{model} is not supported by ImageToTextGenerationPipeline,"
                                 f"please selected from {self._support_list}.")

        if not isinstance(model, (PreTrainedModel, Model)):
            raise TypeError(f"model should be inherited from PreTrainedModel or Model, but got type {type(model)}.")

        if image_processor is None:
            raise ValueError("ImageToTextGenerationPipeline"
                             " requires for a image_processor.")
        self.hypothesis_template = kwargs.pop("hypothesis_template", "{}")
        super().__init__(model.set_train(mode=False), image_processor=image_processor, tokenizer=tokenizer, **kwargs)

    def _sanitize_parameters(self, **pipeline_parameters):
        r"""Sanitize Parameters

        Args:
            pipeline_parameters (Optional[dict]): The parameter dict to be parsed.
        """
        preprocess_params = {}
        postprocess_params = {}
        forward_params = {}

        post_list = ["top_k"]
        pre_list = ["hypothesis_template", "max_length", "padding"]
        forward_list = ['top_k', 'top_p', 'do_sample', 'eos_token_id', 'repetition_penalty', 'max_length', 'seed']
        for item in post_list:
            if item in pipeline_parameters:
                postprocess_params[item] = pipeline_parameters.get(item)

        for item in pre_list:
            if item in pipeline_parameters:
                preprocess_params[item] = pipeline_parameters.get(item)

        for item in forward_list:
            if item in pipeline_parameters:
                forward_params[item] = pipeline_parameters.get(item)

        return preprocess_params, forward_params, postprocess_params

    @staticmethod
    def generate_glm_prompt(unhandled_prompts, history=None, english=False):
        """ generate glm prompt from raw prompt """
        if history is None:
            history = []
        post_prompts, image_positions = [], []
        for query in unhandled_prompts:
            prompt = "</img>"
            if english:
                for _, (old_query, response) in enumerate(history):
                    prompt += "Q:{}\nA:{}\n".format(old_query, response)
                prompt += "Q:{}\nA:".format(query)
            else:
                for _, (old_query, response) in enumerate(history):
                    prompt += "问：{}\n答：{}\n".format(old_query, response)
                prompt += "问：{}\n答：".format(query)
            post_prompts.append(prompt)
        pre_prompts = ["<img>"] * len(post_prompts)
        image_positions = [len("<img>")] * len(post_prompts)
        return pre_prompts, post_prompts, image_positions

    def handle_prompt(self, prompt, image_size):
        if not prompt:
            raw_prompts = [""] * image_size
        else:
            raw_prompts = prompt.split(',')

        # handle prompt using chatglm type
        pre_prompts, post_prompts, image_positions = self.generate_glm_prompt(raw_prompts)

        return pre_prompts, post_prompts, image_positions

    def preprocess(self, inputs: (Union[str, dict, Image.Image, Tensor, np.ndarray]),
                   **preprocess_params):
        r"""The Preprocess For Task

        Args:
            inputs (Union[url, dict, PIL.Image, tensor, numpy]): Inputs used to generate text, including image,
                and prompt (if provided).
            preprocess_params (dict): The parameter dict for preprocess.

        Return:
            Processed image and prompt.
        """
        if isinstance(inputs, dict):
            image = inputs['image']
            prompt = inputs.get('prompt', None)
        else:
            image = inputs
            prompt = ""

        if self._batch_size is None:
            batch_size = 1
        else:
            batch_size = self._batch_size

        image_size = 1
        print(f"batch_size: {self._batch_size}")
        if isinstance(image, str):
            image = image.split(',')
            image_size = len(image)
            if batch_size > 1:
                diff = batch_size - image_size
                if diff > 0:
                    extend_filepath = [image[-1]] * diff
                    image.extend(extend_filepath)
                else:
                    image = image[:batch_size]
            image_list = [load_image(filepath) for filepath in image]
        else:
            image_list = [image]

        pre_prompts, post_prompts, image_positions = self.handle_prompt(prompt, image_size)
        if batch_size > 1:
            diff = batch_size - image_size
            if diff > 0:
                extend_pre_prompt = [pre_prompts[-1]] * diff
                extend_post_prompt = [post_prompts[-1]] * diff
                extend_positions = [image_positions[-1]] * diff
                pre_prompts.extend(extend_pre_prompt)
                post_prompts.extend(extend_post_prompt)
                image_positions.extend(extend_positions)
            else:
                pre_prompts = pre_prompts[:batch_size]
                post_prompts = post_prompts[:batch_size]

        image_processed = self.image_processor(image_list)

        max_length = preprocess_params.pop("max_length", 32)
        padding = preprocess_params.pop("padding", "max_length")

        pre_input_ids = self.tokenizer(pre_prompts, add_special_tokens=False, return_tensors="ms")["input_ids"]
        post_input_ids = self.tokenizer(post_prompts,
                                        max_length=max_length - len(pre_input_ids[0]),
                                        padding=padding,
                                        return_tensors="ms")["input_ids"]

        return {"image_processed": image_processed, "pre_input_ids": pre_input_ids, "post_input_ids": post_input_ids}

    def forward(self, model_inputs: dict,
                **forward_params):
        r"""The Forward Process of Model

        Args:
            model_inputs (dict): The output of preprocess.
            forward_params (dict): The parameter dict for model forward.
        """
        del forward_params
        image_processed = model_inputs["image_processed"]
        pre_input_ids = model_inputs["pre_input_ids"]
        post_input_ids = model_inputs["post_input_ids"]

        output_ids_per_image = self.network.generate_text_for_image(image_processed, pre_input_ids, post_input_ids)
        return {"output_ids": output_ids_per_image}

    @staticmethod
    def process_response(response_list):
        """ get standard response """
        handled_response = []
        for response in response_list:
            response = response.strip()
            response = response.replace("[[训练时间]]", "2023年")
            punkts = [
                [",", "，"],
                ["!", "！"],
                [":", "："],
                [";", "；"],
                [r"\?", "？"],
            ]
            for item in punkts:
                response = re.sub(r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1], response)
                response = re.sub(r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1], response)
            response = response.split('答：')[-1].strip()
            handled_response.append(response)
        return handled_response

    def postprocess(self, model_outputs, **postprocess_params):
        """ post process """
        del postprocess_params
        output_ids = model_outputs["output_ids"]
        outputs = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        outputs = self.process_response(outputs)
        return outputs
