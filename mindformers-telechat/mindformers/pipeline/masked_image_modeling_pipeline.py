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
"""Masked Image Modeling Pipeline API."""
import os
from typing import Optional, Union

import numpy as np
from PIL import Image

from mindspore import Tensor, Model

from mindformers.mindformer_book import MindFormerBook
from mindformers.models import PreTrainedModel, BaseImageProcessor
from mindformers.tools.image_tools import load_image
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from .base_pipeline import Pipeline

__all__ = ['MaskedImageModelingPipeline']

from ..tools.utils import LOCAL_DEFAULT_PATH


@MindFormerRegister.register(MindFormerModuleType.PIPELINE, alias="masked_image_modeling")
class MaskedImageModelingPipeline(Pipeline):
    r"""Pipeline for masked image modeling

    Args:
        model (Union[PretrainedModel, Model]):
            The model used to perform task, the input should be a model instance inherited from PretrainedModel.
        image_processor (Optional[BaseImageProcessor]):
            The image_processor of model, it could be None if the model do not need image_processor.

    Raises:
        TypeError: If input model and image_processor's types are not corrected.
        ValueError: If the input model is not in support list.

    Examples:
        >>> import numpy as np
        >>> from mindformers.pipeline import MaskedImageModelingPipeline
        >>> from mindformers import AutoModel, ViTMAEImageProcessor
        >>> model = AutoModel.from_pretrained('mae_vit_base_p16')
        >>> processor = ViTMAEImageProcessor(size=224)
        >>> reconstructor = MaskedImageModelingPipeline(
        ...     model=model,
        ...     image_processor=processor,
        ...     top_k=5
        ...     )
        >>> reconstructor(np.uint8(np.random.random((5, 3, 255, 255))))
    """
    _support_list = MindFormerBook.get_pipeline_support_task_list()['masked_image_modeling'].keys()

    def __init__(self, model: Union[PreTrainedModel, Model],
                 image_processor: Optional[BaseImageProcessor] = None,
                 **kwargs):

        if image_processor is None:
            raise ValueError("MaskedImageModelingPipeline"
                             " requires for a image_processor.")

        super().__init__(model, image_processor=image_processor, **kwargs)

    def _sanitize_parameters(self, **pipeline_parameters):
        r"""Sanitize Parameters

        Args:
            pipeline_parameters (Optional[dict]):
                The parameter dict to be parsed.
        """
        preprocess_params = {}
        postprocess_params = {}

        post_list = []
        for item in post_list:
            if item in pipeline_parameters:
                postprocess_params[item] = pipeline_parameters.get(item)

        return preprocess_params, {}, postprocess_params

    def preprocess(self, inputs: (Union[str, Image.Image, Tensor, np.ndarray]),
                   **preprocess_params):
        r"""The Preprocess For Task

        Args:
            inputs (Union[url, PIL.Image, tensor, numpy]):
                The image to be reconstructed.
            preprocess_params (dict):
                The parameter dict for preprocess.

        Return:
            Processed image.
        """
        if isinstance(inputs, dict):
            inputs = inputs['image']
        if isinstance(inputs, str):
            inputs = load_image(inputs)

        image_processed = self.image_processor(inputs)
        return {"image_processed": image_processed}

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

        image_processed = model_inputs["image_processed"]

        images_array = self.network(*image_processed)
        images_array = images_array.asnumpy()
        return {"images_array": images_array}

    def postprocess(self, model_outputs, **postprocess_params):
        r"""Postprocess

        Args:
            model_outputs (dict):
                Outputs of forward process.

        Return:
            classification results.
        """
        images_array = model_outputs['images_array']

        outputs = []
        for i, image_array in enumerate(images_array):
            image_array = image_array.transpose((1, 2, 0))
            mean = np.array(self.image_processor.normalize.mean)
            std = np.array(self.image_processor.normalize.std)
            image_array = image_array * std + mean
            image_array = image_array * 255
            image_array = image_array.astype(np.uint8)
            reconstruct_image = Image.fromarray(image_array)
            image_path = os.path.join(LOCAL_DEFAULT_PATH, f'output_image{i}.jpg')
            reconstruct_image.save(image_path)
            outputs.append({"info": image_path, "data": reconstruct_image})

        return outputs
