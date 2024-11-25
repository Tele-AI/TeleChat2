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
"""Image Classification Pipeline API."""
from typing import Optional, Union

import numpy as np
from PIL import Image

from mindspore.ops import operations as P
from mindspore import Tensor, Model

from mindformers.mindformer_book import MindFormerBook
from mindformers.models import PreTrainedModel, BaseImageProcessor
from mindformers.tools.image_tools import load_image
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.dataset.labels import labels
from .base_pipeline import Pipeline


@MindFormerRegister.register(MindFormerModuleType.PIPELINE, alias="image_classification")
class ImageClassificationPipeline(Pipeline):
    r"""Pipeline for image classification

    Args:
        model (Union[PretrainedModel, Model]):
            The model used to perform task, the input should be a model instance inherited from PretrainedModel.
        image_processor (Optional[BaseImageProcessor]):
            The image_processor of model, it could be None if the model do not need image_processor.

    Raises:
        TypeError:
            If input model and image_processor's types are not corrected.
        ValueError:
            If the input model is not in support list.

    Examples:
        >>> import numpy as np
        >>> from mindformers.pipeline import ImageClassificationPipeline
        >>> from mindformers import AutoModel, ViTImageProcessor
        >>> model = AutoModel.from_pretrained("vit_base_p16")
        >>> processor = ViTImageProcessor(size=224)
        >>> classifier = ImageClassificationPipeline(
        ...     model=model,
        ...     image_processor=processor,
        ...     top_k=5
        ...     )
        >>> classifier(np.uint8(np.random.random((5, 3, 255, 255))))
            [[{'score': 0.0016654134, 'label': 'matchstick'},
            {'score': 0.0015071577, 'label': 'theater curtain'},
            {'score': 0.0014839625, 'label': 'ocarina'},
            {'score': 0.0014319294, 'label': 'abaya'},
            {'score': 0.0014109017, 'label': 'bottlecap'}],
            ..., {'score': 0.0014109018, 'label': 'bottlecap'}]]
    """
    _support_list = MindFormerBook.get_pipeline_support_task_list()['image_classification'].keys()

    def __init__(self, model: Union[PreTrainedModel, Model],
                 image_processor: Optional[BaseImageProcessor] = None,
                 **kwargs):

        if image_processor is None:
            raise ValueError("ImageClassificationFoPipeline"
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

        post_list = ["top_k", "candidate_labels"]
        for item in post_list:
            if item in pipeline_parameters:
                postprocess_params[item] = pipeline_parameters.get(item)

        return preprocess_params, {}, postprocess_params

    def preprocess(self, inputs: (Union[str, Image.Image, Tensor, np.ndarray]),
                   **preprocess_params):
        r"""The Preprocess For Task

        Args:
            inputs (Union[url, PIL.Image, tensor, numpy]):
                The image to be classified.
            preprocess_params (dict):
                The parameter dict for preprocess.

        Returns:
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

        Returns:
            Probs of classification
        """
        forward_params.pop("None", None)

        image_processed = model_inputs["image_processed"]

        logits_per_image = self.network(image_processed)[0]
        probs = P.Softmax()(logits_per_image).asnumpy()
        return {"probs": probs}

    def postprocess(self, model_outputs, **postprocess_params):
        r"""Postprocess

        Args:
            model_outputs (dict):
                Outputs of forward process.
            top_k (Optional[int]):
                Return top_k probs of result. Default: 3.

        Returns:
            outputs (list):
                Classification results.
        """
        top_k = postprocess_params.pop("top_k", 3)
        candidate_labels = postprocess_params.pop("candidate_labels", 'imagenet')

        scores = model_outputs['probs']

        outputs = []
        if isinstance(candidate_labels, str):
            inputs_labels = labels.get(candidate_labels)
        elif isinstance(candidate_labels, list):
            inputs_labels = candidate_labels
        else:
            raise ValueError(f"The candidate_labels should be dataset name (str) or custom labels (list)"
                             f" but got {type(candidate_labels)}")

        if inputs_labels is None:
            raise ValueError(f"The custom candidate_labels is None or "
                             f"the input dataset labels name is not supported yet.")

        for score in scores:
            sorted_res = sorted(zip(score, inputs_labels), key=lambda x: -x[0])
            if top_k is not None:
                sorted_res = sorted_res[:min(top_k, len(inputs_labels))]
            outputs.append([{"score": score_item, "label": label}
                            for score_item, label in sorted_res])
        return outputs
