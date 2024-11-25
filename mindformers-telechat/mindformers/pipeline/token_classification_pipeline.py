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
# This file was refer to project:
# https://github.com/lonePatient/daguan_2019_rank9/blob/master/pydatagrand/train/ner_utils.py
# ============================================================================

"""TokenClassificationPipeline"""

import numpy as np
from mindspore import ops
from ..mindformer_book import MindFormerBook
from .base_pipeline import Pipeline
from ..tools.register import MindFormerRegister, MindFormerModuleType

__all__ = ['TokenClassificationPipeline']


@MindFormerRegister.register(MindFormerModuleType.PIPELINE, alias="token_classification")
class TokenClassificationPipeline(Pipeline):
    """Pipeline for token classification

    Args:
        model (Union[PretrainedModel, Model]):
            The model used to perform task, the input should be a model instance inherited from PretrainedModel.
        tokenizer (Optional[PreTrainedTokenizerBase]):
            A tokenizer (None or PreTrainedTokenizer) for text processing. Default: None.
        id2label (dict):
            A dict which maps label id to label str.

    Raises:
        TypeError:
            If input model and image_processor's types are not corrected.
        ValueError:
            If the input model is not in support list.

    Examples:
        >>> from mindformers.pipeline import TokenClassificationPipeline
        >>> from mindformers import AutoTokenizer, BertForTokenClassification, AutoConfig
        >>> from mindformers.dataset.labels import cluener_labels
        >>> id2label = {label_id: label for label_id, label in enumerate(cluener_labels)}
        >>> input_data = ["表身刻有代表日内瓦钟表匠freresoltramare的“fo”字样。"]
        >>> tokenizer = AutoTokenizer.from_pretrained('tokcls_bert_base_chinese_cluener')
        >>> ner_dense_cluener_config = AutoConfig.from_pretrained('tokcls_bert_base_chinese_cluener')
        >>> model = BertForTokenClassification(ner_dense_cluener_config)
        >>> tokcls_pipeline = TokenClassificationPipeline(task='token_classification',
        ...                                               model=model,
        ...                                               id2label=id2label,
        ...                                               tokenizer=tokenizer,
        ...                                               max_length=model.config.seq_length,
        ...                                               padding="max_length")
        >>> results = tokcls_pipeline(input_data)
        >>> print(results)
            [[{'entity_group': 'address', 'start': 6, 'end': 8, 'score': 0.52329, 'word': '日内瓦'},
              {'entity_group': 'name', 'start': 12, 'end': 25, 'score': 0.83922, 'word': 'freresoltramar'}]]
    """
    _support_list = MindFormerBook.get_pipeline_support_task_list()['token_classification'].keys()

    def __init__(self, model, id2label, tokenizer=None, **kwargs):

        if tokenizer is None:
            raise ValueError(f"{self.__class__.__name__}"
                             " requires for a tokenizer.")

        if id2label is None:
            raise ValueError(f"{self.__class__.__name__}"
                             " requires for a dict which maps label id to label str.")

        self.id2label = id2label
        self.input_text = ""

        super().__init__(model, tokenizer=tokenizer, **kwargs)

    def _sanitize_parameters(self, **pipeline_parameters):
        """sanitize parameters for preprocess, forward, and postprocess."""
        if 'batch_size' in pipeline_parameters:
            raise ValueError(f"The {self.__class__.__name__} does not support batch inference, please remove the "
                             f"batch_size")

        postprocess_params = {'id2label'}

        preprocess_key_name = ['max_length', 'padding']
        preprocess_params = {k: v for k, v in pipeline_parameters.items() if k in preprocess_key_name}

        return preprocess_params, {}, {k: v for k, v in pipeline_parameters.items() if k in postprocess_params}

    def preprocess(self, inputs, **preprocess_params):
        """
        Preprocess of token classification

        Args:
            inputs (str):
                the str to be classified.
            max_length (Optional[int]):
                Max length of tokenizer's output. Default: 128.
            padding (Optional[bool, str]):
                Padding for max_length. Default: "max_length".

        Return:
            Processed text.
        """
        if not isinstance(inputs, str):
            raise ValueError("Inputs type must be str")

        self.input_text = inputs
        max_length = preprocess_params.pop("max_length", 128)
        padding = preprocess_params.pop("padding", "max_length")
        inputs = self.tokenizer(inputs, max_length=max_length, padding=padding,
                                return_tensors="ms", **preprocess_params)
        expand_dims = ops.ExpandDims()

        return {"input_ids": expand_dims(inputs["input_ids"], 0),
                "input_mask": expand_dims(inputs["attention_mask"], 0),
                "token_type_ids": expand_dims(inputs["token_type_ids"], 0)}

    def _forward(self, model_inputs, **forward_params):
        """
        Forward process

        Args:
            model_inputs (dict):
                outputs of preprocess.

        Return:
            Probs dict.
        """
        self.model.set_train(False)
        logits = self.network(**model_inputs)
        return {"logits": logits}

    def postprocess(self, model_outputs, **postprocess_params):
        """
        Postprocess

        Args:
            model_outputs (dict):
                outputs of forward process.

        Return:
            The generated results.
        """

        logits = model_outputs["logits"].asnumpy()
        maxes = np.max(logits, axis=-1, keepdims=True)
        shifted_exp = np.exp(logits - maxes)
        probs = shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)

        batch_pred_ids = np.argmax(probs, axis=2).tolist()
        batch_best_scores = np.max(probs, axis=2).tolist()

        # remove CLS and SEP token
        pred_paths = [[self.id2label[id_] for id_ in pred_ids[1:-1]] for pred_ids in batch_pred_ids]
        best_scores = [best_scores[1:-1] for best_scores in batch_best_scores]

        total_result = []
        for pred_path, best_score in zip(pred_paths, best_scores):
            single_result = []
            pred_entities = self.get_entities_bios(pred_path)
            for pred_entity in pred_entities:
                entity_result = {}
                entity_result["entity_group"] = pred_entity[0]
                entity_result["start"] = pred_entity[1]
                entity_result["end"] = pred_entity[2]
                entity_result["score"] = sum(best_score[entity_result["start"]:entity_result["end"] + 1]) / \
                    (entity_result["end"] + 1 - entity_result["start"])
                entity_result["score"] = round(entity_result["score"], 5)
                entity_result["word"] = self.input_text[entity_result["start"]:entity_result["end"] + 1]
                single_result.append(entity_result)
            total_result.append(single_result)
        return total_result

    def get_entities_bios(self, seq):
        """Gets entities from sequence.

        Args:
            seq (list):
                sequence of labels.

        Returns:
            List of (chunk_type, chunk_start, chunk_end).
        """
        chunks = []
        chunk = [-1, -1, -1]
        for indx, tag in enumerate(seq):
            if tag.startswith("S-"):
                if chunk[2] != -1:
                    chunks.append(chunk)
                chunk = [-1, -1, -1]
                chunk[1] = indx
                chunk[2] = indx
                chunk[0] = tag.split('-')[1]
                chunks.append(chunk)
                chunk = [-1, -1, -1]
            if tag.startswith("B-"):
                if chunk[2] != -1:
                    chunks.append(chunk)
                chunk = [-1, -1, -1]
                chunk[1] = indx
                chunk[0] = tag.split('-')[1]
            elif tag.startswith('I-') and chunk[1] != -1:
                entity_type = tag.split('-')[1]
                if entity_type == chunk[0]:
                    chunk[2] = indx
                if indx == len(seq) - 1:
                    chunks.append(chunk)
            else:
                if chunk[2] != -1:
                    chunks.append(chunk)
                chunk = [-1, -1, -1]
        return chunks
