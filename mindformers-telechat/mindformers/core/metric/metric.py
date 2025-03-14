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
"""MindFormer Self-Define Metric."""
import os
import sys
import re
import collections
import json
import math
import string
import shutil
import six
import jieba
import numpy as np
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import mindspore.nn as nn
import mindspore as ms
from mindspore.ops import operations as P
from mindspore.communication import get_group_size, get_rank
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.models import BasicTokenizer
from mindformers.core.loss import CrossEntropyLoss
from mindformers.utils import deprecated

from .utils import PerplexityCell
from ...dataset.labels import cluener_labels

__all__ = ['EntityScore', 'SQuADMetric', 'PerplexityMetric', 'ADGENMetric', 'PromptAccMetric', 'EmF1Metric']


@MindFormerRegister.register(MindFormerModuleType.METRIC)
class EntityScore(nn.Metric):
    r"""
    Evaluates the precision, recall, and F1 score of predicted entities against the ground truth.

    Mathematically, these metrics are defined as follows:

    Precision: Measures the fraction of correctly predicted entities out of all predicted entities.

    .. math::
        \text{Precision} = \frac{\text{Number of correct entities}}{\text{Number of predicted entities}}

    Recall: Measures the fraction of correctly predicted entities out of all actual entities.

    .. math::
        \text{Recall} = \frac{\text{Number of correct entities}}{\text{Number of actual entities}}

    F1 Score: The harmonic mean of precision and recall, providing a balance between them.

    .. math::
        \text{F1} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindformers.core.metric.metric import EntityScore
        >>> x = Tensor(np.array([[np.arange(0, 22)]]))
        >>> y = Tensor(np.array([[21]]))
        >>> metric = EntityScore()
        >>> metric.clear()
        >>> metric.update(x, y)
        >>> result = metric.eval()
        >>> print(result)
        ({'precision': 1.0, 'recall': 1.0, 'f1': 1.0}, {'address': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}})
    """
    def __init__(self):
        super(EntityScore, self).__init__()
        self.label2id = {label: label_id for label_id, label in enumerate(cluener_labels)}
        self.id2label = {label_id: label for label, label_id in self.label2id.items()}
        self.clear()

    def clear(self):
        """Clearing the internal evaluation result."""
        self.origins = []
        self.founds = []
        self.rights = []

    def update(self, *inputs):
        """
        Updating the internal evaluation result.

        Args:
            *inputs (List): Logits and labels. The logits are tensors of shape :math:`[N,C]` with data type Float16 or
                Float32, and the labels are tensors of shape :math:`[N,]` with data type Int32 or Int64, where :math:`N`
                is batch size, and :math:`C` is the total number of entity types.
        """
        batch_logits = inputs[0].asnumpy()
        batch_label_ids = inputs[1].asnumpy()
        batch_pred_ids = np.argmax(batch_logits, axis=2).tolist()

        pred_paths = [[self.id2label[id_] for id_ in pred_ids] for pred_ids in batch_pred_ids]
        label_paths = [[self.id2label[id_] for id_ in label_ids] for label_ids in batch_label_ids]

        for label_path, pre_path in zip(label_paths, pred_paths):
            label_entities = self.get_entities_bios(label_path)
            pred_entities = self.get_entities_bios(pre_path)
            self.origins.extend(label_entities)
            self.founds.extend(pred_entities)
            self.rights.extend([pred_entity for pred_entity in pred_entities if pred_entity in label_entities])

    def eval(self):
        """
        Computing the evaluation result.

        Returns:
            A dict of evaluation results with precision, recall, and F1 scores of entities relative to their true
            labels.
        """
        class_info = {}
        origin_counter = collections.Counter([x[0] for x in self.origins])
        found_counter = collections.Counter([x[0] for x in self.founds])
        right_counter = collections.Counter([x[0] for x in self.rights])
        for type_, count in origin_counter.items():
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            recall, precision, f1 = self.compute(origin, found, right)
            class_info[type_] = {"precision": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)}
        origin = len(self.origins)
        found = len(self.founds)
        right = len(self.rights)
        recall, precision, f1 = self.compute(origin, found, right)
        return {"precision": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)}, class_info

    def compute(self, origin, found, right):
        """Compute f1, precision and recall."""
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def get_entities_bios(self, seq):
        """Get entities from sequence."""
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


@deprecated(version="1.5.0")
@MindFormerRegister.register(MindFormerModuleType.METRIC)
class SQuADMetric(nn.Metric):
    r"""
    The SQuAD Metric primarily employs two key evaluation metrics: Exact Match (EM) and F1 Score.

    Exact Match (EM): Measures the percentage of predictions that match any one of the ground truth
    answers exactly. It is defined as:

    .. math::
        EM = \frac{\text{Number of exact matches}}{\text{Total number of questions}} \times 100

    F1 Score: Measures the average overlap between the predicted answer and the ground truth answer. It treats
    both the prediction and the ground truth as bags of tokens and computes their F1 score as:

    .. math::
        F1 = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}

    Where :math:`\text{Precision} = \frac{\text{Number of correct tokens in prediction}}
    {\text{Number of tokens in prediction}}` , :math:`\text{Recall} = \frac{\text{Number of correct
    tokens in prediction}}{\text{Number of tokens in ground truth answer}}` .
    """
    def __init__(self, dataset_dir, n_best_size=20, max_answer_len=30, do_lower_case=True,
                 temp_file_dir="./squad_temp"):
        self.outputs = []
        self.temp_file_dir = temp_file_dir
        temp_examples_file = os.path.join(temp_file_dir, "temp_examples.json")
        temp_features_file = os.path.join(temp_file_dir, "temp_features.json")
        self.all_examples = self._load_temp_data(temp_examples_file)
        self.all_features = self._load_temp_data(temp_features_file)
        self.dev_file_path = os.path.join(dataset_dir, "dev-v1.1.json")
        self.basic_tokenizer = BasicTokenizer(do_lower_case)
        self.n_best_size = n_best_size
        self.max_answer_len = max_answer_len

    def clear(self):
        """Clearing the internal evaluation result."""
        return

    def update(self, *inputs):
        """Update results for every batch"""
        ids = inputs[0].asnumpy()
        start = inputs[1].asnumpy()
        end = inputs[2].asnumpy()

        batch_size = len(ids)

        RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])

        for i in range(batch_size):
            unique_id = int(ids[i])
            start_logits = [float(x) for x in start[i].flat]
            end_logits = [float(x) for x in end[i].flat]
            self.outputs.append(RawResult(unique_id=unique_id, start_logits=start_logits,
                                          end_logits=end_logits))

    def eval(self):
        """Compute final result"""
        predictions = self._get_predictions()

        with open(self.dev_file_path) as ds:
            dataset_json = json.load(ds)
            dataset = dataset_json['data']

        f1 = exact_match = total = 0
        for article in dataset:
            for paragraph in article['paragraphs']:
                for qa in paragraph['qas']:
                    total += 1
                    if qa['id'] not in predictions:
                        message = 'Unanswered question ' + qa['id'] + \
                                  ' will receive score 0.'
                        print(message, file=sys.stderr)
                        continue
                    ground_truths = list(map(lambda x: x['text'], qa['answers']))
                    if not ground_truths:
                        continue
                    prediction = predictions[qa['id']]
                    exact_match += self._metric_max_over_ground_truths(
                        self._exact_match_score, prediction, ground_truths)
                    f1 += self._metric_max_over_ground_truths(
                        self._f1_score, prediction, ground_truths)

        exact_match = 100.0 * exact_match / total
        self._remove_temp_data()
        return {'exact_match': exact_match, 'f1': 100.0 * f1 / total}

    def _remove_temp_data(self):
        shutil.rmtree(self.temp_file_dir)

    def _load_temp_data(self, temp_file_path):
        with open(temp_file_path, "r", encoding="utf-8") as f:
            data = []
            for line in f.readlines():
                data.append(json.loads(line.strip()))
        return data

    def _normalize_answer(self, s):
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def _f1_score(self, prediction, ground_truth):
        """calculate f1 score"""
        prediction_tokens = self._normalize_answer(prediction).split()
        ground_truth_tokens = self._normalize_answer(ground_truth).split()
        common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def _exact_match_score(self, prediction, ground_truth):
        return self._normalize_answer(prediction) == self._normalize_answer(ground_truth)

    def _metric_max_over_ground_truths(self, metric_fn, prediction, ground_truths):
        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)

    def _get_predictions(self):
        """Get final predictions"""
        example_index_to_features = collections.defaultdict(list)
        for feature in self.all_features:
            example_index_to_features[feature["example_index"]].append(feature)

        unique_id_to_result = {}
        for result in self.outputs:
            unique_id_to_result[result.unique_id] = result
        all_predictions = collections.OrderedDict()

        for (example_index, example) in enumerate(self.all_examples):
            features = example_index_to_features[example_index]
            prelim_predictions = self._get_prelim_predictions(features, unique_id_to_result)
            nbest = self._get_nbest(prelim_predictions, features, example)

            total_scores = []
            best_non_null_entry = None
            for entry in nbest:
                total_scores.append(entry.start_logit + entry.end_logit)
                if not best_non_null_entry:
                    if entry.text:
                        best_non_null_entry = entry

            probs = self._compute_softmax(total_scores)

            nbest_json = []
            for (i, entry) in enumerate(nbest):
                output = collections.OrderedDict()
                output["text"] = entry.text
                output["probability"] = probs[i]
                output["start_logit"] = entry.start_logit
                output["end_logit"] = entry.end_logit
                nbest_json.append(output)

            if not nbest_json:
                raise ValueError(f"len(nbest_json) should not less than 1, but got {len(nbest_json)}.")

            all_predictions[example["qas_id"]] = nbest_json[0]["text"]
        return all_predictions

    def _get_prelim_predictions(self, features, unique_id_to_result):
        """get prelim predictions"""
        _PrelimPrediction = collections.namedtuple(
            "PrelimPrediction",
            ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])
        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        for (feature_index, feature) in enumerate(features):
            if feature["unique_id"] not in unique_id_to_result:
                continue
            result = unique_id_to_result[feature["unique_id"]]
            start_indexes = self._get_best_indexes(result.start_logits)
            end_indexes = self._get_best_indexes(result.end_logits)
            # if we could have irrelevant answers, get the min score of irrelevant
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature["tokens"]):
                        continue
                    if end_index >= len(feature["tokens"]):
                        continue
                    if str(start_index) not in feature["token_to_orig_map"]:
                        continue
                    if str(end_index) not in feature["token_to_orig_map"]:
                        continue
                    if not feature["token_is_max_context"].get(str(start_index), False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > self.max_answer_len:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        return prelim_predictions

    def _get_nbest(self, prelim_predictions, features, example):
        """get nbest predictions"""
        _NbestPrediction = collections.namedtuple(
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= self.n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature["tokens"][pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature["token_to_orig_map"][str(pred.start_index)]
                orig_doc_end = feature["token_to_orig_map"][str(pred.end_index)]
                orig_tokens = example["doc_tokens"][orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)
                final_text = self._get_final_text(tok_text, orig_text)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        if not nbest:
            raise ValueError(f"nbest should not be empty.")
        return nbest

    def _compute_softmax(self, scores):
        """Compute softmax probability over raw logits."""
        if not scores:
            return []

        max_score = None
        for score in scores:
            if max_score is None or score > max_score:
                max_score = score

        exp_scores = []
        total_sum = 0.0
        for score in scores:
            x = math.exp(score - max_score)
            exp_scores.append(x)
            total_sum += x

        probs = []
        for score in exp_scores:
            probs.append(score / total_sum)
        return probs

    def _get_final_text(self, pred_text, orig_text):
        """Project the tokenized prediction back to the original text."""

        def _strip_spaces(text):
            ns_chars = []
            ns_to_s_map = collections.OrderedDict()
            for (i, c) in enumerate(text):
                if c == " ":
                    continue
                ns_to_s_map[len(ns_chars)] = i
                ns_chars.append(c)
            ns_text = "".join(ns_chars)
            return (ns_text, ns_to_s_map)

        tok_text = " ".join(self.basic_tokenizer.tokenize(orig_text))

        start_position = tok_text.find(pred_text)
        if start_position == -1:
            return orig_text
        end_position = start_position + len(pred_text) - 1

        (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
        (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

        if len(orig_ns_text) != len(tok_ns_text):
            return orig_text

        tok_s_to_ns_map = {}
        for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
            tok_s_to_ns_map[tok_index] = i

        orig_start_position = None
        if start_position in tok_s_to_ns_map:
            ns_start_position = tok_s_to_ns_map[start_position]
            if ns_start_position in orig_ns_to_s_map:
                orig_start_position = orig_ns_to_s_map[ns_start_position]

        if orig_start_position is None:
            return orig_text

        orig_end_position = None
        if end_position in tok_s_to_ns_map:
            ns_end_position = tok_s_to_ns_map[end_position]
            if ns_end_position in orig_ns_to_s_map:
                orig_end_position = orig_ns_to_s_map[ns_end_position]

        if orig_end_position is None:
            return orig_text

        output_text = orig_text[orig_start_position:(orig_end_position + 1)]
        return output_text

    def _get_best_indexes(self, logits):
        """Get the n-best logits from a list."""
        index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

        best_indexes = []
        for (i, score) in enumerate(index_and_score):
            if i >= self.n_best_size:
                break
            best_indexes.append(score[0])
        return best_indexes


@MindFormerRegister.register(MindFormerModuleType.METRIC)
class PerplexityMetric(nn.Metric):
    r"""
    Perplexity is defined as the exponentiated average negative log-probability assigned by the model to each word
    in the test set. Mathematically, for a sequence of words :math:`W = (w_1, w_2, \ldots, w_N)` , the
    perplexity (PP) is given by:

    .. math::
        PP(W) = P(w_1, w_2, \ldots, w_N)^{-\frac{1}{N}} = \sqrt[N]{\frac{1}{P(w_1, w_2, \ldots, w_N)}}

    Where :math:`P(w_1, w_2, \ldots, w_N)` is the probability of the sequence under the model.

    In practical terms, perplexity can be rewritten as:

    .. math::
        PP(W) = \exp\left(-\frac{1}{N} \sum_{i=1}^{N} \log P(w_i | w_1, w_2, \ldots, w_{i-1})\right)

    This equation highlights that a lower perplexity indicates a better-performing language model, as it suggests
    that the model assigns higher probabilities to the actual sequence of words.

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindformers.core.metric.metric import PerplexityMetric
        >>> x = Tensor(np.array([[[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]]))
        >>> y = Tensor(np.array([[1, 0, 1]]))
        >>> mask = Tensor(np.array([[1, 1, 1]]))
        >>> metric = PerplexityMetric()
        >>> metric.clear()
        >>> metric.update(x, y, mask)
        >>> perplexity = metric.eval()
        >>> print(perplexity)
        'loss': 0.8262470960617065, 'PPL': 2.284728265028813}
    """
    def __init__(self):
        super(PerplexityMetric, self).__init__()
        self.num_data = None
        self.total_loss = None
        self.loss = CrossEntropyLoss()
        self.pipeline_stages = ms.get_auto_parallel_context('pipeline_stages')
        self.pipeline_parallel = self.pipeline_stages > 1
        self.rank_id = 0
        self.device_num = 1
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.not_equal = P.NotEqual()
        self.sub = P.Sub()
        self.ppl_loss = PerplexityCell(self.pipeline_parallel)

        if self.pipeline_parallel:
            self.rank_id = get_rank()
            self.device_num = get_group_size()

        per_stage_device_num = self.device_num // self.pipeline_stages
        stage_id = self.rank_id // per_stage_device_num
        self.is_last_stage = (stage_id == self.pipeline_stages - 1)

        self.parallel_mode = ms.get_auto_parallel_context("parallel_mode")
        self.full_batch = ms.get_auto_parallel_context("full_batch")
        self.auto_parallel = self.parallel_mode in ['semi_auto_parallel', 'auto_parallel']

    def clear(self):
        """Clearing the internal evaluation result."""
        self.num_data = 0
        self.total_loss = 0.0

    def update(self, *inputs):
        """
        Updating the internal evaluation result.

        Args:
            *inputs (List): Logits, labels, and input_mask. Logits is a tensor of shape :math:`[N,S,W]`
                with data type Float16 or Float32, Labels and input_mask is a tensor of shape :math:`[N,S]` with
                data type Int32 or Int64. where :math:`N` is the batch size, :math:`S` is the sequence length,
                and :math:`W` is the vocabulary size.
        """
        if self.pipeline_parallel:
            if not self.is_last_stage:
                return
            if self.auto_parallel:
                ms.context.set_auto_parallel_context(parallel_mode='data_parallel', full_batch=False)
            loss = self.ppl_loss(*inputs)
            loss = float(loss.asnumpy())
            self.total_loss += loss
            self.num_data += 1
            if self.auto_parallel:
                ms.set_auto_parallel_context(parallel_mode=self.parallel_mode,
                                             full_batch=True,
                                             pipeline_stages=self.pipeline_stages)
        else:
            loss = self.ppl_loss(*inputs)
            loss = float(loss.asnumpy())
            self.total_loss += loss
            self.num_data += 1

    def eval(self):
        """
        Computing the evaluation result.

        Returns:
            A dict of evaluation results with loss and PPL scores.
        """
        if self.pipeline_parallel and not self.is_last_stage:
            return None
        avg_loss = float(self.total_loss / self.num_data)
        result = {"loss": avg_loss, "PPL": math.exp(avg_loss)}
        if self.pipeline_parallel:
            print("Average Loss and PPL Metric:", result)
        return result


@MindFormerRegister.register(MindFormerModuleType.METRIC)
class ADGENMetric(nn.Metric):
    """Compute the f1, precision and recall score of each entity"""

    def __init__(self):
        super(ADGENMetric, self).__init__()
        self.score_dict = {
            "rouge-1": [],
            "rouge-2": [],
            "rouge-l": [],
            "bleu-4": []
        }

    def clear(self):
        self.score_dict = {
            "rouge-1": [],
            "rouge-2": [],
            "rouge-l": [],
            "bleu-4": []
        }

    def update(self, *inputs):
        """Update results for every batch"""
        preds = inputs[0]  # list[numpy]
        labels = inputs[1]  # numpy

        if isinstance(preds, tuple):
            preds = preds[0]

        for pred, label in zip(preds, labels):
            print(f"pred is:\n {pred}\n",
                  f"label is:\n {label}")
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))
            hypothesis_str = ' '.join(hypothesis)
            reference_str = ' '.join(reference)
            rouge = Rouge()
            if hypothesis_str.strip() == "":
                continue
            scores = rouge.get_scores(hypothesis_str, reference_str)
            result = scores[0]

            for k, v in result.items():
                self.score_dict.get(k).append(round(v["f"] * 100, 4))
            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            self.score_dict["bleu-4"].append(round(bleu_score * 100, 4))

    def eval(self):
        """Compute final result"""
        for k, v in self.score_dict.items():
            self.score_dict[k] = float(np.mean(v))
        print('metric: ADGENMetric\n' +
              f'rouge-1: {self.score_dict["rouge-1"]:.4f}\n' +
              f'rouge-2: {self.score_dict["rouge-2"]:.4f}\n' +
              f'rouge-l: {self.score_dict["rouge-l"]:.4f}\n' +
              f'bleu-4:  {self.score_dict["bleu-4"]:.4f}')
        return self.score_dict


@MindFormerRegister.register(MindFormerModuleType.METRIC)
class PromptAccMetric(nn.Metric):
    r"""
    Computes the prompt acc of each entity. The prompt acc is the accuracy of text classification base on building
    prompt. The accurate index is the index of the prompt which has the minimum perplexity.

    1. Build the prompt for this metric is described as follows:

       .. code-block::

           这是关于**体育**的文章：$passage
           这是关于**文化**的文章：$passage

    2. Computes perplexity of each generated context based on prompt.
       Perplexity is a measurement about how well a probability distribution or a model predicts a sample.
       A low perplexity indicates the model can predict the sample well. The function is shown as follows:

       .. math::
           PP(W)=P(w_{1}w_{2}...w_{N})^{-\frac{1}{N}}=\sqrt[N]{\frac{1}{P(w_{1}w_{2}...w_{N})}}

       Where :math:`w` represents words in corpus.

    3. Compute classification result by choosing the index of the prompt which has the minimum perplexity.

    4. Count the number of correctly classified and the total number of samples and compute the acc as follows:

       .. math::
           \text{accuracy} =\frac{\text{correct_sample_nums}}{\text{total_sample_nums}}

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindformers.core.metric.metric import PromptAccMetric
        >>> logtis = Tensor(np.array([[[[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]]]))
        >>> input_ids = Tensor(np.array([[15, 16, 17]]))
        >>> labels = Tensor(np.array([[1, 0, 1]]))
        >>> mask = Tensor(np.array([[1, 1, 1]]))
        >>> metric = PromptAccMetric()
        >>> metric.clear()
        >>> metric.update(logtis, input_ids, mask, labels)
        >>> result = metric.eval()
        >>> print(result)
        Current data num is 1, total acc num is 1.0, ACC is 1.000
        Acc: 1.000, total_acc_num: 1.0, total_num: 1
        {'Acc': 1.0}
    """

    def __init__(self):
        super(PromptAccMetric, self).__init__()
        self.num_data = None
        self.total_acc_num = None
        self.loss = CrossEntropyLoss()
        self.pipeline_stages = ms.get_auto_parallel_context('pipeline_stages')
        self.pipeline_parallel = self.pipeline_stages > 1
        self.last_card_id = 0
        self.rank_id = 0
        self.device_num = 1
        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.equal = P.Equal()
        self.softmax = P.Softmax()
        self.argmin = P.Argmin()
        self.sum = P.ReduceSum()

        if self.pipeline_parallel:
            self.rank_id = get_rank()
            self.device_num = get_group_size()

        per_stage_device_num = self.device_num // self.pipeline_stages
        stage_id = self.rank_id // per_stage_device_num
        self.is_last_stage = (stage_id == self.pipeline_stages - 1)

        self.parallel_mode = ms.get_auto_parallel_context("parallel_mode")
        self.full_batch = ms.get_auto_parallel_context("full_batch")
        self.auto_parallel = self.parallel_mode in ['semi_auto_parallel', 'auto_parallel']

    def clear(self):
        """Clearing the internal evaluation result."""
        self.num_data = 0
        self.total_acc_num = 0

    def calculate_circle(self, *inputs):
        """The main calculate logic."""
        logits, input_ids, input_mask, labels = inputs[0], inputs[1], inputs[2], inputs[3]
        batch_size, num_labels, seq_length, _ = logits.shape
        logits = self.reshape(logits, (batch_size * num_labels, seq_length, -1))
        ppl_list = []
        for index in range(batch_size * num_labels):
            sub_logits, sub_tokens, sub_mask_list = logits[index], input_ids[index], input_mask[index]

            sub_logits = sub_logits[:-1, ::]
            sub_tokens = sub_tokens[1:]
            sub_mask_list = sub_mask_list[1:]

            loss = self.loss(sub_logits, sub_tokens, sub_mask_list)
            loss = float(loss.asnumpy())
            ppl_list.append(loss)  # smaller, better
        ppl_ms = ms.Tensor(ppl_list, dtype=ms.float32)
        ppl_ms = self.reshape(ppl_ms, (batch_size, num_labels))
        ppl_ms = self.cast(self.argmin(ppl_ms), ms.int32)
        label = self.reshape(labels, (-1,))
        cur_acc_num = self.sum(self.cast(self.equal(ppl_ms, label), ms.float16)).asnumpy()
        self.num_data += batch_size
        self.total_acc_num += cur_acc_num

    def update(self, *inputs):
        """
        Updating the internal evaluation result.

        Args:
            *inputs (List): Logits, input_ids, input_mask, and labels.
                where logits is a tensor of shape :math:`[N,C,S,W]` with data type Float16 or Float32,
                and input_ids, input_mask, and labels are tensors of shape :math:`[N*C,S]` with data type
                Int32 or Int64. Where :math:`N` is batch size, :math:`C` the total number of entity types,
                :math:`S` is the sequence length, and :math:`W` is the vocabulary size.
        """
        if self.pipeline_parallel:
            if not self.is_last_stage:
                return
            if self.auto_parallel:
                ms.context.set_auto_parallel_context(parallel_mode='data_parallel', full_batch=False)

            self.calculate_circle(*inputs)

            if self.auto_parallel:
                ms.set_auto_parallel_context(parallel_mode=self.parallel_mode,
                                             full_batch=True,
                                             pipeline_stages=self.pipeline_stages)
        else:
            self.calculate_circle(*inputs)
        print("Current data num is {}, total acc num is {}, ACC is {}".format(
            self.num_data, self.total_acc_num, "%.3f" % (self.total_acc_num / self.num_data)))
        return

    def eval(self):
        """
        Computing the evaluation result.

        Returns:
            A dict of evaluation results with Acc scores.
        """
        if self.pipeline_parallel and not self.is_last_stage:
            return None
        acc_rate = float(self.total_acc_num / self.num_data)
        result = {"Acc": acc_rate}
        print(f"Acc: {('%.3f' % result.get('Acc', 0))}, total_acc_num: {self.total_acc_num}, "
              f"total_num: {self.num_data}")
        return result


@MindFormerRegister.register(MindFormerModuleType.METRIC)
class EmF1Metric(nn.Metric):
    r"""
    Calculate the Em and F1 scores for each example to evaluate the model's performance in prediction tasks.

    Em Score: The Em score measures the accuracy of predictions that exactly match the labels,
    ignoring punctuation. For example, if the question is "河南的省会是哪里？" and the label is "郑州市":

    When the prediction is "郑州市", the Em score is 100.
    When the prediction is "郑州市。", the Em score is 100.
    When the prediction is "郑州", the Em score is 0.

    F1 Score: The F1 score is the harmonic mean of precision and recall, calculated as follows:

    .. math::
        F1 = \frac{2 \times \text{precision} \times \text{recall}}{\text{precision} + \text{recall}}

    Where precision and recall are calculated as:

    .. math::
        \text{precision} = \frac{\text{lcs_length}}{\text{len(prediction_segment)}},
        \quad \text{recall} = \frac{\text{lcs_length}}{\text{len(label_segment)}}

    In the above formulas, :math:`\text{lcs_length}` represents the length of the longest common subsequence (LCS).

    Calculation Process:

    First, calculate the longest common subsequence (LCS) between the prediction and the label to measure
    the degree of matching.
    Then, compute the precision and recall based on the respective formulas.
    Finally, use the F1 score formula to calculate the final F1 value.
    This evaluation metric comprehensively measures the accuracy and completeness of the model, providing
    data support for model optimization and debugging.

    Examples:
        >>> from mindformers.core.metric.metric import EmF1Metric
        >>>
        >>> str_pre = ["I love Beijing, because it's beautiful", "Hello world。"]
        >>> str_label = ["I love Beijing.", "Hello world"]
        >>> metric = EmF1Metric()
        >>> metric.clear()
        >>> for pre, label in zip(str_pre, str_label):
        ...    metric.update([pre], [label])
        >>> result = metric.eval()
        >>> print(result)
        The F1/Em of this example is:  {'F1': 100.0, 'Em': 100.0}
        F1 score: 75.0, Em score: 50.0, total_count: 2
        {'F1': 75.0, 'Em': 50.0}
    """
    def __init__(self):
        super(EmF1Metric, self).__init__()
        self.gens = None
        self.labels = None
        self.metrics = None
        self.num_data = None

    def clear(self):
        """Clearing the internal evaluation result."""
        self.gens = []
        self.labels = []
        self.metrics = {
            'Em': 0.0,
            'F1': 0.0
        }
        self.num_data = 0

    def update(self, *inputs):
        """
        Updating the internal evaluation result.

        Args:
            *inputs (List): Predictions and labels. Both are lists containing :math:`N` strings.
                Where :math:`N` is the batch size.
        """
        gen, label = inputs[0], inputs[1]
        for i, _ in enumerate(gen):
            gen[i] = gen[i].strip()
            gen[i] = gen[i].split("\n")[0]
        print(f"pred is:\n {gen}\n",
              f"label is:\n {label}")

        self.gens.extend(gen)
        self.labels.extend(label)
        self.num_data += len(gen)

        result, current_count = self.evaluate_pairs(gen, label)
        print("The F1/Em of this example is: ", result)
        if self.num_data % 10 == 0:
            result, current_count = self.evaluate_pairs(self.gens, self.labels)
            print(f"F1 score: {result.get('F1', 0)}, Em score: {result.get('Em', 0)}, current_count: {current_count}")

    def eval(self):
        """
        Computing the evaluation result.

        Returns:
            A dict of evaluation results with Em and F1 scores.
        """
        result, total_count = self.evaluate_pairs(self.gens, self.labels)
        print(f"F1 score: {result.get('F1', 0)}, Em score: {result.get('Em', 0)}, total_count: {total_count}")
        return result

    def mixed_segmentation(self, in_str, rm_punc=False):
        """cut input for calculating lcs"""
        in_str = str(in_str).lower().strip()
        segs_out = []
        temp_str = ""
        sp_char = ['-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=',
                   '，', '。', '：', '？', '！', '“', '”', '；', '’', '《', '》', '……', '·', '、',
                   '「', '」', '（', '）', '－', '～', '『', '』']
        for char in in_str:
            if rm_punc and char in sp_char:
                continue
            if re.search(r'[\u4e00-\u9fa5]', char) or char in sp_char:
                if temp_str != "":
                    ss = list(jieba.cut(temp_str))
                    segs_out.extend(ss)
                    temp_str = ""
                segs_out.append(char)
            else:
                temp_str += char

        if temp_str != "":
            ss = list(jieba.cut(temp_str))
            segs_out.extend(ss)

        return segs_out

    def remove_punctuation(self, in_str):
        """remove punctuations in inputs"""
        in_str = str(in_str).lower().strip()
        sp_char = ['-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=',
                   '，', '。', '：', '？', '！', '“', '”', '；', '’', '《', '》', '……', '·', '、',
                   '「', '」', '（', '）', '－', '～', '『', '』']
        out_segs = []
        for char in in_str:
            if char in sp_char:
                continue
            else:
                out_segs.append(char)
        return ''.join(out_segs)

    def find_lcs(self, s1, s2):
        """calculate the length of lcs"""
        m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]
        mmax = 0
        p = 0
        for i, s1_i in enumerate(s1):
            for j, s2_j in enumerate(s2):
                if s1_i == s2_j:
                    m[i + 1][j + 1] = m[i][j] + 1
                    if m[i + 1][j + 1] > mmax:
                        mmax = m[i + 1][j + 1]
                        p = i + 1
        return s1[p - mmax:p], mmax

    def calc_f1_score(self, answers, prediction):
        """calculate f1 score"""
        f1_scores = []
        for ans in answers:
            ans_segs = self.mixed_segmentation(ans, rm_punc=True)
            prediction_segs = self.mixed_segmentation(prediction, rm_punc=True)
            _, lcs_len = self.find_lcs(ans_segs, prediction_segs)
            if lcs_len == 0:
                f1_scores.append(0)
                continue
            precision = 1.0 * lcs_len / len(prediction_segs)
            recall = 1.0 * lcs_len / len(ans_segs)
            f1 = (2 * precision * recall) / (precision + recall)
            f1_scores.append(f1)
        return max(f1_scores)

    def calc_em_score(self, answers, prediction):
        """calculate em score"""
        em = 0
        for ans in answers:
            ans_ = self.remove_punctuation(ans)
            prediction_ = self.remove_punctuation(prediction)
            if ans_ == prediction_:
                em = 1
                break
        return em

    def evaluate_pairs(self, pred_, ans_):
        """calculate metric"""
        f1 = 0
        em = 0
        total_count = 0
        for (prediction, answer) in zip(pred_, ans_):
            total_count += 1
            f1 += self.calc_f1_score([answer], prediction)
            em += self.calc_em_score([answer], prediction)
        if total_count > 0:
            f1_score = 100.0 * f1 / total_count
            em_score = 100.0 * em / total_count
            result = {'F1': f1_score, 'Em': em_score}
        else:
            print("total_count is zero")
            result = {}
        return result, total_count
