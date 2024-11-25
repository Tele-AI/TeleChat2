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

"""QuestionAnsweringPipeline"""
import collections
import math
import six

from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindformers.mindformer_book import MindFormerBook
from mindformers.models import BasicTokenizer
from ..dataset.dataloader.squad_dataloader import convert_examples_to_features, SquadExample
from .base_pipeline import Pipeline
from ..tools.register import MindFormerRegister, MindFormerModuleType

__all__ = ['QuestionAnsweringPipeline']


@MindFormerRegister.register(MindFormerModuleType.PIPELINE, alias="question_answering")
class QuestionAnsweringPipeline(Pipeline):
    r"""Pipeline for token classification

    Args:
        model (Union[PretrainedModel, Model]):
            The model used to perform task, the input should be a model instance inherited from PretrainedModel.
        tokenizer (Optional[BaseTokenizer]):
            A tokenizer (None or PreTrainedTokenizer) for text processing.

    Raises:
        TypeError:
            If input model and image_processor's types are not corrected.
        ValueError:
            If the input model is not in support list.

    Examples:
        >>> from mindformers.pipeline import QuestionAnsweringPipeline
        >>> from mindformers import AutoTokenizer, BertForQuestionAnswering, AutoConfig
        >>> input_data = ["My name is Wolfgang and I live in Berlin - Where do I live?"]
        >>> tokenizer = AutoTokenizer.from_pretrained('qa_bert_base_uncased_squad')
        >>> qa_squad_config = AutoConfig.from_pretrained('qa_bert_base_uncased_squad')
        >>> model = BertForQuestionAnswering(qa_squad_config)
        >>> qa_pipeline = QuestionAnsweringPipeline(task='question_answering',
        ...                                         model=model,
        ...                                         tokenizer=tokenizer)
        >>> results = qa_pipeline(input_data)
        >>> print(results)
            [{'text': 'Berlin', 'score': 0.9941, 'start': 34, 'end': 40}]
    """
    _support_list = MindFormerBook.get_pipeline_support_task_list()['question_answering'].keys()

    def __init__(self, model, tokenizer, doc_stride=128, max_question_len=64, max_seq_len=384, top_k=1,
                 n_best_size=20, max_answer_len=30, **kwargs):

        if tokenizer is None:
            raise ValueError(f"{self.__class__.__name__}"
                             " requires for a tokenizer.")

        self.features = None
        self.examples = None
        self.basic_tokenizer = BasicTokenizer(do_lower_case=True)

        if doc_stride > max_seq_len:
            raise ValueError(f"`doc_stride` ({doc_stride}) is larger than `max_seq_len` ({max_seq_len})")
        if top_k < 1:
            raise ValueError(f"top_k parameter should be >= 1 (got {top_k})")
        if max_answer_len < 1:
            raise ValueError(f"max_answer_len parameter should be >= 1 (got {max_answer_len}")

        kwargs["doc_stride"] = doc_stride
        kwargs["max_question_len"] = max_question_len
        kwargs["max_seq_len"] = max_seq_len
        kwargs["top_k"] = top_k
        kwargs["n_best_size"] = n_best_size
        kwargs["max_answer_len"] = max_answer_len

        super().__init__(model, tokenizer=tokenizer, **kwargs)

    def _sanitize_parameters(self, **pipeline_parameters):
        """sanitize parameters for preprocess, forward, and postprocess."""
        if 'batch_size' in pipeline_parameters:
            raise ValueError(f"The {self.__class__.__name__} does not support batch inference, please remove the "
                             f"batch_size")

        preprocess_key_name = ['doc_stride', 'max_question_len', 'max_seq_len']
        postprocess_key_name = ['top_k', 'n_best_size', 'max_answer_len']

        preprocess_params = {k: v for k, v in pipeline_parameters.items() if k in preprocess_key_name}
        postprocess_params = {k: v for k, v in pipeline_parameters.items() if k in postprocess_key_name}

        return preprocess_params, {}, postprocess_params

    def preprocess(self, inputs, **preprocess_params):
        """
        Preprocess of token classification

        Args:
            inputs (str):
                The str to be classified.
            max_length (int):
                Max length of tokenizer's output
            return_tensors ("ms"):
                The type of returned tensors

        Return:
            processed text.
        """

        if not isinstance(inputs, str):
            raise ValueError("The type of inputs should be str.")

        if '-' not in inputs:
            raise ValueError("The inputs should contain context and question separeated by '-' syntax.")


        context_text, question_text = inputs.split("-")
        context_text = context_text.strip()
        question_text = question_text.strip()

        squad_example = SquadExample(None, question_text, context_text, None, None, None)
        features = convert_examples_to_features(
            examples=[squad_example],
            tokenizer=self.tokenizer,
            max_seq_len=preprocess_params.get('max_seq_len'),
            max_question_len=preprocess_params.get('max_question_len'),
            doc_stride=preprocess_params.get('doc_stride'),
            is_training=False
        )

        self.features = features
        self.examples = [squad_example]

        return features

    def _forward(self, model_inputs, **forward_params):
        """
        Forward process

        Args:
            model_inputs (dict):
                Outputs of preprocess.

        Return:
            probs dict.
        """
        self.model.set_train(False)

        model_outputs = []
        for feature in model_inputs:
            model_inputs = {"input_ids": Tensor(feature.input_ids, mstype.int32).expand_dims(0),
                            "input_mask": Tensor(feature.input_mask, mstype.int32).expand_dims(0),
                            "token_type_id": Tensor(feature.token_type_id, mstype.int32).expand_dims(0),
                            "start_position": Tensor(feature.start_position, mstype.int32).expand_dims(0),
                            "end_position": Tensor(feature.end_position, mstype.int32).expand_dims(0),
                            "unique_id": Tensor(feature.unique_id, mstype.int32).expand_dims(0)}

            ids, start, end = self.network(**model_inputs)

            RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])
            unique_id = int(ids[0])
            start_logits = [float(x) for x in start[0]]
            end_logits = [float(x) for x in end[0]]
            model_outputs.append(RawResult(unique_id=unique_id, start_logits=start_logits,
                                           end_logits=end_logits))

        return model_outputs

    def postprocess(self, model_outputs, **postprocess_params):
        """
        Postprocess

        Args:
            model_outputs (dict):
                Outputs of forward process.

        Return:
            The generated results
        """
        top_k = postprocess_params.get('top_k')
        n_best_size = postprocess_params.get('n_best_size')
        max_answer_len = postprocess_params.get('max_answer_len')

        example_index_to_features = collections.defaultdict(list)
        for feature in self.features:
            example_index_to_features[feature.example_index].append(feature)

        unique_id_to_result = {}
        for result in model_outputs:
            unique_id_to_result[result.unique_id] = result

        for (example_index, example) in enumerate(self.examples):
            features = example_index_to_features[example_index]
            prelim_predictions = self._get_prelim_predictions(features, unique_id_to_result,
                                                              max_answer_len, n_best_size)
            nbest = self._get_nbest(prelim_predictions, features, example, n_best_size)

            total_scores = []
            best_non_null_entry = None
            for entry in nbest:
                total_scores.append(entry.start_logit + entry.end_logit)
                if not best_non_null_entry:
                    if entry.text:
                        best_non_null_entry = entry

            probs = self._compute_softmax(total_scores)

            results = []
            for (i, entry) in enumerate(nbest):
                output = {}
                output["text"] = entry.text
                output["score"] = round(probs[i], 4)
                output["start"] = entry.start
                output["end"] = entry.end
                results.append(output)

        return results[:top_k]

    def _get_prelim_predictions(self, features, unique_id_to_result, max_answer_len, n_best_size):
        """get prelim predictions"""
        _PrelimPrediction = collections.namedtuple(
            "PrelimPrediction",
            ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])
        prelim_predictions = []

        # keep track of the minimum score of null start+end of position 0
        for (feature_index, feature) in enumerate(features):
            if feature.unique_id not in unique_id_to_result:
                continue
            result = unique_id_to_result[feature.unique_id]
            start_indexes = self._get_best_indexes(result.start_logits, n_best_size)
            end_indexes = self._get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_len:
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

    def _get_nbest(self, prelim_predictions, features, example, n_best_size):
        """get nbest predictions"""
        _NbestPrediction = collections.namedtuple(
            "NbestPrediction", ["text", "start_logit", "end_logit", "start", "end"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = self._get_final_text(tok_text, orig_text)
                char_start_index, char_end_index = self._get_answer_index(example.context_text,
                                                                          orig_doc_start, orig_doc_end)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True
                char_start_index, char_end_index = -1, -1

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit,
                    start=char_start_index,
                    end=char_end_index))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        if not nbest:
            raise ValueError(f"nbest should not be empty.")
        return nbest

    def _get_answer_index(self, context_text, orig_doc_start, orig_doc_end):
        char_index = 0
        for i, word in enumerate(context_text.split(" ")):
            if i == orig_doc_start:
                char_start_index = char_index
            if i == orig_doc_end:
                char_end_index = char_index + len(word)
            char_index += len(word) + 1

        return char_start_index, char_end_index

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

    def _get_best_indexes(self, logits, n_best_size):
        """Get the n-best logits from a list."""
        index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

        best_indexes = []
        for (i, score) in enumerate(index_and_score):
            if i >= n_best_size:
                break
            best_indexes.append(score[0])
        return best_indexes
