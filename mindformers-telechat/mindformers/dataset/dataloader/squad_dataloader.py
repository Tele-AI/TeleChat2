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
"""SQuAD DataLoader"""
import os
import json
import collections

from mindspore.dataset import GeneratorDataset

from mindformers.tools.register import MindFormerRegister, MindFormerModuleType


@MindFormerRegister.register(MindFormerModuleType.DATASET_LOADER)
class SQuADDataLoader:
    """SQuAD Dataloader"""
    _default_column_names = ["input_ids", "input_mask", "token_type_id",
                             "start_positions", "end_positions", "unique_id"]

    def __new__(cls, dataset_dir, tokenizer, column_names=None, stage="train",
                max_question_len=64, max_seq_len=384, doc_stride=128, **kwargs):
        r"""
        SQuAD Dataloader API.

        Args:
            dataset_dir: The directory to SQuAD dataset.
            tokenizer: a tokenizer for text processing.
            column_names (Optional[Union[List[str], Tuple[str]]]): The output column names,
                                                                   a tuple or a list of string with length 6
            stage: The supported keywords are in ["train", "dev"]
            max_question_len: The maximum number of tokens for the question,
                              Questions longer than this will be truncated to this length.
            max_seq_len: Maximum sequence length.
            doc_stride: When splitting up a long document into chunks, how much stride to take between chunks.

        Return:
            A GeneratorDataset for SQuAD dataset

        Raises:
            ValueError: Error input for dataset_dir, and column_names.
            TypeError: Type error for column_names.

        Examples:
            >>> from mindformers import SQuADDataLoader
            >>> from mindformers.models import BertTokenizer
            >>> bert_tokenizer = BertTokenizer.from_pretrained('qa_bert_base_uncased')
            >>> data_loader = SQuADDataLoader("./squad/", bert_tokenizer)
            >>> data_loader = data_loader.batch(1)
            >>> for item in data_loader:
            >>>     print(item)
            >>>     break
                [Tensor(shape=[24, 384], dtype=Int32, value=
                [[ 101, 2054, 3589 ...    0,    0,    0],
                [ 101, 2054, 3589 ...    0,    0,    0],
                [ 101, 2054, 3589 ...    0,    0,    0],
                ...
                [ 101, 2054, 3589 ...    0,    0,    0],
                [ 101, 2054, 3589 ...    0,    0,    0],
                [ 101, 2054, 3589 ...    0,    0,    0]]),
                Tensor(shape=[24, 384], dtype=Int32, value=
                [[1, 1, 1 ... 0, 0, 0],
                [1, 1, 1 ... 0, 0, 0],
                [1, 1, 1 ... 0, 0, 0],
                ...
                [1, 1, 1 ... 0, 0, 0],
                [1, 1, 1 ... 0, 0, 0],
                [1, 1, 1 ... 0, 0, 0]]),
                Tensor(shape=[24, 384], dtype=Int32, value=
                [[0, 0, 0 ... 0, 0, 0],
                [0, 0, 0 ... 0, 0, 0],
                [0, 0, 0 ... 0, 0, 0],
                ...
                [0, 0, 0 ... 0, 0, 0],
                [0, 0, 0 ... 0, 0, 0],
                [0, 0, 0 ... 0, 0, 0]]),
                Tensor(shape=[24], dtype=Int32, value= [24, 24, 24, 24,
                24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24
                ]), Tensor(shape=[24], dtype=Int32, value= [24, 24, 24,
                 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24
                ]), Tensor(shape=[24], dtype=Int32, value= [1000000029, 1000000020, 1000000023, 1000000012,
                1000000024, 1000000004, 1000000006, 1000000003, 1000000017, 1000000022, 1000000028, 1000000007,
                1000000005, 1000000027, 1000000014, 1000000015, 1000000002, 1000000025, 1000000011, 1000000008,
                1000000021, 1000000010, 1000000019, 1000000016])]
        """
        dataset_dir = os.path.realpath(dataset_dir)
        if not os.path.isdir(dataset_dir):
            raise ValueError(f"{dataset_dir} is not existed.")

        if stage not in ["train", "dev"]:
            raise ValueError(f"stage should be in train or dev.")

        if column_names is None:
            column_names = cls._default_column_names

        if not isinstance(column_names, (tuple, list)):
            raise TypeError(f"column_names should be a tuple or a list"
                            f" of string with length 7, but got {type(column_names)}")

        if len(column_names) != 6:
            raise ValueError(f"the length of column_names should be 6,"
                             f" but got {len(column_names)}")

        for name in column_names:
            if not isinstance(name, str):
                raise ValueError(f"the item type of column_names should be string,"
                                 f" but got {type(name)}")

        kwargs.pop("None", None)
        squad_dataset = SQuADDataset(dataset_dir, tokenizer, stage, max_question_len, max_seq_len,
                                     doc_stride)
        return GeneratorDataset(squad_dataset, column_names, **kwargs)


class SQuADDataset:
    """SQuAD Dataset"""

    def __init__(self, dataset_dir, tokenizer, stage="train", max_question_len=64,
                 max_seq_len=384, doc_stride=128, temp_file_dir="./squad_temp"):
        r"""
        SQuAd Dataset

        Args:
            dataset_dir (str): The directory to SQuAd dataset.
            tokenizer (PreTrainedTokenizer): A tokenizer for text processing.
            stage (str): The supported keywords are in ["train", "dev"]
            max_question_len (int): The maximum number of tokens for the question,
                                    Questions longer than this will be truncated to this length.
            max_seq_len (int): Maximum sequence length.
            doc_stride (int): When splitting up a long document into chunks, how much stride to take between chunks.
            temp_file_dir (str): Save temporary files for SQuAD dataset.

        Return:
            A iterable dataset for SQuAd dataset

        Raises:
            ValueError: Error input for dataset_dir, stage.
        """
        dataset_dir = os.path.realpath(dataset_dir)
        if not os.path.isdir(dataset_dir):
            raise ValueError(f"{dataset_dir} is not existed.")

        self.dataset_dir = dataset_dir
        self.tokenizer = tokenizer
        self.max_question_len = max_question_len
        self.max_seq_len = max_seq_len
        self.doc_stride = doc_stride

        if stage == "train":
            self.is_training = True
            train_data_path = os.path.join(dataset_dir, "train-v1.1.json")
            self.examples = self._get_train_examples(train_data_path)

        elif stage == "dev":
            self.is_training = False
            dev_data_path = os.path.join(dataset_dir, "dev-v1.1.json")
            self.examples = self._get_dev_examples(dev_data_path)

        else:
            raise ValueError("unsupported stage.")

        self.input_features = convert_examples_to_features(self.examples, self.tokenizer,
                                                           self.max_seq_len, self.max_question_len,
                                                           self.doc_stride, self.is_training)

        if stage == "dev":
            self._save_eval_examples_and_features(temp_file_dir)

    def __getitem__(self, item):
        """Return input data for model"""
        feature = self.input_features[item]
        input_ids = feature.input_ids
        input_mask = feature.input_mask
        token_type_id = feature.token_type_id
        start_position = feature.start_position
        end_position = feature.end_position
        unique_id = feature.unique_id

        return input_ids, input_mask, token_type_id, start_position, end_position, unique_id

    def __len__(self):
        """Get the size of dataset"""
        return len(self.input_features)

    def _get_train_examples(self, train_data_path):
        """Get train examples."""
        return self._read_squad_examples(train_data_path)

    def _get_dev_examples(self, dev_data_path):
        """Get dev examples."""
        return self._read_squad_examples(dev_data_path)

    def _save_eval_examples_and_features(self, temp_file_dir):
        """Save examples and features for evaluation"""
        os.makedirs(temp_file_dir, exist_ok=True)
        temp_examples_file = os.path.join(temp_file_dir, "temp_examples.json")
        temp_features_file = os.path.join(temp_file_dir, "temp_features.json")

        flags_ = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        with os.fdopen(os.open(temp_examples_file, flags_, 0o750), 'w', encoding='utf-8') as f:
            for example in self.examples:
                f.write(json.dumps(example.__dict__) + '\n')

        with os.fdopen(os.open(temp_features_file, flags_, 0o750), 'w', encoding='utf-8') as f:
            for feature in self.input_features:
                f.write(json.dumps(feature.__dict__) + '\n')

    def _read_squad_examples(self, input_file_path):
        """Read a SQuAD json file into a list of SquadExample."""
        with open(input_file_path, "r") as reader:
            input_data = json.load(reader)["data"]

        examples = []
        for entry in input_data:
            title = entry["title"]
            for paragraph in entry["paragraphs"]:
                context_text = paragraph["context"]
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position_character = None
                    answer_text = None
                    answers = []

                    is_impossible = qa.get("is_impossible", False)
                    if not is_impossible:
                        if self.is_training:
                            answer = qa["answers"][0]
                            answer_text = answer["text"]
                            start_position_character = answer["answer_start"]
                        else:
                            answers = qa["answers"]

                    example = SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        context_text=context_text,
                        answer_text=answer_text,
                        start_position_character=start_position_character,
                        title=title,
                        is_impossible=is_impossible,
                        answers=answers,
                    )
                    examples.append(example)

        return examples


def convert_examples_to_features(examples, tokenizer, max_seq_len, max_question_len,
                                 doc_stride, is_training):
    """Convert examples to features"""
    input_features = []
    unique_id = 1000000000
    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_question_len:
            query_tokens = query_tokens[0: max_question_len]

        tok_to_orig_index, orig_to_tok_index, all_doc_tokens = [], [], []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position, tok_end_position = None, None
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(tokenizer,
                                                                          all_doc_tokens,
                                                                          tok_start_position,
                                                                          tok_end_position,
                                                                          example.answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_len - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        doc_spans = _get_doc_spans(doc_stride, all_doc_tokens, max_tokens_for_doc)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens, token_type_id = [], []
            token_to_orig_map, token_is_max_context = {}, {}
            tokens.append("[CLS]")
            token_type_id.append(0)
            for token in query_tokens:
                tokens.append(token)
                token_type_id.append(0)
            tokens.append("[SEP]")
            token_type_id.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
                is_max_context = _check_is_max_context(doc_spans, doc_span_index, split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                token_type_id.append(1)
            tokens.append("[SEP]")
            token_type_id.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_len:
                input_ids.append(0)
                input_mask.append(0)
                token_type_id.append(0)

            start_position, end_position = _get_positions(doc_span, tok_start_position, tok_end_position,
                                                          len(query_tokens), example.is_impossible, is_training)

            feature = InputFeatures(
                unique_id=unique_id,
                example_index=example_index,
                doc_span_index=doc_span_index,
                tokens=tokens,
                token_to_orig_map=token_to_orig_map,
                token_is_max_context=token_is_max_context,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_id=token_type_id,
                start_position=start_position,
                end_position=end_position,
                is_impossible=example.is_impossible)

            input_features.append(feature)
            unique_id += 1

    return input_features


def _get_positions(doc_span, tok_start_position, tok_end_position,
                   query_tokens_length, is_impossible, is_training):
    """Get start position and end position"""
    start_position, end_position = -1, -1
    if is_training and not is_impossible:
        # For training, if our document chunk does not contain an annotation
        # we throw it out, since there is nothing to predict.
        doc_start = doc_span.start
        doc_end = doc_span.start + doc_span.length - 1
        out_of_span = False
        if not (tok_start_position >= doc_start and
                tok_end_position <= doc_end):
            out_of_span = True
        if out_of_span:
            start_position, end_position = 0, 0
        else:
            doc_offset = query_tokens_length + 2
            start_position = tok_start_position - doc_start + doc_offset
            end_position = tok_end_position - doc_start + doc_offset

    if is_training and is_impossible:
        start_position, end_position = 0, 0

    return start_position, end_position


def _get_doc_spans(doc_stride, all_doc_tokens, max_tokens_for_doc):
    """Get doc span"""
    _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(all_doc_tokens):
            break
        start_offset += min(length, doc_stride)
    return doc_spans


def _improve_answer_span(tokenizer, doc_tokens, input_start, input_end, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


class SquadExample:
    """
    A single training/test example for the Squad dataset, as loaded from disk.
    Args:
        qas_id: The example's unique identifier
        question_text: The question string
        context_text: The context string
        answer_text: The answer string
        start_position_character: The character position of the start of the answer
        title: The title of the example
        answers: None by default, this is used during evaluation. Holds answers as well as their start positions.
        is_impossible: False by default, set to True if the example has no possible answer.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 context_text,
                 answer_text,
                 start_position_character,
                 title,
                 answers=None,
                 is_impossible=False):

        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text
        self.answer_text = answer_text
        self.title = title
        self.is_impossible = is_impossible
        self.answers = answers

        self.start_position, self.end_position = 0, 0

        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        # Split on whitespace so that different tokens may be attributed to their original position.
        for c in self.context_text:
            if self._is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        self.doc_tokens = doc_tokens
        self.char_to_word_offset = char_to_word_offset

        # Start and end positions only has a value during evaluation.
        if start_position_character is not None and not is_impossible:
            self.start_position = char_to_word_offset[start_position_character]
            self.end_position = char_to_word_offset[
                min(start_position_character + len(answer_text) - 1, len(char_to_word_offset) - 1)
            ]

    def _is_whitespace(self, c):
        """Check whether character is whitespace"""
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False


class InputFeatures:
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 token_type_id,
                 start_position=-1,
                 end_position=-1,
                 is_impossible=False):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.token_type_id = token_type_id
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
