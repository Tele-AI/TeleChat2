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
"""Base Dataset Handler."""
import abc
from dataclasses import dataclass

from mindformers.auto_class import AutoTokenizer
from mindformers.models.build_tokenizer import build_tokenizer


@dataclass
class BaseTemplate:
    """Base Conv Template."""
    system_token = ""
    user_token = "### Instruction:"
    input_token = ""
    assistant_token = "### Response:"
    end_token = ""
    system = "Below is an instruction that describes a task, paired with an input that provides further context. " \
             "Write a response that appropriately completes the request. " \
             "Please note that you need to think through your response logically and step by step."
    sep = " "
    sep2 = "</s>"


class BaseInstructDataHandler:
    """Base class for instruct data handler"""
    # keys
    instruction_key = "instruction"
    input_key = "input"
    output_key = "output"
    # roles
    user_role = "user"
    assistant_role = "assistant"
    # columns after preprocess
    output_columns = ["input_ids", "labels"]
    ignore_token_id = -100
    template = None

    def __init__(self, config):
        self.seq_length = config.seq_length

        self.tokenizer = self.get_tokenizer(config)
        self.config = config
        if config.prompt_key:
            self.prompt_key = config.prompt_key
        if config.output_columns:
            self.output_columns = config.output_columns
        self.template = BaseTemplate()

    @property
    def seq_length(self):
        return self._seq_length

    @property
    def tokenizer(self):
        return self._tokenizer

    @seq_length.setter
    def seq_length(self, value):
        self._seq_length = value

    @tokenizer.setter
    def tokenizer(self, value):
        self._tokenizer = value

    @abc.abstractmethod
    def format_func(self, example):
        raise NotImplementedError

    def tokenize_func(self, messages):
        prompt = self.gen_prompt(messages)
        return self.tokenizer(prompt)

    def _preprocess(self, example):
        format_example = self.format_func(example)
        return self.tokenize_func(format_example)

    def handle(self, dataset):
        dataset = dataset.map(self._preprocess)

        if self.output_columns:
            remove_col_names = list(set(dataset.column_names) - set(self.output_columns))
            dataset = dataset.remove_columns(remove_col_names)
        return dataset

    def get_tokenizer(self, config):
        """get tokenizer"""
        tokenizer_name = config.tokenizer_name
        if tokenizer_name is not None and tokenizer_name.strip() != "":
            word_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        elif config.tokenizer:
            tokenizer_dict = config.tokenizer
            word_tokenizer = build_tokenizer(tokenizer_dict)
        else:
            return None

        if hasattr(word_tokenizer, 'add_bos_token'):
            word_tokenizer.add_bos_token = True
        if hasattr(word_tokenizer, 'add_eos_token'):
            word_tokenizer.add_eos_token = True
        return word_tokenizer

    def gen_prompt(self, messages):
        """gen prompt"""
        prompt = self.template.system_token + self.template.system_token + self.template.end_token + "\n"

        for message in messages:
            if message["from"] == self.user_role:
                prompt += self.template.user_token + "\n" + message["value"] + self.template.end_token + "\n"
            else:
                prompt += self.template.assistant_token + "\n" + message["value"] \
                          + self.template.end_token + "\n"

        return prompt
