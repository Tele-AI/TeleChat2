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
"""code alpaca instruct datahandler"""
import numpy as np

from mindformers.dataset.handler.base_handler import BaseInstructDataHandler
from mindformers.tools.register import MindFormerModuleType, MindFormerRegister

IGNORE_TOKEN_ID = -100
PROMPT_DICT = {
    "prompt": (
        'You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, '
        'and you only answer questions related to computer science. '
        'For politically sensitive questions, security and privacy issues, and other non-computer science questions,'
        ' you will refuse to answer.\n'
        '### Instruction:\n'
        '{instruction}\n'
        '### Response:\n'
    )
}


@MindFormerRegister.register(MindFormerModuleType.DATA_HANDLER)
class CodeAlpacaInstructDataHandler(BaseInstructDataHandler):
    """CodeAlpaca Data Handler"""
    user_role = "human"
    assistant_role = "gpt"

    def format_func(self, example):
        """format func"""
        prompt = PROMPT_DICT["prompt"]

        source = prompt.format_map(example)

        target = example["output"]

        new_example = {
            "conversations": [
                {
                    "from": "human",
                    "value": source,
                },
                {
                    "from": "gpt",
                    "value": target,
                },
            ],
        }

        return new_example

    def tokenize_func(self, messages):
        """conversation preprocess."""
        source = messages["conversations"]
        seq_length = self.seq_length + 1

        input_s = source[0]["value"].lstrip('\n').rstrip(' ') + source[1]["value"] + '\n'
        q_len = len(self.tokenizer(source[0]["value"].lstrip('\n').rstrip(' '))['input_ids']) - 1
        conversation = [input_s, q_len]

        ids = self.tokenizer(conversation[0])['input_ids']
        mask = self.tokenizer(conversation[0])['attention_mask']
        d = {'input_ids': ids, 'attention_mask': mask}
        target = np.array(d['input_ids'])
        len_inputid = len(d['input_ids'])
        l_target = len(target)
        if l_target < seq_length:
            d['input_ids'] = np.pad(d['input_ids'], ((0), (seq_length - len_inputid)),
                                    mode='constant', constant_values=32014)
            target = np.pad(target, ((0), (seq_length - l_target)),
                            mode='constant', constant_values=IGNORE_TOKEN_ID)

        target[:conversation[1]] = IGNORE_TOKEN_ID
        targets = target[:seq_length].tolist()
        input_ids = d['input_ids'][:seq_length]
        input_ids = np.array(input_ids, dtype=np.int32)
        targets = np.array(targets, dtype=np.int32)

        return dict(
            input_ids=input_ids,
            labels=targets,
        )
