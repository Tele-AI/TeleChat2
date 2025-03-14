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
"""
Prompt processor
"""
import copy

import numpy as np

from mindformers.tools import MindFormerModuleType, MindFormerRegister
from mindformers.tools.dataset_preprocess.llama.conversation import conv_templates


@MindFormerRegister.register(MindFormerModuleType.PROCESSOR)
class BasePromptProcessor:
    """
        Based class for prompt processor
    """

    def __init__(self, tokenizer, roles):
        super().__init__()
        self.tokenizer = tokenizer
        self.roles = roles

    def build_prompt(self, raw_inputs, result_recorder, **kwargs):
        raise NotImplementedError

    def build_labels(self, text_id_list, result_recorder, ignore_token_id, **kwargs):
        raise NotImplementedError


@MindFormerRegister.register(MindFormerModuleType.PROCESSOR)
class QwenPromptProcessor(BasePromptProcessor):
    """
    Prompt processor for Qwen
    """

    # pylint: disable=W0613
    def build_prompt(self, raw_inputs, result_recorder, **kwargs):
        conversations = []
        no_image_tag = result_recorder.get("no_image_tag")[0]
        system_message = "You are a helpful assistant."
        if self.roles[raw_inputs[0][0]] != self.roles["human"]:
            raw_inputs = raw_inputs[1:]
        conversation = self.tokenizer.apply_chat_template([{"role": "system", "content": system_message}],
                                                          tokenize=False)
        conversations.append(conversation)
        role_speak = ["system"]
        for source in raw_inputs:
            role = self.roles[source[0]].lower()
            conv = [{"role": role, "content": source[1]}]
            conversation = self.tokenizer.apply_chat_template(conv, tokenize=False)
            conversations.append(conversation)
            role_speak.append(role)
        result_recorder.put("role_speak", role_speak)
        if no_image_tag:
            conversations[-1] += self.tokenizer.image_token

        return conversations

    # pylint: disable=W0613
    def build_labels(self, text_id_list, result_recorder, ignore_token_id, **kwargs):
        targets = copy.deepcopy(text_id_list)
        role_speak = result_recorder.get("role_speak")
        new_target = []
        for idx, target in enumerate(targets):
            role = role_speak[idx]
            if role in ["user", "system"]:
                new_target += [ignore_token_id] * len(target)
            else:
                if isinstance(target, np.ndarray):
                    target = target.tolist()
                new_target += target

        return np.array(new_target)


@MindFormerRegister.register(MindFormerModuleType.PROCESSOR)
class VicunaPromptProcessor(BasePromptProcessor):
    """
    Prompt processor for vicuna
    """

    def __init__(self, tokenizer, roles, conv):
        super().__init__(tokenizer, roles)
        if conv in conv_templates.keys():
            self.conv = copy.deepcopy(conv_templates[conv])

    # pylint: disable=W0613
    def build_prompt(self, raw_inputs, result_recorder, **kwargs):
        # Apply prompt templates
        conversations = []
        self.conv.messages = []
        no_image_tag = result_recorder.get("no_image_tag")[0]
        # Skip the first one if it is not from human
        if self.roles[raw_inputs[0][0]] != self.conv.roles[0]:
            raw_inputs = raw_inputs[1:]
        for i, source in enumerate(raw_inputs):
            role = self.roles[source[0]]
            if role == self.conv.roles[i % 2]:
                self.conv.append_message(role, source[1])
            else:
                raise ValueError(f"Current role is {role}, correct role should be {self.conv.roles[i % 2]},"
                                 f"please check.")
        conversation = self.conv.get_prompt()
        if no_image_tag:
            conversation += self.tokenizer.image_token
        conversation = conversation.replace("<image>", "<image> ") if not (hasattr(self.tokenizer, "legacy") and
                                                                           self.tokenizer.legacy) else conversation
        conversations.append(conversation)
        result_recorder.put("conversations", conversations)
        return conversations

    def build_labels(self, text_id_list, result_recorder, ignore_token_id, **kwargs):
        context_length = kwargs.get("context_length")
        image_token = self.tokenizer.image_token
        sep = self.conv.sep + self.conv.roles[1] + ": "
        conversations = result_recorder.get("conversations")
        targets = copy.deepcopy(text_id_list)
        no_image_tag = result_recorder.get("no_image_tag")[0]
        for conversation, target in zip(conversations, targets):
            total_len = len(target)

            rounds = conversation.split(self.conv.sep2)
            if self.tokenizer.add_bos_token:
                cur_len = 1
                target[:cur_len] = ignore_token_id
            else:
                cur_len = 0
            for i, rou in enumerate(rounds):
                if rou == "":
                    break

                parts = rou.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                if image_token in rou:
                    round_len = len(self.tokenizer(rou).input_ids) + context_length - 1
                    instruction_len = len(
                        self.tokenizer(parts[0]).input_ids) - 2 + context_length - 1
                else:
                    round_len = len(self.tokenizer(rou).input_ids)
                    instruction_len = len(self.tokenizer(parts[0]).input_ids) - 2
                if not self.tokenizer.add_bos_token:
                    round_len += 1
                    instruction_len += 1
                if i != 0 and not (hasattr(self.tokenizer, "legacy") and self.tokenizer.legacy):
                    round_len -= 1
                    instruction_len -= 1
                target[cur_len: cur_len + instruction_len] = ignore_token_id

                cur_len += round_len
            target[cur_len:] = ignore_token_id

            if cur_len < self.tokenizer.model_max_length:
                if cur_len != total_len and not no_image_tag:
                    target[:] = ignore_token_id
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )

        return targets[0]

    # pylint: disable=W0613
    def build_predict_prompt(self, raw_inputs, **kwargs):
        """predict prompt generation"""
        user_inputs = "".join(raw_inputs)
        self.conv.messages = []
        self.conv.append_message(self.conv.roles[0], user_inputs)
        self.conv.append_message(self.conv.roles[1], None)
        conversation = self.conv.get_prompt()
        return conversation
