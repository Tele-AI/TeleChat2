# Adapted from lm-sys@FastChat. Below is the original copyright:
#    Copyright 2024 Wei-Lin Chiang, Lianmin Zheng, Ying Sheng
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""conversation prompt templates"""

import dataclasses
from enum import auto, Enum
from typing import List, Any


class SeparatorStyle(Enum):
    """Different separator style."""

    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()
    NO_COLON_SINGLE = auto()
    BAIZE = auto()
    DOLLY = auto()
    RWKV = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""

    # System prompts
    system: str
    # Two roles
    roles: List[str]
    # All messages
    messages: List[List[str]]
    # Offset of few shot examples
    offset: int
    # Separator
    sep_style: SeparatorStyle
    sep: str
    sep2: str = None
    # Stop criteria (the default one is EOS token)
    stop_str: str = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    # Used for the state in the gradio servers.
    conv_id: Any = None
    skip_next: bool = False
    model_name: str = None

    def get_prompt(self):
        """Get the prompt for generation."""
        seps = [self.sep, self.sep2]
        ret = ""
        for i, (role, message) in enumerate(self.messages):
            if message:
                ret += role + ": " + message + seps[i % 2]
            else:
                ret += role + ":"
        return ret

    def append_message(self, role, message):
        """Append a new message."""
        self.messages.append([role, message])

    def to_openai_api_messages(self):
        """Convert the conversation to OpenAI chat completion format."""
        ret = [{"role": "system", "content": self.system}]

        for i, (_, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append({"role": "user", "content": msg})
            else:
                if msg is not None:
                    ret.append({"role": "assistant", "content": msg})
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
            conv_id=self.conv_id,
            model_name=self.model_name,
        )

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "conv_id": self.conv_id,
            "model_name": self.model_name,
        }
