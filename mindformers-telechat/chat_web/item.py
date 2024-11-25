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
"""Item"""
from typing import Optional, Union, List
import time
from pydantic import BaseModel, Field

from config.server_config import default_config


class ChatMessage(BaseModel):
    """ChatMessage"""
    role: str
    content: str


class DeltaMessage(BaseModel):
    """DeltaMessage"""
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    """ChatCompletionRequest"""
    question: str = None
    history: List = None
    messages: List[ChatMessage]
    do_sample: Optional[bool] = default_config['default_generation_args']['do_sample']
    temperature: Optional[float] = default_config['default_generation_args']['temperature']
    repetition_penalty: Optional[float] = default_config['default_generation_args']['repetition_penalty']
    top_p: Optional[float] = default_config['default_generation_args']['top_p']
    top_k: Optional[int] = default_config['default_generation_args']['top_k']
    max_length: Optional[int] = default_config['default_generation_args']['max_length']
    stream: Optional[bool] = False


class ChatCompletionResponseChoice(BaseModel):
    """ChatCompletionResponseChoice"""
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class ChatCompletionResponseStreamChoice(BaseModel):
    """ChatCompletionResponseStreamChoice"""
    index: int
    delta: DeltaMessage
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    """ChatCompletionResponse"""
    object: str
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))


class ChatErrorOutResponseChoice(BaseModel):
    """ChatErrorOutResponseChoice"""
    index: int
    message: str
    finish_reason: Optional[str] = None


class ChatErrorOutResponseStreamChoice(BaseModel):
    """ChatErrorOutResponseStreamChoice"""
    index: int
    message: str
    finish_reason: Optional[str] = None


class ChatErrorOutResponse(BaseModel):
    """ChatErrorOutResponse"""
    object: str
    choices: List[Union[ChatErrorOutResponseChoice, ChatErrorOutResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
