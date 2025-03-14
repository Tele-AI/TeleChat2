# Copyright 2025 Huawei Technologies Co., Ltd
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
"""VLMEvalKit-MindFormers supported models."""
from functools import partial

from toolkit.benchmarks.vlmevalkit_models.cogvlm2_image import CogVlmImage
from toolkit.benchmarks.vlmevalkit_models.cogvlm2_video import CogVlmVideo


SUPPORT_MODEL_LIST = {"image": "cogvlm2-image-llama3-chat", "video": "cogvlm2-video-llama3-chat"}


def get_model(args):
    mindformers_series = {}
    mindformers_series['cogvlm2-image-llama3-chat'] = partial(CogVlmImage, args.model_path, args.config_path)
    mindformers_series['cogvlm2-video-llama3-chat'] = partial(CogVlmVideo, args.model_path, args.config_path)
    return mindformers_series
