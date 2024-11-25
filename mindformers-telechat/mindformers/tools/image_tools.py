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

'''
Image tools
'''
import os
import requests
import urllib3

import PIL
import mindspore as ms


def load_image(content, timeout=4):
    """load image"""
    if isinstance(content, str):
        if content.startswith("https://") or content.startswith("http://"):

            try:
                response = requests.get(content, stream=True, timeout=timeout)
            except (TimeoutError, urllib3.exceptions.MaxRetryError,
                    requests.exceptions.ProxyError) as exc:
                raise ConnectionError(f"Connect error, please download {content}.") from exc

            content = PIL.Image.open(response.raw)
        elif os.path.isfile(content):
            content = PIL.Image.open(content)
        else:
            raise ValueError(
                f"{content} is not a valid path. If URL, it must start with `http://` or `https://`."
            )

    if isinstance(content, ms.Tensor):
        return content

    if not isinstance(content, PIL.Image.Image):
        raise ValueError(
            "Input should be an url linking to an image,"
            " a local path, a Mindspore Tensor, or a PIL image."
        )
    content = PIL.ImageOps.exif_transpose(content)
    content = content.convert("RGB")
    return content
