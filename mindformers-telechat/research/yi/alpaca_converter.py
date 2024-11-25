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
fastchat stanford alpaca data convert tools.
"""
import argparse
import json
import os

import pathlib

from mindformers.tools import logger


# pylint: disable=W0703
def main(data_path, output_path):
    data_path = pathlib.Path(data_path)
    with data_path.open() as f:
        data = json.load(f)

    sources = []
    for example in data:
        if example.get("input", "") == "":
            sources.append(example['instruction'])
        else:
            instruction = example['instruction']
            if instruction[-1] == ".":
                instruction = instruction[:-1]
            instruction = instruction + "\n" + example['input']
            sources.append(instruction)

    targets = []
    for example in data:
        targets.append(example['output'])

    new_data = []
    cnt = 1
    for s, t in zip(sources, targets):
        new_data.append(
            {
                "id": str(cnt),
                "conversations": [
                    {
                        "from": "human",
                        "value": s,
                    },
                    {
                        "from": "gpt",
                        "value": t,
                    },
                ],
            }
        )

        cnt += 1
    file = None
    flags_ = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    try:
        with os.fdopen(os.open(output_path, flags_, 0o750), "w") as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2)
    except FileNotFoundError as file_not_found_error:
        logger.error(file_not_found_error)
    except UnicodeDecodeError as decode_error:
        logger.error(decode_error)
    except IOError as io_error:
        logger.error(io_error)
    except Exception as exception:
        logger.error(exception)
    finally:
        if file is not None:
            file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="alpaca-data.json")
    parser.add_argument(
        "--output_path", type=str, default="alpaca-data-conversation.json"
    )
    args = parser.parse_args()
    main(args.data_path, args.output_path)
