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
"""ChatGLM32k preprocess."""
import os
import json
import argparse
from tqdm import tqdm


def build_prompt(used_prompt, input_file, output_file):
    """build prompt"""
    new_samples = []
    with open(input_file, encoding="utf-8") as in_file:
        while True:
            line = in_file.readline()
            if not line:
                break
            sample = json.loads(line)
            content = used_prompt.format_map(sample)
            summary = sample["answers"][0]
            new_samples.append({"content": content, "summary": summary})
    flags_ = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    with os.fdopen(os.open(output_file, flags_, 0o750), "a+", encoding='utf-8') as out_file:
        for new_sample in new_samples:
            json.dump(new_sample, out_file, ensure_ascii=False)
            out_file.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--output_path", type=str, default="./")
    parser.add_argument("--prompt_config_file", type=str, default="./dataset2prompt.json")
    parser.add_argument("--is_longbench_e", type=bool, default=True)

    args = parser.parse_args()
    data_path = args.data_path
    output_path = args.output_path
    prompt_config_file = args.prompt_config_file
    is_longbench_e = args.is_longbench_e

    with open(prompt_config_file, encoding='utf-8') as f:
        prompts = json.load(f)

    files = os.listdir(data_path)
    longbench_files = []
    longbench_e_files = []
    for file in files:
        file_name = os.path.splitext(file)[0]
        if file_name.endswith("_e"):
            longbench_e_files.append(file)
        else:
            longbench_files.append(file)

    if is_longbench_e:
        output_file_path = os.path.join(output_path, "longbench_e.jsonl")
        data_files = longbench_e_files
    else:
        output_file_path = os.path.join(output_path, "longbench.jsonl")
        data_files = longbench_files

    if os.path.exists(output_file_path):
        os.unlink(output_file_path)

    for i in tqdm(range(len(data_files))):
        file = data_files[i]
        file_name = os.path.splitext(file)[0]
        prompt_name = file_name.rstrip("_e") if file_name.endswith("_e") else file_name
        prompt = prompts[prompt_name]
        file_path = os.path.join(data_path, file)
        build_prompt(prompt, file_path, output_file_path)
