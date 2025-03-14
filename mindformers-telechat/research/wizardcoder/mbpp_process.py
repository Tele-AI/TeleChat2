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
"""mbpp process script"""

import glob
import argparse
import json
import os

from tqdm import tqdm


def read_file_method(opt):
    """read file method"""
    file_tuple = [(int(file.split("/")[-1].split(".")[0]), file) for file in glob.glob(opt.path + '/*.json')]
    sorted_files = sorted(file_tuple, key=lambda x: x[0])
    gen_files = [item[1] for item in sorted_files]
    return gen_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help="")
    parser.add_argument('--out_path', type=str, help="")
    args = parser.parse_args()

    files = read_file_method(args)
    print("{} files in {}".format(len(files), args.path))

    res = []
    count = 0
    for code_file in tqdm(files, total=len(files)):
        with open(code_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            completion = data["output"][0]
            if '```python' in completion:
                def_line = completion.index('```python')
                completion = completion[def_line:].strip()
                completion = completion.replace('```python', '')
                try:
                    next_line = completion.index('\n```')
                    completion = completion[:next_line].strip()
                except ValueError:
                    print("completion: ", completion)
                    count += 1
            if "__name__ == \"__main__\"" in completion:
                next_line = completion.index('if __name__ == "__main__":')
                completion = completion[:next_line].strip()

            if "# Example usage" in completion:
                next_line = completion.index('# Example usage')
                completion = completion[:next_line].strip()

            if "# Test examples" in completion:
                next_line = completion.index('# Test examples')
                completion = completion[:next_line].strip()

            res.append([completion])
    print("count: ", count)
    print("save to {}".format(args.out_path))
    flags_ = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    with os.fdopen(os.open(args.out_path, flags_, 0o750), "w", encoding='utf-8') as fout:
        json.dump(res, fout)
