"""
fastchat stanford alpaca data convert tools.
"""

import argparse
import json
from pathlib import Path
from typing import Union, List

# Prompt from stanford alpaca's training script

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}"
    ),

    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}"
    )
}


def search_data(root_path: Union[str, Path]) -> List[Path]:
    """find json file in root path """
    if isinstance(root_path, str):
        root_path = Path(root_path)

    data_paths = []

    if root_path.is_dir():
        for path in root_path.rglob('*.jsonl'):
            data_paths.append(path)
        for path in root_path.rglob('*.json'):
            data_paths.append(path)
    else:
        if root_path.suffix in {'.jsonl', '.json'}:
            data_paths.append(root_path)

    return data_paths


def add_json_data(data, json_data):
    if isinstance(json_data, list):
        data.extend(json_data)
    else:
        data.append(json_data)


def load_json_or_jsonl(path_list: List[Path]) -> List[dict]:
    """ load json file"""
    data = []
    for data_path in path_list:
        with Path(data_path).open(encoding='utf-8') as f:
            if data_path.suffix == ".json":
                json_data = json.loads(f.read())
                add_json_data(data, json_data)
            elif data_path.suffix == ".jsonl":
                data.extend([json.loads(j) for j in f.readlines()])
    return data



def main(args_param):
    data_paths = search_data(args_param.data_path)
    data = load_json_or_jsonl(data_paths)

    sources = [
        PROMPT_DICT["prompt_input"].format_map(example)
        if example.get("input", "") != ""
        else PROMPT_DICT["prompt_no_input"].format_map(example)
        for example in data
    ]
    targets = [example["output"] for example in data]

    new_data = []
    cnt = 1
    for s, t in zip(sources, targets):
        new_data.append({"id": str(cnt),
                         "conversations": [{"role": "user", "content": s},
                                           {"role": "assistant", "content": t}]})
        cnt += 1

    output_path = Path(args_param.output_path)
    if output_path.is_dir():
        output_path.joinpath("processed_data.jsonl")
    with output_path.open('w', encoding='utf-8') as f:
        for obj in new_data:
            json_str = json.dumps(obj, ensure_ascii=False)
            f.write(json_str + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=r"alpaca_data.json")
    parser.add_argument("--output_path", type=str, default=r"alpaca_data.jsonl")
    args = parser.parse_args()
    main(args)
