"""
fastchat stanford alpaca data convert tools.
"""
import argparse
import json
import os

import pathlib


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
            instruction = instruction + ": " + example['input']
            sources.append(instruction)

    targets = []
    for example in data:
        targets.append(example['output'])

    new_data = []
    for s, t in zip(sources, targets):
        new_data.append({
            "type": "chatml",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": s,
                },
                {
                    "role": "assistant",
                    "content": t,
                },
            ]
        })

    flags_ = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    with os.fdopen(os.open(output_path, flags_, 0o750), "w") as f:
        for sample in new_data:
            f.write(json.dumps(sample) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="alpaca-data.json")
    parser.add_argument(
        "--output_path", type=str, default="alpaca-data-conversation.json"
    )
    args = parser.parse_args()
    main(args.data_path, args.output_path)
