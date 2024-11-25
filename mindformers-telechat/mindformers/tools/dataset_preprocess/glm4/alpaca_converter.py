"""
fastchat stanford alpaca data convert tools.
"""
import argparse
import json
import pathlib
import os


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
            "messages": [
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
    with os.fdopen(os.open(output_path, flags_, 0o750), 'w', encoding='utf-8') as f:
        for data in new_data:
            f.write(json.dumps(data) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="alpaca_data.json")
    parser.add_argument("--output_path", type=str, default="alpaca_glm4_data.jsonl")
    args = parser.parse_args()
    main(args.data_path, args.output_path)
