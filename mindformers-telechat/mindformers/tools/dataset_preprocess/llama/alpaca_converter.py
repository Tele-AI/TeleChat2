"""
fastchat stanford alpaca data convert tools.
"""

import argparse

import json
import os

import pathlib

from mindformers.tools import logger

# Prompt from stanford alpaca's training script

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),

    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),

}


# pylint: disable=W0703
def main(args_param):
    data_path = pathlib.Path(args_param.data_path)
    with data_path.open() as f:
        data = json.load(f)
    prompt_input, prompt_no_input = (
        PROMPT_DICT["prompt_input"],
        PROMPT_DICT["prompt_no_input"],
    )

    sources = [
        prompt_input.format_map(example)
        if example.get("input", "") != ""
        else prompt_no_input.format_map(example)
        for example in data
    ]

    targets = [example["output"] for example in data]

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
    try:
        flags_ = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        with os.fdopen(os.open(args_param.output_path, flags_, 0o750), 'w') as f:
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
    main(args)
