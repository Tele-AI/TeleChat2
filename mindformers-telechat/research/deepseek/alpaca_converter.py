"""
fastchat stanford code_alpaca data convert tools.
"""

import argparse

import json
import os

import pathlib

from mindformers.tools import logger

# Prompt from stanford code_alpaca's training script

PROMPT_DICT = {
    "prompt": (
        'You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, '
        'and you only answer questions related to computer science. '
        'For politically sensitive questions, security and privacy issues, and other non-computer science questions,'
        ' you will refuse to answer.\n'
        '### Instruction:\n'
        '{instruction}\n'
        '### Response:\n'
    )
}


# pylint: disable=W0703
def main(args_param):
    data_path = pathlib.Path(args_param.data_path)
    with data_path.open(encoding='utf-8') as f:
        data = json.load(f)
    prompt = PROMPT_DICT["prompt"]

    sources = [
        prompt.format_map(example)
        for example in data
    ]

    targets = [example["output"] for example in data]

    new_data = []

    cnt = 1
    max_source_length = 0
    for s, t in zip(sources, targets):
        max_source_length = max(max_source_length, len(s))
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
    print('1')
    file = None
    flags_ = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    try:
        with os.fdopen(os.open(args_param.output_path, flags_, 0o750), 'w', encoding='utf-8') as file:
            json.dump(new_data, file, ensure_ascii=False, indent=2)
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
    print('1', max_source_length)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="alpaca-data.json")
    parser.add_argument(
        "--output_path", type=str, default="alpaca-data-conversation.json"
    )
    args = parser.parse_args()
    main(args)
