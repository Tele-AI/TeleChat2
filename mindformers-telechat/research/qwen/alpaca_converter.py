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
            instruction = instruction + ": " + example['input']
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
                        "from": "user",
                        "value": s,
                    },
                    {
                        "from": "assistant",
                        "value": t,
                    },
                ],
            }
        )

        cnt += 1
    file = None
    flags_ = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    try:
        with os.fdopen(os.open(output_path, flags_, 0o750), "w") as fi:
            json.dump(new_data, fi, indent=2)
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
