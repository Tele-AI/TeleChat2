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
"""Process of ADGEN dataset."""
import argparse
import json
import numpy as np
from tqdm import tqdm
from mindspore.mindrecord import FileWriter
from mindformers.models.glm.chatglm_6b_tokenizer import ChatGLMTokenizer
from mindformers.tools.logger import logger

prompt_column = "content"
response_column = "summary"
history_column = None
prefix = ""


def get_masks(input_ids, bos_token_id=130004):
    """Get attention mask."""
    seq_length = input_ids.shape[0]

    mask = bos_token_id * np.ones(shape=(seq_length), dtype=np.int32)
    mask = np.equal(input_ids, mask)
    # 要求input_ids中有且仅有一个bos_token_id
    context_lengths = np.argwhere(mask)[:, -1]

    attention_mask = np.tril(np.ones((seq_length, seq_length), dtype=np.float32))
    for _, context_length in enumerate(context_lengths):
        attention_mask[:, :context_length] = 1

    attention_mask = np.logical_not(attention_mask.astype(np.bool_))
    attention_mask = attention_mask.astype(np.float32)
    attention_mask = np.expand_dims(attention_mask, 0)
    return attention_mask


def get_position_ids(input_ids, mask_positions, use_gmasks=None, bos_token_id=130004, position_encoding_2d=True):
    """Get position ids."""
    seq_length = input_ids.shape[0]
    if use_gmasks is None:
        use_gmasks = [False]
    mask = bos_token_id * np.ones(shape=(seq_length), dtype=np.int32)
    mask = np.equal(input_ids, mask)
    # 要求input_ids中有且仅有一个bos_token_id
    context_lengths = np.argwhere(mask)[:, -1]
    if position_encoding_2d:
        position_ids = np.arange(seq_length, dtype=np.int64)
        for i, context_length in enumerate(context_lengths):
            position_ids[context_length:] = mask_positions[i]
        block_position_ids = [np.concatenate((
            np.zeros(context_length, dtype=np.int64),
            np.arange(seq_length - context_length, dtype=np.int64) + 1
        )) for context_length in context_lengths]
        block_position_ids = np.stack(block_position_ids, axis=0).squeeze()
        position_ids = np.stack((position_ids, block_position_ids), axis=0)
    else:
        position_ids = np.arange(seq_length, dtype=np.int64)
        for i, context_length in enumerate(context_lengths):
            if not use_gmasks[i]:
                position_ids[context_length:] = mask_positions[i]
    return position_ids


def create_position_ids(input_ids, gmask_token_id=130001):
    """Get position ids."""
    seq_length = input_ids.shape[0]
    seqs = input_ids
    # 要求input_ids中, 每行有且仅有一个gMASK
    use_gmasks = gmask_token_id * np.ones(shape=(seq_length), dtype=np.int32)
    mask = np.equal(seqs, use_gmasks)
    mask_positions = np.argwhere(mask)[:, -1]

    position_ids = get_position_ids(input_ids, mask_positions=mask_positions, use_gmasks=use_gmasks)
    return position_ids


def preprocess_function(input_file, vocab_file, output_file, num_splits, max_source_length, max_target_length,
                        mode='train', ignore_pad_token_for_loss=True):
    """
    Process data file to mindrecord data file.

    Args:
        input_file (str): Original data set file.
        vocab_file (str): Vocab file path.
        output_file (str): Output MindRecord file.
        num_splits (str): The MindRecord file will be split into the number of partition.
        max_source_length (int): The max sequence length.
        max_target_length (int): The max label sequence length.
        mode (str): Dataset use to train or test. Default: 'train'.
        ignore_pad_token_for_loss (bool): Whether ignore pad token for loss, default: True.

    Returns:
        MindDataset object
    """
    if input_file is None:
        raise ValueError("Please enter a valid dataset path")

    if vocab_file is None:
        raise ValueError("Please enter a valid vocab file path")

    max_seq_length = max_source_length + max_target_length

    tokenizer = ChatGLMTokenizer(vocab_file=vocab_file)
    writer = FileWriter(output_file, num_splits, overwrite=True)
    if mode == 'train':
        data_schema = {"input_ids": {"type": "int32", "shape": [-1]},
                       "labels": {"type": "int32", "shape": [-1]},
                       "position_ids": {"type": "int32", "shape": [2, max_seq_length]},
                       "attention_mask": {"type": "int32", "shape": [1, max_seq_length, max_seq_length]}
                       }
    else:
        data_schema = {"input_ids": {"type": "int32", "shape": [-1]},
                       "labels": {"type": "int32", "shape": [-1]}}

    writer.add_schema(data_schema, "lm-schema")

    total_written = 0

    file = open(input_file, 'r')
    lines = file.readlines()
    examples = {}
    content_list = []
    summary_list = []
    for line in lines:
        content_list.append(json.loads(line)[prompt_column])
        summary_list.append(json.loads(line)[response_column])
    examples[prompt_column] = content_list
    examples[response_column] = summary_list

    model_inputs = {}

    for i in tqdm(range(len(examples[prompt_column])), ascii=True, ncols=120):
        if examples[prompt_column][i] and examples[response_column][i]:
            query, answer = examples[prompt_column][i], examples[response_column][i]

            if history_column is None:
                prompt = query
            else:
                prompt = ""
                history = examples.get(history_column)[i]
                for turn_idx, (old_query, response) in enumerate(history):
                    prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
                prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)

            prompt = prefix + prompt

            if mode == 'train':
                prompt_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
                answer_ids = tokenizer.encode(text=answer, add_special_tokens=False)

                if len(prompt_ids) > max_source_length - 1:
                    prompt_ids = prompt_ids[: max_source_length - 1]

                if len(answer_ids) > max_target_length - 2:
                    answer_ids = answer_ids[: max_target_length - 2]

                input_ids = tokenizer.build_inputs_with_special_tokens(prompt_ids, answer_ids)
                context_length = input_ids.index(tokenizer.bos_token_id)
                mask_position = context_length - 1
                label = [-100] * context_length + input_ids[mask_position + 2:]  # +1 for logits shift

                pad_len = max_seq_length - len(input_ids)
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                label = label + [tokenizer.pad_token_id] * (pad_len + 1)  # +1 for logits shift
                if ignore_pad_token_for_loss:
                    label = [(label_id if label_id != tokenizer.pad_token_id else -100) for label_id in label]

                position_ids = create_position_ids(np.array(input_ids))
                attention_mask = get_masks(np.array(input_ids))

                model_inputs["position_ids"] = np.array(position_ids).astype(np.int32)
                model_inputs["attention_mask"] = np.array(attention_mask).astype(np.int32)
            else:
                if len(prompt) > max_source_length - 2:
                    prompt = prompt[: max_source_length - 2]

                if len(answer) > max_target_length - 2:
                    answer = answer[: max_target_length - 2]

                input_ids = tokenizer.encode(text=prompt, add_special_tokens=True)
                label = tokenizer.encode(text=answer, add_special_tokens=True)

                pad_len = max_source_length - len(input_ids)
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len

            model_inputs["input_ids"] = np.array(input_ids).astype(np.int32)
            model_inputs["labels"] = np.array(label).astype(np.int32)

            writer.write_raw_data([model_inputs])
            total_written += 1

    writer.commit()
    file.close()
    logger.info(f"Wrote {total_written} total instances")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OPT convert script")
    parser.add_argument('--input_file',
                        type=str,
                        required=True,
                        default='./AdvertiseGen/train.json',
                        help="The number of layers of the model to be converted.")
    parser.add_argument("--vocab_file",
                        type=str,
                        default='ice_text.model',
                        required=True,
                        help="The torch checkpoint path.")
    parser.add_argument("--output_file",
                        type=str,
                        required=True,
                        default="adgen_train.mindrecord",
                        help="The output mindspore checkpoint path.")
    parser.add_argument("--max_source_length",
                        type=int,
                        default=64,
                        help="The max input sequence length")
    parser.add_argument("--max_target_length",
                        type=int,
                        default=64,
                        help="The max label sequence length")
    parser.add_argument("--mode",
                        type=str,
                        default="train",
                        help="Process train or eval dataset.")

    opt = parser.parse_args()

    preprocess_function(
        input_file=opt.input_file,
        vocab_file=opt.vocab_file,
        output_file=opt.output_file,
        num_splits=1,
        max_source_length=opt.max_source_length,
        max_target_length=opt.max_target_length,
        mode=opt.mode,
        ignore_pad_token_for_loss=True
    )
