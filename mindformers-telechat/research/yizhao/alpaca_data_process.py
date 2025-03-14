"""
transform alpaca dataset to mindrecord.
"""
import argparse
import os
from functools import partial
from typing import List, Tuple

import numpy as np
from datasets import load_dataset
from mindspore.mindrecord import FileWriter

from yizhao_model.yizhao_tokenizer import YiZhaoTokenizer


class YiZhaoPreprocessor:
    """preprocess alpaca data for yizhao"""
    def __init__(self, seq_length: int, gmask_id: int, sop_id: int, eos_id: int, user_id: int,
                 aggregated_multitask: bool):
        self.seq_length = seq_length
        self.gmask_id = gmask_id
        self.sop_id = sop_id
        self.eos_id = eos_id
        self.user_id = user_id
        self.aggregated_multitask = aggregated_multitask

    # pylint: disable=C0103
    def _pack_samples(self, sequences: List[Tuple[np.array, np.array, np.array, int]]):
        """packing sample for alpaca data"""
        if self.aggregated_multitask:
            tokens, targets, loss_masks = zip(*sequences)
        else:
            tokens, targets, loss_masks, actual_len = zip(*sequences)
        tokens = np.concatenate(tokens, axis=-1)
        targets = np.concatenate(targets, axis=-1)
        loss_masks = np.concatenate(loss_masks, axis=-1)
        division, cur_length = [], 0
        if self.aggregated_multitask:
            for _tokens, _, _ in sequences:
                cur_length += len(_tokens)
                division.append(cur_length)
            division = np.array(division + [-1] * (len(tokens) - len(division)), dtype=np.int64)
        else:
            division = np.array([0, actual_len[0]] + [-1] * (len(tokens) - 2), dtype=np.int64)
        return tokens, targets, loss_masks, division

    def pad_batch(self, tokens: np.array, targets: np.array, loss_masks: np.array, max_seq_length: int):
        """padding tokens with batch"""
        if self.aggregated_multitask:
            tokens = np.concatenate((tokens, [self.eos_id] * (max_seq_length - len(tokens))))
            targets = np.concatenate((targets, [-100] * (max_seq_length - len(targets))))
            loss_masks = np.concatenate((loss_masks, [0] * (max_seq_length - len(loss_masks))))
        else:
            tokens = np.concatenate(([self.eos_id] * (max_seq_length - len(tokens)), tokens))
            targets = np.concatenate(([-100] * (max_seq_length - len(targets)), targets))
            loss_masks = np.concatenate(([0] * (max_seq_length - len(loss_masks)), loss_masks))
        return tokens, targets, loss_masks

    def _get_single_multitask_chat_data(self, text: np.array, loss_mask: np.array, max_seq_length: int):
        """get single multitask chat data"""
        if self.aggregated_multitask:
            tokens = np.concatenate(([self.gmask_id, self.sop_id], text))
            targets = np.concatenate(([self.sop_id], text, [self.user_id]))
            loss_masks = np.concatenate(([0, 0], loss_mask))
        else:
            tokens = np.concatenate(([self.gmask_id, self.sop_id], text))
            targets = np.array(loss_mask) * text + (1 - np.array(loss_mask)) * (-100)
            targets = np.concatenate(([self.sop_id], targets, [self.user_id]))
            loss_masks = np.concatenate(([0, 0], loss_mask))
        if len(tokens) > max_seq_length:
            tokens = tokens[: max_seq_length]
            targets = targets[: max_seq_length]
            loss_masks = loss_masks[: max_seq_length]
        tokens, targets, loss_masks = self.pad_batch(tokens, targets, loss_masks, max_seq_length=max_seq_length)
        too_long = 0
        if sum(loss_masks) == 0:
            too_long = 1
        return tokens, targets, loss_masks, too_long

    def get_greedily_aggregated_multitask_chat_data(self, texts: List[np.array], loss_masks: List[np.array]):
        """get greedily aggregated multitask chat data"""
        sequences, length = [], 0
        for idx, (text, loss_mask) in enumerate(zip(texts, loss_masks)):
            cur_length = self.seq_length - length if idx + 1 == len(texts) else len(text) + 2
            tks, tgts, ls_masks, too_long = self._get_single_multitask_chat_data(text, loss_mask,
                                                                                 max_seq_length=cur_length)
            if too_long:
                print('too_long_cnt:')
                continue
            if self.aggregated_multitask:
                sequences.append((tks, tgts, ls_masks))
            else:
                sequences.append((tks, tgts, ls_masks, len(text) + 2))
            length += cur_length
        if not sequences:
            return None
        tokens, targets, loss_masks, division = self._pack_samples(sequences)
        return tokens, targets, loss_masks, division


def build_eos_attention_mask(divisions: np.ndarray, aggregated_multitask: bool):
    """build attention mask """
    seq_len = divisions.shape[0]
    attention_mask = np.zeros((seq_len,), dtype=np.int32)
    index = divisions
    prev_index = 0
    count = 0
    for i in range(index.size):
        count = count + 1
        idx = index[i]  # no need plus one
        if idx == -1:
            break
        attention_mask[prev_index: idx] = count
        prev_index = idx
    if aggregated_multitask:
        return attention_mask
    attention_mask = np.ones((seq_len,), dtype=np.int32)
    return attention_mask


def write_to_mindrecord(dataset, tokenizer, args_param):
    """write mindrecord"""
    processor_args = {
        "seq_length": args_param.seq_length,
        "eos_id": tokenizer.special_tokens['<|endoftext|>'],
        "gmask_id": tokenizer.special_tokens['[gMASK]'],
        "sop_id": tokenizer.special_tokens['<sop>'],
        "user_id": tokenizer.special_tokens['<|user|>'],
        "aggregated_multitask": args_param.aggregated_multitask
    }
    processor = YiZhaoPreprocessor(**processor_args)
    schema = {'input_ids': {"type": "int32", "shape": [-1]},
              'labels': {"type": "int32", "shape": [-1]},
              'attention_mask': {"type": "int32", "shape": [-1]},
              "loss_mask": {"type": "int32", "shape": [-1]}}
    output_file = args_param.output_file

    file_partition = 1
    dataset_type = "lmdb_chat"
    max_seq_length = args_param.seq_length
    writer = FileWriter(file_name=output_file,
                        shard_num=file_partition)
    writer.add_schema(schema, dataset_type)

    # pylint: disable=C1801
    def write_to_record(items):
        data = processor.get_greedily_aggregated_multitask_chat_data(*list(zip(*items)))
        if not data:
            return
        sample = {
            'input_ids': np.array(data[0], dtype=np.int32),
            'labels': np.array(data[1], dtype=np.int32),
            'attention_mask': build_eos_attention_mask(np.array(data[3], dtype=np.int32),
                                                       args_param.aggregated_multitask),
            'loss_mask': np.array(data[2], dtype=np.int32)
        }
        writer.write_raw_data([sample])

    items, length = [], 0
    cnt = 0
    dataset_length = len(dataset)
    aggregated_multitask = args_param.aggregated_multitask
    if aggregated_multitask:
        for i in range(dataset_length):
            item = (dataset[i]['input_ids'], dataset[i]['loss_mask'])
            new_length = len(item[0]) + 2
            if length + new_length > max_seq_length:
                if length == 0:
                    items.append(item)
                write_to_record(items)
                cnt += 1
                items.clear()
                length = 0

            length += new_length
            items.append(item)
        if len(items) > 0:
            cnt += 1
            write_to_record(items)
    else:
        for i in range(dataset_length):
            items = []
            item = (dataset[i]['input_ids'], dataset[i]['loss_mask'])
            items.append(item)
            cnt += 1
            write_to_record(items)
    writer.commit()
    print(f"Transformed {cnt} samples.")


def add_single_message(tokenized_dict, role, content, tokenizer, loss=False):
    sent = role + content
    tokens = tokenizer(sent)["input_ids"][2:]
    loss_mask = [1] * len(tokens) if loss else [0] * len(tokens)
    tokenized_dict["input_ids"].extend(tokens)
    tokenized_dict["loss_mask"].extend(loss_mask)
    return tokenized_dict


# pylint: disable=W0612
def process_sft_batch(data_point, tokenizer):
    """data_point {"id": [id1, id2], "conversations": [[样本1对话messages1], [样本2对话messages2]]}"""
    conversations_all = data_point["conversations"]
    ids = data_point["id"]
    samples = []
    for order, messages in zip(ids, conversations_all):
        result = {"input_ids": [], "loss_mask": []}
        for conv in messages:
            if conv["role"] in ["user", "system"]:
                # 用户或系统部分
                result = add_single_message(result, f"<|{conv['role']}|>", conv["content"], tokenizer, loss=False)
            elif conv["role"] == "assistant":
                # 模型回答部分
                result = add_single_message(result, "<|assistant|>", conv["content"], tokenizer, loss=True)

        samples.append(result)
    return {"samples": samples}


# pylint: disable=W0612
def finetune_dataset_process(ori_data_file_path, tokenizer, num_proc=1):
    """dataset process for finetune"""
    json_list = []
    if os.path.isdir(ori_data_file_path):
        for path, dir_list, file_list in os.walk(ori_data_file_path):
            for file_name in file_list:
                if file_name.endswith('jsonl'):
                    json_list.append(os.path.join(path, file_name))
    else:
        json_list.append(ori_data_file_path)
    print(json_list, flush=True)
    datasets_example = load_dataset("json", data_files=json_list, split="train")
    tokenized_dataset_example = datasets_example.map(partial(process_sft_batch, tokenizer=tokenizer), batched=True,
                                                     remove_columns=datasets_example.column_names,
                                                     num_proc=num_proc)
    return tokenized_dataset_example


def main(args_param):
    word_tokenizer = YiZhaoTokenizer(args_param.vocab_file)
    samples = finetune_dataset_process(args_param.ori_data_file_path, word_tokenizer,
                                       num_proc=args_param.num_proc)
    write_to_mindrecord(samples['samples'], word_tokenizer, args_param)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_file", type=str, default=r"tokenizer.model")
    parser.add_argument("--ori_data_file_path", type=str, default=r"alpaca_data.jsonl")
    parser.add_argument("--output_file", type=str, default=r"alpaca_yizhao_tokenized.mindrecord")
    parser.add_argument("--seq_length", type=int, default=8192)
    parser.add_argument("--aggregated_multitask", type=bool, default=True)
    parser.add_argument("--num_proc", type=int, default=4)
    args = parser.parse_args()
    main(args)
