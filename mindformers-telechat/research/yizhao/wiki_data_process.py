"""
transform wiki dataset to mindrecord.
"""
import argparse
import os
import time
from functools import partial

import numpy as np
from datasets import load_dataset
from mindspore.mindrecord import FileWriter
from tqdm import tqdm

from yizhao_model.yizhao_tokenizer import YiZhaoTokenizer


def chunks(lst, n):
    """ yield n sized chunks from list"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def process_func_tokenized(data_point, tokenizer, seq_length, model, repeat=1):
    """ Processes a data point to convert text into tokenized input for yizhao """

    eos_id = tokenizer.special_tokens['<|endoftext|>']
    gmask_id = tokenizer.special_tokens['[gMASK]']
    sop_id = tokenizer.special_tokens['<sop>']
    pad_id = -100

    content = []
    sentences = data_point['text']
    for sentence in sentences:
        if model == 'YiZhao':
            content += tokenizer(sentence)['input_ids'][2:] + [eos_id]
    content_out = []
    for _ in range(repeat):
        content_out.extend(content)
    content = content_out
    input_ids = []
    for chunk in chunks(content, seq_length - 1):
        chunk = [gmask_id, sop_id] + chunk  # chunk.len = 8193
        chunk = chunk + [pad_id] * (seq_length + 1 - len(chunk))
        input_ids.append(chunk)
    return {"input_ids": np.array(input_ids, dtype=np.int32)}


# pylint: disable=W0612
def pretrain_dataset_process(model, ori_data_file_path, output_data_file_path, tokenizer, seq_length,
                             num_proc=100, file_partition=1, parallel_writer=False):
    """ data set process for pretrain """
    json_list = []
    if os.path.isdir(ori_data_file_path):
        paths = os.walk(ori_data_file_path)
        for path, dir_list, file_list in paths:
            for file_name in file_list:
                path_tmp = os.path.join(path, file_name)
                if path_tmp.endswith('jsonl'):
                    json_list.append(path_tmp)
    else:
        json_list.append(ori_data_file_path)
    print(json_list, flush=True)
    datasets_example = load_dataset("text", data_files=json_list, split="train")
    tokenized_dataset_example = datasets_example.map(partial(process_func_tokenized, tokenizer=tokenizer,
                                                             seq_length=seq_length, model=model), batched=True,
                                                     remove_columns=datasets_example.column_names,
                                                     num_proc=num_proc)

    time1 = time.time()
    pd_dataset = tokenized_dataset_example.to_pandas()
    print('tokenized_dataset_example:', tokenized_dataset_example, flush=True)
    convert_pandas_time = time.time() - time1
    print(f"convert pandas time is: {convert_pandas_time}")

    # using mindrecord api to save preprocess results.
    schema = {'input_ids': {"type": "int32", "shape": [-1]}}
    # set file writer, refer to <https://www.mindspore.cn/docs/zh-CN/r2.2/api_python/mindspore.mindrecord.html>
    writer = FileWriter(file_name=output_data_file_path,
                        shard_num=file_partition,
                        overwrite=True)
    writer.add_schema(schema, desc="pt_dataset")

    with tqdm(total=len(pd_dataset)) as pbar:
        pbar.set_description("Writing mindrecords")
        transforms_count = 0
        transform_steps = 10000
        while transforms_count < len(pd_dataset):
            if transforms_count + transform_steps >= len(pd_dataset):
                # end of dataframe, write the end and exit.
                data_samples = pd_dataset.iloc[transforms_count:].to_dict('records')
                pbar.update(len(pd_dataset) - transforms_count)
                transforms_count = len(pd_dataset)
            else:
                # slice a range of data and convert to dict list
                data_samples = pd_dataset.iloc[transforms_count:transforms_count + transform_steps].to_dict('records')
                transforms_count += transform_steps
                pbar.update(transform_steps)
            # write to mindrecord
            writer.write_raw_data(data_samples, parallel_writer=parallel_writer)

    print("Transformed {} records.".format(transforms_count))
    # write done, commit it
    writer.commit()
    print(f"Transform finished, output files refer: {output_data_file_path}")


if __name__ == "__main__":
    work_path = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_file', default=r'tokenizer.model', type=str)
    parser.add_argument('--ori_file_path', default=r'wiki.train.tokens', type=str)
    parser.add_argument('--output_file_path', default=r'wiki.mindrecord', type=str)
    parser.add_argument('--seq_length', default=8192, type=int)
    parser.add_argument('--num_proc', default=4, type=int)
    args = parser.parse_args()

    word_tokenizer = YiZhaoTokenizer(args.vocab_file)
    pretrain_dataset_process(model='YiZhao',
                             ori_data_file_path=args.ori_file_path,
                             output_data_file_path=args.output_file_path,
                             tokenizer=word_tokenizer,
                             seq_length=int(args.seq_length),
                             num_proc=args.num_proc,
                             file_partition=1,
                             parallel_writer=False)
