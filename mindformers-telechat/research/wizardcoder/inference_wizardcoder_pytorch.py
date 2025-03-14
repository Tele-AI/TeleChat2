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
"""test wizardcoder pytorch"""
import sys
import time
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def evaluate(prompts, tokenizer, model, max_length=1024, **kwargs):
    """evaluate function"""
    start_time_with_tokenizer = time.time()
    inputs = tokenizer(prompts, return_tensors="pt", max_length=max_length, truncation=True, padding=True)
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=1,
        top_p=1,
        top_k=1,
        do_sample=False,
        num_beams=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_length=max_length,
        **kwargs
    )
    start_time_no_tokenizer = time.time()
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True
        )
    seq = generation_output.sequences
    end_time_no_tokenizer = time.time()
    output = tokenizer.batch_decode(seq, skip_special_tokens=True)
    end_time_with_tokenizer = time.time()
    elapsed_time_with_tokenizer = end_time_with_tokenizer - start_time_with_tokenizer
    elapsed_time_no_tokenizer = end_time_no_tokenizer - start_time_no_tokenizer
    generate_length = sum([len(item) for item in seq]) - sum([len(ids) for ids in input_ids])
    return output, generate_length, elapsed_time_with_tokenizer, elapsed_time_no_tokenizer


def generate_prompt(input_query):
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Create a Python script for this problem:
{input_query}

### Response:"""


def main(args, with_prompt=True, load_8bit: bool = False):

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    if not load_8bit:
        model.half()

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    input_data = [["使用python编写快速排序代码"] * args.batch_size]
    for _, instruction in enumerate(input_data):
        print('\n开始推理.......')
        if with_prompt:
            prompt = instruction
        else:
            prompt = generate_prompt(instruction)
        decode_output, generate_length, time_with_tokenizer, time_no_tokenizer = \
            evaluate(prompt, tokenizer, model, max_length=args.seq_length)
        print("output: \n", decode_output[0])
        speed_with_tokenizer = generate_length / time_with_tokenizer
        speed_no_tokenizer = generate_length / time_no_tokenizer
        print("\n generate length: ", generate_length,
              " elapsed_time_with_tokenizer: ", time_with_tokenizer,
              " elapsed_time_no_tokenizer: ", time_no_tokenizer,
              " speed_with_tokenizer: ", speed_with_tokenizer,
              " speed_no_tokenizer: ", speed_no_tokenizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default='', type=str,
                        help='base model')
    parser.add_argument('--seq_length', default=2048, type=int,
                        help='batch_size')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='batch_size')

    opt = parser.parse_args()
    main(opt)
