# Copyright 2024 Huawei Technologies Co., Ltd
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
"""do infer using wizardcoder"""
import os
import argparse

from mindspore import log as logger

from mindformers import MindFormerConfig
from mindformers.tools.utils import str2bool
from mindformers.core.context import build_context
from wizardcoder_config import WizardCoderConfig
from wizardcoder import WizardCoderLMHeadModel
from wizardcoder_tokenizer import WizardCoderTokenizer


def load_model_and_tokenizer(args):
    """load model and tokenizer using args."""
    config = MindFormerConfig(os.path.realpath(args.config_path))
    config.context.device_id = args.device_id
    build_context(config)
    wizard_config = WizardCoderConfig.from_pretrained(os.path.realpath(args.config_path))
    wizard_config.use_past = args.use_past
    wizard_config.batch_size = args.batch_size
    tokenizer = WizardCoderTokenizer(config.processor.tokenizer.vocab_file,
                                     config.processor.tokenizer.merge_file)
    model = WizardCoderLMHeadModel(wizard_config)
    return model, tokenizer


def main(args):
    """do infer"""
    model, tokenizer = load_model_and_tokenizer(args)
    # test 4 cases:
    prompts = [
        [
            'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nCreate a Python script for this problem:\n\nWrite a python function to find the volume of a triangular prism.\nTest examples:\nassert find_Volume(10,8,6) == 240\nassert find_Volume(3,2,2) == 6\nassert find_Volume(1,2,1) == 1\n\n### Response:'],
        [
            'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nCreate a Python script for this problem:\n\nWrite a function to find sequences of lowercase letters joined with an underscore.\nTest examples:\nassert text_lowercase_underscore("aab_cbbbc")==(\'Found a match!\')\nassert text_lowercase_underscore("aab_Abbbc")==(\'Not matched!\')\nassert text_lowercase_underscore("Aaab_abbbc")==(\'Not matched!\')\n\n### Response:'],
        [
            'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nCreate a Python script for this problem:\n\nWrite a function to check if the given tuple list has all k elements.\nTest examples:\nassert check_k_elements([(4, 4), (4, 4, 4), (4, 4), (4, 4, 4, 4), (4, )], 4) == True\nassert check_k_elements([(7, 7, 7), (7, 7)], 7) == True\nassert check_k_elements([(9, 9), (9, 9, 9, 9)], 7) == False\n\n### Response:'],
        [
            'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nCreate a Python script for this problem:\n\nWrite a function to find the n-th rectangular number.\nTest examples:\nassert find_rect_num(4) == 20\nassert find_rect_num(5) == 30\nassert find_rect_num(6) == 42\n\n### Response:']
    ]

    for idx, prompt in enumerate(prompts):
        logger.info(f'==============================Start Case{idx} infer============================')
        prompt = prompt * args.batch_size
        logger.info(f"prompt: {[prompt]}")
        output = model.generate(input_ids=tokenizer(prompt)["input_ids"], use_past=args.use_past,
                                max_length=args.max_length)
        output_decode = tokenizer.decode(output[0])
        logger.info(f"output: {[output_decode]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='run_wizardcoder_15b.yaml', type=str,
                        help='config')
    parser.add_argument('--max_length', default=2048, type=int,
                        help='max length')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='batch_size')
    parser.add_argument('--device_id', default=0, type=int,
                        help='device_id')
    parser.add_argument('--use_past', default=True, type=str2bool,
                        help="use past")
    args_ = parser.parse_args()

    main(args_)
