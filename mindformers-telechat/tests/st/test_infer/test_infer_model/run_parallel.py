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
"""
Test module for testing the paralleled infer interface used for mindformers.
How to run this:
    pytest tests/st/test_infer/test_infer_model/run_parallel.py
"""
import argparse
import os

import mindspore as ms
from mindspore import Tensor, Model
from mindspore.common import initializer as init

from mindformers import build_context, MindFormerConfig, build_parallel_config, LlamaConfig, \
    LlamaForCausalLM, Trainer
from mindformers.tools.logger import logger
from mindformers.trainer.utils import transform_and_load_checkpoint
from research.qwen2.qwen2_tokenizer import Qwen2Tokenizer


def parallel_qwen2_0_5b_predict_mp2():
    """test qwen2-0.5B predict in model_parallel=2 with dynamic shape"""
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(cur_dir, "ci_predict_qwen2_0_5b_instruct.yaml")

    vocab_file_path = "/home/workspace/mindspore_dataset/weight/Qwen2.5-0.5B-Instruct-tokenizer/vocab.json"
    merges_file_path = "/home/workspace/mindspore_dataset/weight/Qwen2.5-0.5B-Instruct-tokenizer/merges.txt"
    load_checkpoint = "/home/workspace/mindspore_dataset/weight/ms_safetensor_qwen2_0.5/"

    seq_length = 128
    # init config with yaml
    config = MindFormerConfig(config_path)
    config.use_parallel = True
    config.output_dir = './qwen2_05b_dynamic_output'
    config.parallel.strategy_ckpt_save_file = "./qwen2_05b_dynamic_ckpt_strategy.ckpt"
    config.load_ckpt_format = "safetensors"
    config.parallel_config.model_parallel = 2
    config.parallel_config.data_parallel = 1
    config.parallel_config.pipeline_stage = 1
    config.load_checkpoint = load_checkpoint
    config.model.model_config.seq_length = seq_length
    config.processor.tokenizer.vocab_file = vocab_file_path
    config.processor.tokenizer.merges_file = merges_file_path

    # init context
    build_context(config)
    build_parallel_config(config)

    config.model.model_config.parallel_config = config.parallel_config
    model_config = LlamaConfig(**config.model.model_config)
    model_config.checkpoint_name_or_path = None

    # build tokenizer
    tokenizer = Qwen2Tokenizer(**config.processor.tokenizer)

    # build model
    network = LlamaForCausalLM(model_config)
    model = Model(network)

    # load checkpoint
    if config.load_checkpoint:
        logger.info("----------------Transform and load checkpoint----------------")
        batch_size = config.model.model_config.batch_size
        input_ids = Tensor(shape=(batch_size, seq_length), dtype=ms.int32, init=init.One())
        infer_data = network.prepare_inputs_for_predict_layout(input_ids)
        transform_and_load_checkpoint(config, model, network, infer_data, do_predict=True)

    # predict
    batch_datas = {1: {"prompt": "你好！",
                       "answer": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n你好！"
                                 "<|im_end|>\n<|im_start|>assistant\n你好！有什么可以帮助你的吗？<|im_end|>"},
                   4: {"prompt": "用python编写快速排序",
                       "answer": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n用"
                                 "python编写快速排序<|im_end|>\n<|im_start|>assistant\n以下是一个使用Python实现的快速排序"
                                 "算法：\n\n```python\ndef quick_sort(arr):\n    if len(arr) <= 1:\n        "
                                 "return arr\n    else:\n        pivot = arr[0]\n        left = [x for x in arr[1:] "
                                 "if x < pivot]\n        right = [x for x in arr[1:] if x >= pivot]\n        "
                                 "return quick_sort(left) + [pivot] + quick_sort(right)\n\n# 示例输入\narr = [3,6,8,1"},
                   8: {"prompt": "I believe the meaning of life is",
                       "answer": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nI "
                                 "believe the meaning of life is<|im_end|>\n<|im_start|>assistant\nThe meaning of "
                                 "life is a philosophical question that has been debated for centuries, and there "
                                 "is no one definitive answer to it. Some people believe that the meaning of life "
                                 "is to find happiness and fulfillment in their lives, while others believe that it "
                                 "is to achieve success or recognition.\n\nOthers may argue that the meaning of life "
                                 "is to make a positive impact on the world, to help others, and to contribute to "
                                 "society as a whole. Others may believe that the meaning of life is to pursue "
                                 "knowledge and understanding, to"},
                   }
    for batch_size, batch_data in batch_datas.items():
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": batch_data["prompt"]}
        ]
        input_ids = tokenizer.encode(tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        ))
        input_ids_list = []
        answer = batch_data["answer"]
        for i in range(0, batch_size):
            input_ids_list.append(input_ids)
        outputs = network.generate(input_ids_list,
                                   max_length=seq_length,
                                   do_sample=False,
                                   return_dict_in_generate=False)

        for i in range(0, len(outputs)):
            output_text = tokenizer.decode(outputs[i])
            print("parallel_qwen2_0_5b_predict_mp2, output_text:", output_text)
            assert output_text == answer


def parallel_glm3_6b_predict_mp2():
    """test glm3-6B predict in model_parallel=2 with static shape"""
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(cur_dir, "ci_predict_glm3_6b.yaml")

    vocab_file_path = "/home/workspace/mindspore_dataset/weight/chatglm3-6b-tokenizer/tokenizer.model"
    load_checkpoint = "/home/workspace/mindspore_dataset/weight/glm3_6b_ckpt_2cards/"

    seq_length = 128
    # init config with yaml
    config = MindFormerConfig(config_path)
    config.use_parallel = True
    config.parallel.strategy_ckpt_save_file = "./glm3_6b_ckpt_strategy.ckpt"
    config.parallel_config.model_parallel = 2
    config.parallel_config.data_parallel = 1
    config.parallel_config.pipeline_stage = 1
    config.parallel_config.vocab_emb_dp = False
    config.load_checkpoint = load_checkpoint
    config.model.model_config.seq_length = seq_length
    config.processor.tokenizer.vocab_file = vocab_file_path

    # init context
    build_context(config)
    task_trainer = Trainer(config)

    batch_datas = {1: {"prompt": "你好!",
                       "answer": "[gMASK]sop 你好!我是人工智能助手。很高兴能为你提供帮助。请问有什么问题我可以帮你解答吗?"},
                   4: {"prompt": "用python编写快速排序",
                       "answer": "[gMASK]sop 用python编写快速排序算法\n\n 快速排序是一种高效的排序算法，其基本思想是通过"
                                 "一趟排序将待排序的数据分割成独立的两部分，其中一部分的所有数据都比另一部分的所有数据要小，"
                                 "然后再按此方法对这两部分数据分别进行快速排序，整个排序过程可以递归进行，以此达到整个数据"
                                 "变成有序序列。\n\n下面是使用 Python 编写快速排序算法的示例代码：\n\n```python\ndef "
                                 "quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr["},
                   8: {"prompt": "I believe the meaning of life is",
                       "answer": "[gMASK]sop I believe the meaning of life is to seek knowledge and understanding. "
                                 "What are some specific ways to achieve this?\nThere are many ways to achieve the goal"
                                 " of seeking knowledge and understanding. Here are a few specific suggestions:\n\n1. "
                                 "Read widely: One of the best ways to gain knowledge and understanding is to read "
                                 "widely in a variety of subjects. This can help you develop a broad range of knowledge"
                                 " and perspectives.\n2. Ask questions: Seeking knowledge and understanding often "
                                 "involves asking questions. Don't be afraid to ask questions of others, and be sure "
                                 "to ask yourself questions as well.\n3. Engage"},
                   }

    for batch_size, batch_data in batch_datas.items():
        answer = batch_data["answer"]
        input_data = batch_data["prompt"]
        input_data_list = []
        for i in range(0, batch_size):
            input_data_list.append(input_data)
        outputs = task_trainer.predict(predict_checkpoint=config.load_checkpoint,
                                       input_data=input_data_list,
                                       max_length=seq_length,
                                       batch_size=batch_size)
        for i in range(0, len(outputs)):
            output_text = outputs[i]
            print("parallel_glm3_6b_predict_mp2, output_text:", output_text['text_generation_text'][0])
            assert output_text['text_generation_text'][0] == answer


def parallel_qwen_moe_predict_mp2():
    """test qwen2-0.5B predict in model_parallel=2 with dynamic shape"""
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(cur_dir, "ci_predict_qwen2_57b_a14b_instruct.yaml")

    vocab_file_path = "/home/workspace/mindspore_dataset/weight/Qwen2.5-0.5B-Instruct-tokenizer/vocab.json"
    merges_file_path = "/home/workspace/mindspore_dataset/weight/Qwen2.5-0.5B-Instruct-tokenizer/merges.txt"

    ms.set_seed(123)

    seq_length = 128
    # init config with yaml
    config = MindFormerConfig(config_path)
    config.use_parallel = True
    config.parallel.strategy_ckpt_save_file = "./qwen_moe_strategy.ckpt"
    config.parallel_config.model_parallel = 2
    config.parallel_config.data_parallel = 1
    config.parallel_config.pipeline_stage = 1
    config.model.model_config.num_layers = 2
    config.model.model_config.seq_length = seq_length
    config.processor.tokenizer.vocab_file = vocab_file_path
    config.processor.tokenizer.merges_file = merges_file_path

    # init context
    build_context(config)
    build_parallel_config(config)

    config.model.model_config.parallel_config = config.parallel_config
    model_config = LlamaConfig(**config.model.model_config)
    model_config.checkpoint_name_or_path = None

    # build tokenizer
    tokenizer = Qwen2Tokenizer(**config.processor.tokenizer)

    # build model
    network = LlamaForCausalLM(model_config)
    model = Model(network)

    # load checkpoint
    if config.load_checkpoint:
        logger.info("----------------Transform and load checkpoint----------------")
        batch_size = config.model.model_config.batch_size
        input_ids = Tensor(shape=(batch_size, seq_length), dtype=ms.int32, init=init.One())
        infer_data = network.prepare_inputs_for_predict_layout(input_ids)
        transform_and_load_checkpoint(config, model, network, infer_data, do_predict=True)

    # predict
    batch_datas = {1: {"prompt": "你好！",
                       "answer": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n你好！"
                                 "<|im_end|>\n<|im_start|>assistant\n静脉静脉静脉静脉静脉静脉静脉静脉静脉静脉静脉静脉静脉"
                                 "静脉静脉静脉静脉静脉axteraxteraxteraxteraxteraxteraxteraxteraxteraxteraxteraxteraxte"
                                 "raxteraxter yet yet yet yet yet yet yet yet yet yet yet yet yet yet yet yet yet yet"
                                 " yet yet yet yet yet yet yet yet yet yet yet yet yet yet yet yet yet yet yet yet yet"
                                 " yet yet yet yet yet yet yet yet yet yet yet yet yet yet yet yet yet yet yet yet yet"
                                 " yet yet yet yet yet yet yet yet yet yet yet yet yet yet"},
                   4: {"prompt": "用python编写快速排序",
                       "answer": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n用"
                                 "python编写快速排序<|im_end|>\n<|im_start|>assistant\n_time_time_time_time_time_time_"
                                 "time_time_time��������������������������������������"
                                 "������ sourced sourced sourced sourced sourced sourced sourced sourced sourced "
                                 "sourced sourced sourced sourced sourced sourced sourced sourced sourced sourced "
                                 "sourced sourced sourced sourced sourced sourced sourced sourced sourced sourced "
                                 "sourced sourced sourced sourced sourced sourced sourced sourced sourced sourced "
                                 "sourced sourced sourced sourced sourced sourced sourced sourced sourced sourced "
                                 "sourced sourced"},
                   8: {"prompt": "I believe the meaning of life is",
                       "answer": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nI "
                                 "believe the meaning of life is<|im_end|>\n<|im_start|>assistant\n.origin.origin."
                                 "origin.origin.origin.origin.origin.origin.origin.origin.origin.origin.origin.origin"
                                 ".origin.origin.origin.origin.origin.origin.origin.origin理事理事理事理事理事理事理事理"
                                 "事理事理事理事理事理事理事理事理事理事理事.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail"
                                 ".Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail"
                                 ".Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail"
                                 ".Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail.Fail"
                                 ".Fail.Fail"},
                   }
    for batch_size, batch_data in batch_datas.items():
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": batch_data["prompt"]}
        ]
        input_ids = tokenizer.encode(tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        ))
        input_ids_list = []
        answer = batch_data["answer"]
        for i in range(0, batch_size):
            input_ids_list.append(input_ids)
        outputs = network.generate(input_ids_list,
                                   max_length=seq_length,
                                   do_sample=False,
                                   return_dict_in_generate=False)

        for i in range(0, len(outputs)):
            output_text = tokenizer.decode(outputs[i])
            print("parallel_qwen2_0_5b_predict_mp2, output_text:", output_text)
            assert output_text == answer


def parallel_qwen2_0_5b_predict_mp2_static():
    """test qwen2-0.5B predict in model_parallel=2 with static shape"""
    # config.model.model_config.is_dynamic = False
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(cur_dir, "ci_predict_qwen2_0_5b_instruct.yaml")

    vocab_file_path = "/home/workspace/mindspore_dataset/weight/Qwen2.5-0.5B-Instruct-tokenizer/vocab.json"
    merges_file_path = "/home/workspace/mindspore_dataset/weight/Qwen2.5-0.5B-Instruct-tokenizer/merges.txt"
    load_checkpoint = "/home/workspace/mindspore_dataset/weight/ms_safetensor_qwen2_0.5/"

    seq_length = 128
    # init config with yaml
    config = MindFormerConfig(config_path)
    config.use_parallel = True
    config.output_dir = './qwen2_05b_static_output'
    config.parallel.strategy_ckpt_save_file = "./qwen2_05b_static_ckpt_strategy.ckpt"
    config.load_ckpt_format = "safetensors"
    config.parallel_config.model_parallel = 2
    config.parallel_config.data_parallel = 1
    config.parallel_config.pipeline_stage = 1
    config.load_checkpoint = load_checkpoint
    config.model.model_config.seq_length = seq_length
    config.model.model_config.is_dynamic = False
    config.model.model_config.do_sample = False
    config.model.model_config.temperature = 1.0
    config.processor.tokenizer.vocab_file = vocab_file_path
    config.processor.tokenizer.merges_file = merges_file_path

    # init context
    build_context(config)
    build_parallel_config(config)

    config.model.model_config.parallel_config = config.parallel_config
    model_config = LlamaConfig(**config.model.model_config)
    model_config.checkpoint_name_or_path = None

    # build tokenizer
    tokenizer = Qwen2Tokenizer(**config.processor.tokenizer)

    # build model
    network = LlamaForCausalLM(model_config)
    model = Model(network)

    # load checkpoint
    if config.load_checkpoint:
        logger.info("----------------Transform and load checkpoint----------------")
        batch_size = config.model.model_config.batch_size
        input_ids = Tensor(shape=(batch_size, seq_length), dtype=ms.int32, init=init.One())
        infer_data = network.prepare_inputs_for_predict_layout(input_ids)
        transform_and_load_checkpoint(config, model, network, infer_data, do_predict=True)

    # predict
    batch_datas = {4: {"prompt": "你好!",
                       "answer": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
                                 "你好!<|im_end|>\n<|im_start|>assistant\n你好！很高兴为你提供帮助。"
                                 "有什么我可以帮助你的吗？<|im_end|>"},
                   }
    for batch_size, batch_data in batch_datas.items():
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": batch_data["prompt"]}
        ]
        input_ids = tokenizer.encode(tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        ))
        input_ids_list = []
        answer = batch_data["answer"]
        for i in range(0, batch_size):
            input_ids_list.append(input_ids)
        outputs = network.generate(input_ids_list,
                                   max_length=seq_length,
                                   do_sample=False,
                                   return_dict_in_generate=False)
        for i in range(0, len(outputs)):
            output_text = tokenizer.decode(outputs[i])
            print("parallel_qwen2_0_5b_predict_mp2_static, output_text:", output_text)
            assert output_text == answer


TEST_MAP = {
    'parallel_qwen2_0_5b_predict_mp2': parallel_qwen2_0_5b_predict_mp2,
    'parallel_glm3_6b_predict_mp2': parallel_glm3_6b_predict_mp2,
    'parallel_qwen_moe_predict_mp2': parallel_qwen_moe_predict_mp2,
    'parallel_qwen2_0_5b_predict_mp2_static': parallel_qwen2_0_5b_predict_mp2_static,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='test mode of llama2 model.')

    args = parser.parse_args()
    TEST_MAP[args.mode]()
