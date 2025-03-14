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

"""LLM Boost task script"""

import argparse
import os
import numpy as np

from mindspore.communication import get_rank
from mindspore._c_expression import _framework_profiler_step_start, _framework_profiler_step_end
from mindformers import MindFormerConfig, logger
from mindformers.models.llama.llama_tokenizer import LlamaTokenizer
from mindformers.core.context import build_context
from mindformers.trainer.utils import load_ckpt
from mindformers.tools import get_output_root_path
from mindformers.tools.utils import str2bool
from research.llm_boost.llm_boost import LlmBoostForCausalLM
from research.llm_boost.llm_boost import LlmBoostConfig
from research.qwen2.qwen2_tokenizer import Qwen2Tokenizer
from research.llm_boost.utils import generate_state_dict, save_strategy_file


tokenizer_dict = {"LlamaTokenizer": LlamaTokenizer, "Qwen2Tokenizer": Qwen2Tokenizer}


# pylint: disable=C0330
def main(
    config_path=None,
    use_parallel=None,
    load_checkpoint=None,
    only_save_strategy=None,
    vocab_file=None,
    merges_file=None,
    seq_length=None,
    batch_size=None,
    max_decode_length=8192,
    device_num=None,
    measure_throughput=False,
    predict_data="",
    save_file=""
):
    """main function."""


    if seq_length and seq_length < max_decode_length:
        raise ValueError("The max_decode_length must be less than or equal to seq_length.")

    yaml_path = os.path.expanduser(config_path)
    if not os.path.exists(yaml_path):
        raise ValueError("The yaml_path should exist.")

    config = MindFormerConfig(os.path.realpath(yaml_path))
    if vocab_file:
        if not os.path.exists(vocab_file):
            raise ValueError("The vocab_file should exist.")
        config.processor.tokenizer.vocab_file = vocab_file
    if merges_file:
        tokenizer = tokenizer_dict[config.processor.tokenizer.type](vocab_file, merges_file)
        config.processor.tokenizer.merges_file = merges_file
    else:
        tokenizer = tokenizer_dict[config.processor.tokenizer.type].from_pretrained(vocab_file)

    if use_parallel is not None:
        config.use_parallel = use_parallel
    if only_save_strategy is not None:
        config.only_save_strategy = only_save_strategy
    if device_num is not None:
        config.parallel_config.model_parallel = device_num
    # init context
    build_context(config)

    os.environ['MS_DISABLE_REF_MODE'] = '1'
    if load_checkpoint is not None:
        config.load_checkpoint = load_checkpoint
    if seq_length is not None:
        config.model.model_config.seq_length = seq_length
    if batch_size is not None:
        config.model.model_config.batch_size = batch_size
    config.model.model_config.parallel_config = config.parallel_config
    model_config = LlmBoostConfig(**config.model.model_config)
    model_config.checkpoint_name_or_path = None

    # build model
    network = LlmBoostForCausalLM(model_config)

    # get strategy file
    if model_config.llm_backend == "BuildIn" and config.only_save_strategy:
        strategy_ckpt_save_dir = os.path.join(get_output_root_path(), "strategy")
        os.makedirs(strategy_ckpt_save_dir, exist_ok=True)
        strategy_file_path = os.path.join(
            strategy_ckpt_save_dir, "ckpt_strategy_rank_0.ckpt"
        )
        shard_state_dict = generate_state_dict(config)
        if get_rank() == 0:
            save_strategy_file(shard_state_dict, strategy_file_path)
        logger.info(f"Strategy file has been saved in {strategy_file_path}.")
        return

    # load checkpoint
    load_ckpt(config, network)
    # generate
    if not measure_throughput:
        inputs = [predict_data for i in range(config.model.model_config.batch_size)]
        inputs_ids = tokenizer(inputs)["input_ids"]
        outputs = network.generate(
            input_ids=inputs_ids,
            max_length=max_decode_length,
            do_sample=model_config.do_sample,
            top_k=model_config.top_k,
            top_p=model_config.top_p,
        )
        with open(save_file, 'w') as file:
            for output in outputs:
                print(tokenizer.decode(output))
                file.write(tokenizer.decode(output) + '\n')
        file.close()

    else:
        _framework_profiler_step_start()
        for bs in [1, 4]:
            length = 512
            max_length = length * 2

            print("***************************Warm up for bs {}*************************".format(bs))
            inputs_ids_arr = np.random.randint(low=1, high=1000, size=(bs, max_length))
            inputs_ids_arr[:, length:] = 0
            outputs = network.generate(input_ids=inputs_ids_arr.tolist(), max_length=max_length,
                                       do_sample=model_config.do_sample, top_k=model_config.top_k,
                                       top_p=model_config.top_p)

            for length in [256, 512, 1024, 2048]:
                print("************************ Measure bs={}, length={} *************************".format(bs, length))
                max_length = length * 2
                inputs_ids_arr = np.random.randint(low=1, high=1000, size=(bs, length))
                outputs = network.generate(input_ids=inputs_ids_arr.tolist(), max_length=max_length,
                                           do_sample=model_config.do_sample, top_k=model_config.top_k,
                                           top_p=model_config.top_p)
        _framework_profiler_step_end()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        default=None,
        type=str,
        help="config file path.",
    )
    parser.add_argument(
        "--load_checkpoint",
        default=None,
        type=str,
        help="checkpoint name or dir to load.",
    )
    parser.add_argument(
        "--only_save_strategy",
        default=None,
        type=str2bool,
        help="only save strategy.",
    )
    parser.add_argument("--vocab_file", default=None, type=str, help="tokenizer model")
    parser.add_argument("--merges_file", default=None, type=str, help="tokenizer model")
    parser.add_argument("--seq_length", default=None, type=int, help="seq_length")
    parser.add_argument(
        "--max_decode_length",
        default=8192,
        type=int,
        help="max length for predict output.",
    )
    parser.add_argument(
        "--use_parallel", default=None, type=str2bool, help="open parallel for model."
    )
    parser.add_argument(
        "--device_id",
        default=-1,
        type=int,
        help="ID of the target device, the value must be in [0, device_num_per_host-1]",
    )
    parser.add_argument("--do_sample", default=None, type=str2bool, help="do_sample")
    parser.add_argument("--top_k", default=None, type=int, help="top_k")
    parser.add_argument("--top_p", default=None, type=float, help="top_p")
    parser.add_argument(
        "--repetition_penalty", default=1.0, type=float, help="repetition_penalty"
    )
    parser.add_argument("--batch_size", default=None, type=int, help="batch_size")
    parser.add_argument("--device_num", default=None, type=int, help="device_num")
    parser.add_argument("--measure_throughput", default=False, type=str2bool, help="measure_throughput")
    parser.add_argument("--save_file", default="results.txt", type=str, help="file to store results")
    parser.add_argument('--predict_data', default="", type=str, help='input predict data.')

    args = parser.parse_args()
    print(args)

    main(
        config_path=args.config_path,
        use_parallel=args.use_parallel,
        only_save_strategy=args.only_save_strategy,
        load_checkpoint=args.load_checkpoint,
        vocab_file=args.vocab_file,
        merges_file=args.merges_file,
        seq_length=args.seq_length,
        max_decode_length=args.max_decode_length,
        batch_size=args.batch_size,
        device_num=args.device_num,
        measure_throughput=args.measure_throughput,
        predict_data=args.predict_data,
        save_file=args.save_file
    )
