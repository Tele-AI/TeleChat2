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
"""QwenVL predict task"""
import argparse
import os
import shutil
from typing import Optional, List

from mindspore import Model

from mindformers import MindFormerConfig, MindFormerRegister, MindFormerModuleType
from mindformers.core.context import build_context
from mindformers.models import build_network, build_processor
from mindformers.tools import get_output_root_path
from mindformers.tools.logger import logger
from mindformers.tools.utils import str2bool
from mindformers.trainer.utils import transform_and_load_checkpoint

from qwen.qwen_config import QwenConfig
from qwen.qwen_model import QwenForCausalLM
from qwenvl import QwenVL
from qwenvl_config import QwenVLConfig
from qwenvl_processor import QwenVLImageProcessor
from qwenvl_processor import QwenVLProcessor
from qwenvl_tokenizer import QwenVLTokenizer
from qwenvl_transform import QwenVLTransform


def register_modules():
    MindFormerRegister.register_cls(QwenVL, MindFormerModuleType.MODELS)
    MindFormerRegister.register_cls(QwenForCausalLM, MindFormerModuleType.MODELS)
    MindFormerRegister.register_cls(QwenConfig, MindFormerModuleType.CONFIG)
    MindFormerRegister.register_cls(QwenVLConfig, MindFormerModuleType.CONFIG)
    MindFormerRegister.register_cls(QwenVLTokenizer, MindFormerModuleType.TOKENIZER)
    MindFormerRegister.register_cls(QwenVLTransform, MindFormerModuleType.TRANSFORMS)
    MindFormerRegister.register_cls(QwenVLProcessor, MindFormerModuleType.PROCESSOR)
    MindFormerRegister.register_cls(QwenVLImageProcessor, MindFormerModuleType.PROCESSOR)


def update_sub_model_config(config, model_type, key, new_value):
    if model_type not in ("vision", "llm"):
        raise ValueError(f"model_type {model_type} is not supported.")

    sub_model_config = config.model.model_config.get(f"{model_type}_model").model_config
    if key not in sub_model_config:
        logger.warning(f"The key={key} is not in {model_type} model config, the key will be added.")

    sub_model_config[key] = new_value


def update_model_config(config, key, new_value):
    model_config = config.model.model_config
    if key not in model_config:
        logger.warning(f"The key={key} is not in Qwen-VL model config, the key will be added.")

    model_config[key] = new_value


def update_llm_model_config(config, key, new_value):
    update_sub_model_config(config, "llm", key, new_value)


def update_vision_model_config(config, key, new_value):
    update_sub_model_config(config, "vision", key, new_value)


def get_sub_model_config(config, model_type, key):
    if model_type not in ("vision", "llm"):
        raise ValueError(f"model_type {model_type} is not supported.")

    sub_model_config = config.model.model_config.get(f"{model_type}_model").model_config
    if key not in sub_model_config:
        raise ValueError(f"The key={key} is not in {model_type} model config.")

    return sub_model_config.get(key)


def get_vision_model_config(config, key):
    return get_sub_model_config(config, "vision", key)


def get_llm_model_config(config, key):
    return get_sub_model_config(config, "llm", key)


def check_predict_config(
        config,
        ckpt=None,
        vocab_file=None,
        image_path=None,
        image_size=None,
        seq_length=None,
        use_past=None,
        max_length=512,
        do_sample=None,
        top_k=None,
        top_p=None,
):
    """check configs when running prediction task"""
    if vocab_file is not None:
        if not os.path.exists(vocab_file):
            raise FileExistsError(f"The vocab file {vocab_file} does not exist.")
        config.processor.tokenizer.vocab_file = vocab_file

    if image_path is None:
        raise ValueError("the image_path should be specified when run_mode=predict")

    if seq_length is not None:
        update_llm_model_config(config, "seq_length", seq_length)
        config.processor.max_length = seq_length
    if do_sample is not None:
        update_llm_model_config(config, "do_sample", do_sample)
    if top_k is not None:
        update_llm_model_config(config, "top_k", top_k)
    if top_p is not None:
        update_llm_model_config(config, "top_p", top_p)
    if max_length is not None:
        seq_length = get_llm_model_config(config, "seq_length")
        max_length = max(max_length, seq_length)
        update_llm_model_config(config, "max_decode_length", max_length)

    if ckpt is not None:
        config["load_checkpoint"] = ckpt

    if image_size is None:
        image_size = get_vision_model_config(config, "image_size")
    update_vision_model_config(config, "image_size", image_size)
    config.processor.image_processor.image_size = image_size

    if use_past is not None:
        config.model.model_config.use_past = use_past
        update_llm_model_config(config, "use_past", use_past)


def clear_auto_trans_output(config):
    """clear transformed_checkpoint and strategy"""
    strategy_dir = os.path.join(get_output_root_path(), "strategy")
    if os.path.exists(strategy_dir) and config.local_rank % 8 == 0:
        shutil.rmtree(strategy_dir)
    transformed_ckpt_dir = os.path.join(get_output_root_path(), "transformed_checkpoint")
    if os.path.exists(transformed_ckpt_dir) and config.local_rank % 8 == 0:
        shutil.rmtree(transformed_ckpt_dir)
    os.makedirs(strategy_dir, exist_ok=True)
    os.makedirs(transformed_ckpt_dir, exist_ok=True)


def main(config="finetune_qwenvl_910b.yaml",
         use_parallel=False,
         use_past=None,
         ckpt=None,
         auto_trans_ckpt=None,
         vocab_file=None,
         image_path=Optional[List[str]],
         image_size=None,
         prompt="",
         seq_length=None,
         max_length=512,
         device_id=0,
         batch_size=None,
         do_sample=None,
         top_k=None,
         top_p=None):
    """main function."""
    yaml_path = os.path.realpath(os.path.expanduser(config))
    if not os.path.exists(yaml_path):
        raise FileExistsError(f"The yaml file {yaml_path} does not exist.")

    config = MindFormerConfig(yaml_path)

    if use_parallel is not None:
        config.use_parallel = use_parallel

    if device_id is not None:
        config.context.device_id = device_id

    # init context
    build_context(config)

    if auto_trans_ckpt is not None:
        config.auto_trans_ckpt = auto_trans_ckpt
        if config.auto_trans_ckpt:
            clear_auto_trans_output(config)

    check_predict_config(config, ckpt=ckpt, vocab_file=vocab_file, image_path=image_path, image_size=image_size,
                         seq_length=seq_length, use_past=use_past, max_length=max_length, do_sample=do_sample,
                         top_k=top_k, top_p=top_p)
    network = build_network(config.model)
    model = Model(network)

    if batch_size is None:
        batch_size = 1

    inputs = [
        [{"image": image_path[0]}, {"text": prompt}],
    ] * batch_size

    processor = build_processor(config.processor)
    processed_res = processor(text_input=inputs)
    input_ids = processed_res.get("input_ids")
    input_images = processed_res.get("image")
    img_pos = processed_res.get("img_pos")

    # load checkpoint
    if config.load_checkpoint:
        logger.info("----------------Transform and load checkpoint----------------")
        # set auto transform ckpt
        if os.path.isdir(config.load_checkpoint) or config.use_parallel:
            config.auto_trans_ckpt = True
        else:
            config.auto_trans_ckpt = False

        infer_data = network.prepare_inputs_for_predict_layout(input_ids, images=input_images, img_pos=img_pos)
        transform_and_load_checkpoint(config, model, network, infer_data, do_predict=True)

    for _ in range(3):
        outputs = network.generate(input_ids=input_ids, images=input_images, img_pos=img_pos)

        for index, output_ids in enumerate(outputs):
            result = processor.tokenizer.decode(output_ids, skip_special_tokens=False)
            result = processor.tokenizer.post_process(result, inputs[index])
            print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default="finetune_qwenvl_9.6b.yaml",
                        type=str,
                        help="config file path.")
    parser.add_argument("--load_checkpoint", default=None, type=str,
                        help="checkpoint name or dir to load.")
    parser.add_argument("--auto_trans_ckpt", default=None, type=str2bool,
                        help="whether to transform checkpoint to the checkpoint matching "
                             "current distribute strategy.")
    parser.add_argument("--vocab_file", default=None, type=str, help="tokenizer model")
    parser.add_argument("--image_path", default="", type=str, nargs="*", help="input predict data.")
    parser.add_argument("--image_size", default=None, type=int, help="image size")
    parser.add_argument("--prompt", default="", type=str, help="input predict data.")
    parser.add_argument("--batch_size", default=None, type=int, help="batch_size")
    parser.add_argument("--seq_length", default=None, type=int, help="seq_length")
    parser.add_argument("--predict_length", default=512, type=int, help="max length for predict output.")
    parser.add_argument("--use_parallel", default=False, type=str2bool, help="open parallel for model.")
    parser.add_argument("--optimizer_parallel", default=False, type=str2bool,
                        help="whether use optimizer parallel. Default: False")
    parser.add_argument("--device_id", default=-1, type=int,
                        help="ID of the target device, the value must be in [0, device_num_per_host-1]")
    parser.add_argument("--use_past", default=None, type=str2bool, help="use_past")
    parser.add_argument("--do_sample", default=None, type=str2bool, help="do_sample")
    parser.add_argument("--top_k", default=None, type=int, help="top_k")
    parser.add_argument("--top_p", default=None, type=float, help="top_p")

    args = parser.parse_args()

    if args.device_id == -1:
        args.device_id = int(os.getenv("RANK_ID", "0"))

    register_modules()

    main(config=args.config,
         use_parallel=args.use_parallel,
         use_past=args.use_past,
         ckpt=args.load_checkpoint,
         auto_trans_ckpt=args.auto_trans_ckpt,
         vocab_file=args.vocab_file,
         image_path=args.image_path,
         image_size=args.image_size,
         prompt=args.prompt,
         batch_size=args.batch_size,
         seq_length=args.seq_length,
         max_length=args.predict_length,
         device_id=args.device_id,
         do_sample=args.do_sample,
         top_k=args.top_k,
         top_p=args.top_p)
