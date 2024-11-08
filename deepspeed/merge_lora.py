# encoding=utf-8
# This code is based on the work of yangjianxin1/Firefly and has been modified from its original version.
# see https://github.com/yangjianxin1/Firefly 

# 使用本脚本，将训练好的 Lora 权重与 Base 模型合并。

import os
import torch
import shutil
import argparse

from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# ================
#    Settings
# ================
BASE_MODEL_NAME_OR_PATH = "Path/to/your/base_model"
ADAPTER_PATH = "Path/to/your/trained_lora_model"
SAVE_PATH = "Path/to/save/your/merged_model"

# ================
#    functions
# ================
def parser_args():
    parser = argparse.ArgumentParser(
        description="script for merge and save lora model."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=BASE_MODEL_NAME_OR_PATH,
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=ADAPTER_PATH,
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=SAVE_PATH,
    )
    args = parser.parse_args()
    return args


def copy_code(
    model_name_or_path,
    save_path
):
    # copy .py files
    os.makedirs(save_path, exist_ok=True)
    if os.path.exists(model_name_or_path):
        remote_codes = [
            file for file in os.listdir(model_name_or_path) \
                if file.endswith(".py")
        ]
        for filename in remote_codes:
            if not os.path.exists(
                os.path.join(save_path, filename)
            ):
                shutil.copy(
                    os.path.join(model_name_or_path, filename),
                    os.path.join(save_path, filename)
                )


def merge_lora_to_model(args):
    # 合并权重，并保存
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="cpu"
    )
    model = PeftModel.from_pretrained(
        model, 
        args.adapter_path,
        device_map="cpu"
    )
    model = model.merge_and_unload()
    
    # 保存模型
    copy_code(args.model_name_or_path, args.save_path)
    tokenizer.save_pretrained(args.save_path)
    model.save_pretrained(args.save_path)


# ================
#      main
# ================
if __name__ == "__main__":
    args = parser_args()
    merge_lora_to_model(args)