import os
import json
import glob
import shutil
import torch
from transformers import AutoTokenizer
from argparse import ArgumentParser, Namespace
from awq_patch import Telechat2AWQForCausalLM, Telechat2AwqQuantizer


def is_safetensor(model_dir):
    """
    Check if the weight files in the model directory are in SafeTensor format.

    Args:
        model_dir (str): Model directory

    Returns:
        bool: If there is a safetensor file, return True; otherwise, return False.
    """
    try:
        files = os.listdir(model_dir)
        for file in files:
            if file.endswith(".safetensors"):
                return True
        return False
    except Exception as e:
        print(f"Error checking safetensor: {e}")
        return False


def update_config_file(conig_path, **kwargs):
    """
    In order to use VLLM inference, a portion of the config. json file is required

    Args:
        conig_path: The path of config.json
        kwargs: Properties that need to be updated
    """
    with open(conig_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    for key, value in kwargs.items():
        if key == "num_key_value_heads" and key in config:
            continue
        config.update({key: value})
    with open(conig_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)


def main(args: Namespace):
    model_dir = args.model_dir
    dtype = args.dtype
    dataset = args.dataset
    output_dir = args.output_dir
    clip = args.clip
    trust_remote_code = args.trust_remote_code

    calib_dataset = []
    with open(dataset, "rt", encoding="utf-8") as f:
        for line in f.readlines():
            # Assuming each row is a piece of calibration data
            calib_dataset.append(line.strip())

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=trust_remote_code
    )
    model = Telechat2AWQForCausalLM.from_pretrained(
        model_path=model_dir,
        model_type="telechat",
        torch_dtype=getattr(torch, dtype),
        safetensors=is_safetensor(model_dir),
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
    )
    qa_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
    }
    model.quantize(
        tokenizer=tokenizer,
        quant_config=qa_config,
        calib_data=calib_dataset,
        apply_clip=clip,
        quantizer_cls=Telechat2AwqQuantizer,
    )
    # In order to use VLLM inference, the following content needs to be updated
    setattr(model.quant_config, "modules_to_not_convert", ["qkv_proj", "o_proj"])

    # Save model and config
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)

    # In order to use VLLM inference, we need to update config.json
    override_fields = {
        "architectures": [
            "TeleChat2ForCausalLM",
        ],
        "num_key_value_heads": model.config.n_head,
    }
    update_config_file(os.path.join(output_dir, "config.json"), **override_fields)

    # cp modeling_telechat2.py to output_dir
    py_files = glob.glob(os.path.join(model_dir, "modeling_telechat*.py"))
    for file in py_files:
        shutil.copy(file, output_dir)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Quantifying the Telechat Model Using the AWQ Algorithm"
    )
    parser.add_argument(
        "--clip",
        action="store_true",
        default=False,
        help="Perform clip operation on the quantified weights",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=False,
        help="Trust remote code when use transformers",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        help="Data type used by model",
    )
    parser.add_argument(
        "model_dir",
        type=str,
        help="Telechar model dir",
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Calibration dataset, in text format, with each line representing one calibration data",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Quantify model output directory",
    )
    main(parser.parse_args())
