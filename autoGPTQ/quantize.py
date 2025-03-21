import os
import json
import torch
from transformers import AutoTokenizer
from argparse import ArgumentParser, Namespace
from gptqmodel import GPTQModel, QuantizeConfig


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
    bits = args.bits
    dataset = args.dataset
    output_dir = args.output_dir
    trust_remote_code = args.trust_remote_code

    calib_dataset = []
    with open(dataset, "rt", encoding="utf-8") as f:
        for line in f.readlines():
            # Assuming each row is a piece of calibration data
            calib_dataset.append(line.strip())

    qa_config = QuantizeConfig(bits=bits, group_size=128)
    model = GPTQModel.load(
        model_dir,
        qa_config,
        torch_dtype=getattr(torch, dtype),
        trust_remote_code=trust_remote_code,
    )

    model.quantize(calib_dataset, batch_size=2)

    # Save model and config
    model.save(output_dir)

    # In order to use VLLM inference, we need to update config.json
    quantize_config = model.quantize_config.to_dict()
    quantize_config.update({"modules_to_not_convert": ["qkv_proj"]})
    override_fields = {
        "architectures": [
            "TeleChat2ForCausalLM",
        ],
        "num_key_value_heads": model.config.n_head,
        "quantization_config": quantize_config,
    }
    update_config_file(os.path.join(output_dir, "config.json"), **override_fields)
    update_config_file(
        os.path.join(output_dir, "quantize_config.json"),
        modules_to_not_convert=["qkv_proj"],
    )


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Quantifying the Telechat Model Using the GPTQ Algorithm"
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
        "--bits",
        default=4,
        type=int,
        choices=[2, 3, 4, 8],
        help="Quant bits used by model",
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
