import os
import json
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

def is_safetensor(model_dir):
    """
    Check if the model weights are safe tensors

    Args:
        model_dir (str): directory to model path。

    Returns:
        bool: True if safetensor else False
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
   

model_path = '/home/hf-models/TeleChat2-7B/'
quant_path = './telechat2-awq'

quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path,device_map={"":"cpu"},
                                           **{"low_cpu_mem_usage": True, "use_cache": False},
                                           safetensors=is_safetensor(model_path))
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# In order to run the quantized model using vllm, we need to update the following fields in the config.json file
setattr(model.quant_config, "modules_to_not_convert", ["qkv_proj", "o_proj"])

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

# In order to run the quantized model using vllm, we need to update the following fields in the config.json file
# Note that the vllm version needs to be greater than 0.6.5
override_fields = {
    "architectures": ["TeleChat2ForCausalLM", ],
    "num_key_value_heads": model.config.n_head
}

json_file = os.path.join(quant_path,"config.json")
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)
for key, value in override_fields.items():
    if key == "num_key_value_heads" and key in data:
        continue
    data.update({key: value})
with open(json_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, sort_keys=True)

print(f'Model is quantized and saved at "{quant_path}"')