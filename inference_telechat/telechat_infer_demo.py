import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers import AutoConfig

config = AutoConfig.from_pretrained("TeleChat2/Telechat2-7B")
config_dict = config.to_dict()

if 'torch_dtype' in config_dict:
    dtype_str = config_dict['torch_dtype']
    # 假设config里是"float16"或"float32"这样的字符串
    if hasattr(torch, dtype_str):
        model_dtype = getattr(torch, dtype_str)
    else:
        raise ValueError(f"torch does not have dtype {dtype_str}")
else:
    # 如果没有定义torch_dtype，定义一个默认值
    model_dtype = torch.float16
  
tokenizer = AutoTokenizer.from_pretrained("TeleChat2/Telechat2-7B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "TeleChat2/Telechat2-7B", 
    trust_remote_code=True, 
    device_map="auto",
    torch_dtype=model_dtype
)

prompt = "生抽与老抽的区别？"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
