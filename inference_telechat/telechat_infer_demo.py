import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
tokenizer = AutoTokenizer.from_pretrained('TeleChat2/Telechat2-7B', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained('TeleChat2/Telechat2-7B', trust_remote_code=True, device_map="auto",
                                                  torch_dtype=torch.float16)
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
