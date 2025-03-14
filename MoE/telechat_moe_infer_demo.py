import os 
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

PATH = ""
tokenizer = AutoTokenizer.from_pretrained(PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(PATH, trust_remote_code=True, device_map="auto", torch_dtype=torch.float32)

prompt = "高音单簧管跟高音萨克斯的调性相同吗？如果相同，请说出他们的调性，如果不同，请分别说出他们的调性。"
messages = [{"role": user, "content": prompt}]

text = tokenizer.apply_chat_template(messages, tokenizer=False, add_generation_prompt=True)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
generated_ids = model.generate(**model_inputs, 
                              max_new_tokens=4096, 
                              repetition_penalty=1.0,
                              do_sample=Falese)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(respons)