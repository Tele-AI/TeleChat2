import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

PATH = 'TeleAI/TeleChat2_115B'


def main():
    # 加载模型相关
    tokenizer = AutoTokenizer.from_pretrained(PATH,trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(PATH, trust_remote_code=True, device_map="auto",
                                                 torch_dtype=torch.float16)
    generate_config = GenerationConfig.from_pretrained(PATH)
    model.eval()
    # 输入
    messages = [
        {"role": "user", "content": "你是谁"},
        {"role": "bot", "content": "我是telechat"},
        {"role": "user", "content": "你在干什么"}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt")
    # 推理
    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=generate_config)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__=="__main__":
    main()
