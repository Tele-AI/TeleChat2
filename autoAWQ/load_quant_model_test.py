import os
os.environ['CUDA_VISIBLE_DEVICES']='0' ##修改为自己的device num

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda"  # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "YOUR_QUANTED_PATH",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",

)

tokenizer = AutoTokenizer.from_pretrained("YOUR_QUANTED_PATH", trust_remote_code=True)
text = "酱油是什么？"

with torch.no_grad():
    query_prompt = text
    inputs = tokenizer(query_prompt,add_special_tokens=False)
    output = model.generate(input_ids = torch.tensor([[4] + inputs['input_ids'] + [5]]).cuda(),#7b的方式，前后+4和+5
                        max_new_tokens=2048,
                        do_sample=False,
                        use_cache=True,
                        top_p = None,
                        top_k = None,
                        eos_token_id=[tokenizer.eos_token_id,tokenizer.pad_token_id]
                        )
    output = output[:,len(inputs['input_ids']) + 2:]
    output_str = tokenizer.decode((output[0].tolist()),skip_special_tokens=True)
    print(output_str)
