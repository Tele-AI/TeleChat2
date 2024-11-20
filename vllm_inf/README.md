# TeleChat vllm 推理使用方式

## 下载vllm
```shell
pip install vllm   #推荐 0.6.1.post2 版本
```

## vllm 添加telechat

### 将telechat model文件放入 vllm 库中
pip show vllm 找到vllm对应位置并进入
```shell
cd ./vllm/model_executor/models/
```
将此路径下的 `telechat.py` 文件放入以上路径

### 修改init文件
修改同路径下的__init__.py
```python
"StableLmForCausalLM": ("stablelm", "StablelmForCausalLM"),
"Starcoder2ForCausalLM": ("starcoder2", "Starcoder2ForCausalLM"),
"TeleChatForCausalLM": ("telechat", "TeleChatForCausalLM"),  #telechat
"ArcticForCausalLM": ("arctic", "ArcticForCausalLM"),
"XverseForCausalLM": ("xverse", "XverseForCausalLM"),
```
添加以上代码中的TeleChat 一行

### 修改模型文件里的config.json
```python
architectures": ["TeleChatForCausalLM"]
```

## 外推
==注：文本长度低于8k时，请使用通用推理方式，不要进行以下修改，否则可能导致精度下降。仅当文本长度在8k-16k范围内时，适用该修改。==
如有外推需求在模型文件中的config.json 中添加
```python
"rope_scaling": {
     "factor": 2.0,
     "original_max_position_embeddings": 8192,
     "rope_type": "dynamic"
 }
```


## 启动
按vllm里的方式启动telechat 推理

#### 示例
```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("TeleAI/TeleChat2-7B/",trust_remote_code=True)
llm = LLM(model="TeleAI/TeleChat2-7B/", trust_remote_code=True, gpu_memory_utilization=0.78,max_model_len=1000)
sampling_params = SamplingParams(max_tokens=1000, temperature=0.3, repetition_penalty=1.02) #推荐repetition_penalty为1.03

prompt = "你好"

messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)

outputs = llm.generate([text], sampling_params)
for output in outputs:
    generated_text = output.outputs[0].text
    print(generated_text)
```
