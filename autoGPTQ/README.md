# GPTQ
为了降低显存消耗并提升推理速度，可以使用 GPTQ 算法将 TeleChat 模型的权重量化为  `8-bit`甚至`2-bit` 。本文档将详细介绍该过程，并利用 vLLM 库进行推理。  

本文档以`TeleChat2-7B-32K`为例，下面是模型的两个下载链接，假设模型已下载到本地目录 **/workspace/TeleChat2-7B**

- huggingface [Tele-AI/TeleChat2-7B-32K](https://huggingface.co/Tele-AI/TeleChat2-7B-32K)
- 魔塔社区 [Tele-AI/TeleChat2-7B-32K](https://www.modelscope.cn/models/TeleAI/TeleChat2-7B-32K/summary)

### 1. 依赖安装

``` shell
pip install gptqmodel --no-build-isolation
pip install transformers
pip install accelerate==0.34.2
pip install flash-attn==2.6.3
```
如果需要使用VLLM推理，还需要安装vllm

```shell
pip install vllm==0.6.5
```
### 2. 量化模型

要对 **TeleChat2-7B-32K** 模型进行量化，需要准备一个 **校准数据集**，格式为 **TXT 文件**，其中每行代表一条校准数据。然后运行当前目录下的 `quantize.py` 脚本。
```python
python3 quantize.py --trust_remote_code /workspace/TeleChat2-7B /path/to/calib_dataset.txt /workspace/TeleChat2-7B-GPTQ
```

### 3. 更新vllm的代码
telechat中的某些层无法使用gptq量化，但是目前vllm中关于gptq部分的代码不支持部分层量化，所以需要对vllm中的代码进行更改，此文档使用的vllm版本是v0.6.5，其他版本请根据样例修改代码。  

😃 **确定vllm版本等于0.6.5**  
在当前目录执行以下命令更新vllm已安装的代码

```shell
bash apply_path.sh
```
如果输出success，说明执行成功  

### 4. 使用vllm库来推理量化后的模型
```shell
vllm serve /workspace/TeleChat2-7B-GPTQ --served-model-name telechat-gptq --dtype float16 --max_model_len 4096 --trust_remote_code 
```
等模型启动后，可以使用curl命令发送请求
```shell
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "telechat-gptq",
  "messages": [
    {"role": "user", "content": "生抽与老抽的区别？"}
  ],
  "temperature": 1.0,
  "top_p": 1.0,
  "repetition_penalty": 1.03,
  "max_tokens": 512
}'
```
或者，你也可以使用Openai提供的api来发送请求
```python
from openai import OpenAI

openai_api_key = "xxx"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

resp = client.chat.completions.create(
    model="telechat-gptq",
    messages=[
        {"role": "user", "content": "生抽与老抽的区别？"}
    ],
    temperature=1.0,
    top_p=1.0,
    max_tokens=512,
    extra_body={
        "repetition_penalty": 1.03,
    },
)
print("response:", resp)
```