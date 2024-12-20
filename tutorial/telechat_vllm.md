# VLLM
我们建议您在部署TeleChat时尝试使用[vLLM](https://github.com/vllm-project/vllm)。它易于使用，且具有最先进的服务吞吐量、高效的注意力键值内存管理（通过PagedAttention实现）、连续批处理输入请求、优化的CUDA内核等功能。要了解更多关于vLLM的信息，请参阅 [论文](https://arxiv.org/abs/2309.06180) 和 [文档](https://vllm.readthedocs.io/)。

## 安装
默认情况下，你可以通过 pip 在新环境中安装vLLM：
```
pip install vllm #需要0.6.5 或以上版本
```


请留意预构建的vllm对torch和其CUDA版本有强依赖。请查看[vLLM官方文档](https://docs.vllm.ai/en/latest/getting_started/installation.html)中的注意事项以获取有关安装的帮助。我们也建议你通过 pip install ray 安装ray， 以便支持分布式服务。

## 离线推理
TeleChat2代码支持的模型都被vLLM所支持。 vLLM最简单的使用方式是通过以下演示进行离线批量推理。
```
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("TeleChat/TeleChat2-7B", trust_remote_code=True)

# Pass the default decoding hyperparameters of TeleChat2-7B
# max_tokens is for the maximum length for generation.
sampling_params = SamplingParams(temperature=0.0, repetition_penalty=1.05, max_tokens=512)

# Input the model name or path.
llm = LLM(model="TeleChat/TeleChat2-7B", trust_remote_code=True)

# Prepare your prompts
prompt = "Tell me something about large language models."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# generate outputs
outputs = llm.generate([text], sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

## OpenAI兼容的API服务
借助vLLM，构建一个与OpenAI API兼容的API服务十分简便，该服务可以作为实现OpenAI API协议的服务器进行部署。默认情况下，它将在 http://localhost:8000 启动服务器。您可以通过 --host 和 --port 参数来自定义地址。请按照以下所示运行命令：
```
vllm serve TeleChat2/TeleChat2-7B
    --trust-remote-code
    --dtype float16
```
你无需担心chat模板，因为它默认会使用由tokenizer提供的chat模板。

然后，您可以利用 [create chat interface](https://platform.openai.com/docs/api-reference/chat/completions/create) 来与TeleChat进行对话：
您可以如下面所示使用 openai Python 包中的 API 客户端：
```
from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="TeleChat/TeleChat2-7B",
    messages=[
        {"role": "user", "content": "Tell me something about large language models."},
    ],
    temperature=0.0,
    max_tokens=512,
    extra_body={
        "repetition_penalty": 1.05,
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
    },
)
print("Chat response:", chat_response)
```

### 工具使用
vLLM中的OpenAI兼容API 可以配置为支持 TeleChat2 的工具调用。详细信息，请参阅[我们关于函数调用的指南]()。

### 结构化/JSON输出
当 TeleChat2 与 vLLM 结合使用时，支持结构化/JSON 输出。请参照[vllm 的文档](https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html#extra-parameters-for-chat-api)了解 guided_json 参数。此外，也建议在系统消息或用户提示中指示模型生成特定格式，避免仅依赖于推理参数配置。

## 多卡分布式部署
要提高模型的处理吞吐量，分布式服务可以通过利用更多的GPU设备来帮助您。特别是对于像 TeleChat-115B 这样的大模型，单个GPU无法支撑其在线服务。在这里，我们通过演示如何仅通过传入参数 tensor_parallel_size ，来使用张量并行来运行 TeleChat-115B 模型：
离线推理
```
from vllm import LLM, SamplingParams
llm = LLM(model="TeleChat/TeleChat2-115B", trust_remote_code=True, tensor_parallel_size=4)
```
API
```
vllm serve TeleChat2/TeleChat2-7B
    --trust-remote-code
    --tensor-parallel-size 4
    --dtype float16
```

## 上下文支持扩展
TeleChat2 模型的上下文长度默认设置为 8192和32768 个token。为了处理超出默认token的大量输入，我们使用了 dynamic，这是一种增强模型长度外推的技术，确保在处理长文本时的最优性能。
vLLM 支持 dynamic，并且可以通过在模型的 config.json 文件中添加一个 rope_scaling 字段来启用它。例如，
```
{
  ...,
  "rope_scaling": {
    "factor": 3.0,
    "rope_type": "dynamic"
  },
}
```
