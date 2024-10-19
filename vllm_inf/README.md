# TeleChat vllm 推理使用方式

## 下载vllm
```
pip install vllm   #推荐 0.6.1.post2 版本
```

## vllm 添加telechat

### 将telechat model文件放入
pip show vllm 找到vllm对应位置并进入
```
cd ./vllm/model_executor/models/
```
将此路径下的telechat.py 文件放入以上路径

### 修改init文件
修改同路径下的__init__.py
```
    "StableLmForCausalLM": ("stablelm", "StablelmForCausalLM"),
    "Starcoder2ForCausalLM": ("starcoder2", "Starcoder2ForCausalLM"),
    "TeleChatForCausalLM": ("telechat", "TeleChatForCausalLM"),  #telechat
    "ArcticForCausalLM": ("arctic", "ArcticForCausalLM"),
    "XverseForCausalLM": ("xverse", "XverseForCausalLM"),
```
添加以上代码中的TeleChat 一行

### 修改模型文件里的config.json
```
>>> architectures": [
>>>     "TeleChatForCausalLM"
>>>     ]
```

## 外推
如有外推需求在模型文件中的config.json 中添加
```
>>> "rope_scaling": {
>>>     "factor": 4.0,
>>>     "original_max_position_embeddings": 16384,
>>>     "rope_type": "dynamic"
>>> }
```
在configuration_telechat.py中添加 rope_scaling 参数
在def __init__() 入参中中添加
```
>>> rope_scaling=None
```
在def __init__() 方法中添加
```
self.rope_scaling = rope_scaling
```

如下代码所示在configuration_telechat2.py文件中
```
    def __init__(
        self,
        vocab_size=160256,
        hidden_size=4096,
        n_layer=30,
        n_head=32,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        bos_token_id=1,
        eos_token_id=2,
        apply_residual_connection_post_layernorm=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        ffn_hidden_size=12288,
        training_seqlen = 8192,
        logn = True,
        embed_layernorm = False,
        rope_scaling=None, #外推添加
        **kwargs,
    ):
        self.vocab_size = vocab_size
        n_embed = kwargs.pop("n_embed", None)
        self.hidden_size = hidden_size if n_embed is None else n_embed
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.logn = logn
        self.ffn_hidden_size = ffn_hidden_size
        self.training_seqlen = training_seqlen
        self.embed_layernorm = embed_layernorm
        self.rope_scaling = rope_scaling   #外推添加
        self.num_key_value_heads= kwargs.pop("num_key_value_heads", None)
```
 外推建议用vllm服务推理

## 启动
按vllm里的方式启动telechat 推理

#### 示例
```
>>> from vllm import LLM, SamplingParams
>>> import torch
>>> llm = LLM(model="模型路径", trust_remote_code=True, tensor_parallel_size=4)
>>> prompts = ['你好']
>>> sampling_params = SamplingParams(max_tokens=100, temperature=0.0, repetition_penalty=1.03) #推荐repetition_penalty为1.03
>>> outputs = llm.generate(prompts, sampling_params)
>>> for output in outputs:
>>>     generated_text = output.outputs[0].text
>>>     print(generated_text)
```
