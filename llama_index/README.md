
# LlamaIndex

为了实现 TeleChat 系列模型根据外部数据（例如文档、网页等）进行检索问答，我们提供了 LlamaIndex 教程。本指南可帮助您使用 LlamaIndex 和TeleChat2-7B 模型快速实现检索增强生成 (RAG)。


## 官方地址

https://github.com/run-llama/llama_index/tree/main


## 适配流程

### 环境安装
```bash
pip install llama-index
pip install llama-index-llms-huggingface
pip install llama-index-readers-web
pip install llama-index-embeddings-huggingface
pip install flash-attn
pip install pydantic==2.8.2
pip install langchain --upgrade 
```
其中：

如果不安装 llama-index-embeddings-huggingface，会出现该错误: No module named 'llama_index,embeddings.huggingface'

flash-attn 用于TeleChat模型推理

pydantic与langchain包的更新是为了解决在旧版本中版本不兼容的报错

### 设置参数

#### TeleChat模型下载
首先下载需要的telechat模型，例如模型所在位置为：
./telechat2-7B
其中包含以下文件：
1. config.json                 
2. modeling_telechat2.py               
3. tokenizer.model
4. configuration_telechat2.py            
5. tokenizer_config.json
6. generation_config.json      
7. pytorch_model.bin.index.json 
8. tokenization_telechat2.py     
9. generation_utils.py        
10. pytorch_model_00001-of-00004.bin  
11. pytorch_model_00002-of-00004.bin
12. pytorch_model_00003-of-00004.bin
13. pytorch_model_00004-of-00004.bin

#### 文本向量化模型下载
模型地址：https://huggingface.co/sentence-transformers/all-mpnet-base-v2/tree/main

将下载后的所有文件放于：./all-mpnet-base-v2 目录下即可。

```python
import torch
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

model_path = "telechat2-7B"

Settings.llm = HuggingFaceLLM(
    model_name=model_path,
    tokenizer_name=model_path,
    context_window=30000,
    max_new_tokens=2000,
    generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    device_map="cuda",
)

# Set embedding model
Settings.embed_model = HuggingFaceEmbedding(
    model_name = "all-mpnet-base-v2",
)

# Set the size of the text chunk for retrieval
Settings.transformations = [SentenceSplitter(chunk_size=1024)]
```

#### 构建索引
现在我们可以从文档或网站构建索引。

以下代码片段展示了如何为本地名为’document’的文件夹中的文件（无论是PDF格式还是TXT格式）构建索引。
```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("./document").load_data()
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=Settings.embed_model,
    transformations=Settings.transformations
)
```

#### 检索增强(RAG)
```python
query_engine = index.as_query_engine()
your_query = "<your query here>"
print(query_engine.query(your_query).response)
```

### 示例代码运行
示例代码位于当前目录 "inference.py" 文件中，使用的知识库为"document"目录，该目录可以放置知识库，例如"example.txt"

```bash
python inference.py
```

在该示例中，问题为：

"What are the challenges in developing the AI?"

模型输出应为：

The challenges in developing AI include:

1. Privacy issues: AI systems often require access to and analysis of large amounts of personal data, raising concerns about data security and user privacy.

2. Job losses: With the widespread application of AI across various industries, there is concern that machines may replace certain human jobs, leading to potential job losses.

3. Ethical issues: Ensuring that AI systems' decision-making processes are fair and transparent, and avoiding algorithmic bias, are challenges that researchers and policymakers need to address together.
