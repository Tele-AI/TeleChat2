
# Langchain

本教程旨在帮助您利用TeleChat模型与langchain，将本地知识库进行向量化，并使用faiss向量数据库进行高效检索，最终构建问答应用。目标是建立一个知识库问答解决方案。


## 官方文档
https://github.com/langchain-ai/langchain


## 适配流程

### 环境安装
```bash
pip install langchain==0.0.174
pip install faiss-gpu
pip install sentence-transformers
pip install flash-attn
```
其中：

faiss-gpu用于对向量化的文本进行高速检索

flash-attn 用于TeleChat模型推理

### TeleChat模型下载
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

### 文本向量化模型下载
模型地址：https://huggingface.co/sentence-transformers/all-mpnet-base-v2/tree/main

将下载后的所有文件放于：./all-mpnet-base-v2 目录下即可。

### 示例代码运行
示例代码位于当前目录 "inference.py" 文件中，使用的知识库为"example.txt"文件。

```bash
python inference.py
```

在该示例中，问题为："What are the challenges in developing the AI?"

模型输出应为：The challenges in developing AI include privacy issues, ensuring the security of personal data, and user privacy, as well as potential job losses due to machine replacement certain human. Ethical issues such as ensuring fair and transparent decision-making proceses and avoiding algorithmic bias also need to be addressed.