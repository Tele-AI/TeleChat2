
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

#### 基础用法
您可以仅使用您的文档配合 langchain 来构建一个问答应用。该项目的实现流程包括加载文件 -> 阅读文本 -> 文本分段 -> 文本向量化 -> 问题向量化 -> 将最相似的前k个文本向量与问题向量匹配 -> 将匹配的文本作为上下文连同问题一起纳入提示 -> 提交给TeleChat2生成答案。以下是一个示例：
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from langchain.llms.base import LLM
from abc import ABC
from langchain.callbacks.manager import CallbackManagerForLLMRun
from typing import Any, List, Mapping, Optional

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


model_path = "telechat2-7B"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16
)

class TeleChat(LLM, ABC):
    max_token: int = 10000
    temperature: float = 0.01
    top_p = 0.9
    history_len: int = 3

    def __init__(self):
         super().__init__()

    @property
    def _llm_type(self) -> str:
         return "TeleChat"

    @property
    def _history_len(self) -> int:
         return self.history_len

    def set_history_len(self, history_len: int = 10) -> None:
         self.history_len = history_len

    def _call(
         self,
         prompt: str,
         stop: Optional[List[str]] = None,
         run_manager: Optional[CallbackManagerForLLMRun] = None,
     ) -> str:

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
  
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
         """Get the identifying parameters."""
         return {"max_token": self.max_token,
                 "temperature": self.temperature,
                 "top_p": self.top_p,
                 "history_len": self.history_len}
```
加载telechat2-7B模型后，您可以指定需要用于知识库问答的txt文件。

```python
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import os
import re
import torch
import argparse
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from typing import List, Tuple
import numpy as np
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import RetrievalQA


class ChineseTextSplitter(CharacterTextSplitter):
    def __init__(self, pdf: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.pdf = pdf

    def split_text(self, text: str) -> List[str]:
        if self.pdf:
            text = re.sub(r"\n{3,}", "\n", text)
            text = re.sub('\s', ' ', text)
            text = text.replace("\n\n", "")
        sent_sep_pattern = re.compile(
            '([﹒﹔﹖﹗．。！？]["’”」』]{0,2}|(?=["‘“「『]{1,2}|$))')
        sent_list = []
        for ele in sent_sep_pattern.split(text):
            if sent_sep_pattern.match(ele) and sent_list:
                sent_list[-1] += ele
            elif ele:
                sent_list.append(ele)
        return sent_list


def load_file(filepath):
    loader = TextLoader(filepath, autodetect_encoding=True)
    textsplitter = ChineseTextSplitter(pdf=False)
    docs = loader.load_and_split(textsplitter)
    write_check_file(filepath, docs)
    return docs

def write_check_file(filepath, docs):
    folder_path = os.path.join(os.path.dirname(filepath), "tmp_files")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    fp = os.path.join(folder_path, 'load_file.txt')
    with open(fp, 'a+', encoding='utf-8') as fout:
        fout.write("filepath=%s,len=%s" % (filepath, len(docs)))
        fout.write('\n')
        for i in docs:
            fout.write(str(i))
            fout.write('\n')
        fout.close()


def separate_list(ls: List[int]) -> List[List[int]]:
    lists = []
    ls1 = [ls[0]]
    for i in range(1, len(ls)):
        if ls[i - 1] + 1 == ls[i]:
            ls1.append(ls[i])
        else:
            lists.append(ls1)
            ls1 = [ls[i]]
    lists.append(ls1)
    return lists

class FAISSWrapper(FAISS):
    chunk_size = 250
    chunk_conent = True
    score_threshold = 0

    def similarity_search_with_score_by_vector(
            self, embedding: List[float], k: int = 4
    ) -> List[Tuple[Document, float]]:
        scores, indices = self.index.search(np.array([embedding], dtype=np.float32), k)
        docs = []
        id_set = set()
        store_len = len(self.index_to_docstore_id)
        for j, i in enumerate(indices[0]):
            if i == -1 or 0 < self.score_threshold < scores[0][j]:
                # This happens when not enough docs are returned.
                continue
            _id = self.index_to_docstore_id[i]
            doc = self.docstore.search(_id)
            if not self.chunk_conent:
                if not isinstance(doc, Document):
                    raise ValueError(f"Could not find document for id {_id}, got {doc}")
                doc.metadata["score"] = int(scores[0][j])
                docs.append(doc)
                continue
            id_set.add(i)
            docs_len = len(doc.page_content)
            for k in range(1, max(i, store_len - i)):
                break_flag = False
                for l in [i + k, i - k]:
                    if 0 <= l < len(self.index_to_docstore_id):
                        _id0 = self.index_to_docstore_id[l]
                        doc0 = self.docstore.search(_id0)
                        if docs_len + len(doc0.page_content) > self.chunk_size:
                            break_flag = True
                            break
                        elif doc0.metadata["source"] == doc.metadata["source"]:
                            docs_len += len(doc0.page_content)
                            id_set.add(l)
                if break_flag:
                    break
        if not self.chunk_conent:
            return docs
        
        if len(id_set) == 0 and self.score_threshold > 0:
            return []
        id_list = sorted(list(id_set))
        id_lists = separate_list(id_list)
        for id_seq in id_lists:
            for id in id_seq:
                if id == id_seq[0]:
                    _id = self.index_to_docstore_id[id]
                    doc = self.docstore.search(_id)
                else:
                    _id0 = self.index_to_docstore_id[id]
                    doc0 = self.docstore.search(_id0)
                    doc.page_content += " " + doc0.page_content
            if not isinstance(doc, Document):
                raise ValueError(f"Could not find document for id {_id}, got {doc}")
            doc_score = min([scores[0][id] for id in [indices[0].tolist().index(i) for i in id_seq if i in indices[0]]])
            doc.metadata["score"] = int(doc_score)
            docs.append((doc, doc_score))
        return docs


if __name__ == '__main__':
    # load docs (pdf file or txt file)
    filepath = './example.txt'
    # Embedding model name
    EMBEDDING_MODEL = 'text2vec'
    PROMPT_TEMPLATE = """Known information:
    {context_str}
    Based on the above known information, respond to the user's question concisely and professionally. If an answer cannot be derived from it, say 'The question cannot be answered with the given information' or 'Not enough relevant information has been provided,' and do not include fabricated details in the answer. Please respond in English. The question is {question}"""
    # Embedding running device
    EMBEDDING_DEVICE = "cuda"
    # return top-k text chunk from vector store
    VECTOR_SEARCH_TOP_K = 3
    CHAIN_TYPE = 'stuff'
    embedding_model_dict = {
        "text2vec": "all-mpnet-base-v2",
    }
    llm = TeleChat()
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[EMBEDDING_MODEL],model_kwargs={'device': EMBEDDING_DEVICE})

    docs = load_file(filepath)

    docsearch = FAISSWrapper.from_documents(docs, embeddings)

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=["context_str", "question"]
    )

    chain_type_kwargs = {"prompt": prompt, "document_variable_name": "context_str"}
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=CHAIN_TYPE,
        retriever=docsearch.as_retriever(search_kwargs={"k": VECTOR_SEARCH_TOP_K}),
        chain_type_kwargs=chain_type_kwargs)

    query = "What are the challenges in developing the AI?"

    print("output is : \n")
    print(qa.run(query))
```
#### 下一步
现在，您可以在您自己的文档上与TeleChat2进行交流。继续阅读文档，尝试探索模型检索的更多高级用法！

### 示例代码运行
示例代码位于当前目录 "inference.py" 文件中，使用的知识库为"example.txt"文件。

```bash
python inference.py
```

在该示例中，问题为："What are the challenges in developing the AI?"

模型输出应为：The challenges in developing AI include privacy issues, ensuring the security of personal data, and user privacy, as well as potential job losses due to machine replacement certain human. Ethical issues such as ensuring fair and transparent decision-making proceses and avoiding algorithmic bias also need to be addressed.
