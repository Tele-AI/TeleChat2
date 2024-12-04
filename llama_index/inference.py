

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

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("./document").load_data()
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=Settings.embed_model,
    transformations=Settings.transformations
)

query_engine = index.as_query_engine()
your_query = "What are the challenges in developing the AI?"
with torch.autocast("cuda", enabled=True):
    print(query_engine.query(your_query).response)


