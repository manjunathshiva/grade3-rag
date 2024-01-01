import logging
import os
import torch

from llama_index import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
    ServiceContext,
)

from transformers import BitsAndBytesConfig
from langchain.llms import HuggingFaceHub
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index import set_global_service_context
from llama_index.vector_stores import QdrantVectorStore
from qdrant_client import QdrantClient
from llama_index.prompts import PromptTemplate
from llama_index.llms import HuggingFaceLLM

STORAGE_DIR = "./qdrant_storage"  # directory to cache the generated index
# creates a persistant index to disk
client = QdrantClient("localhost", port=6333)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# create our vector store with hybrid indexing enabled
# batch_size controls how many nodes are encoded with sparse vectors at once
vector_store = QdrantVectorStore(
    "grade3", client=client, enable_hybrid=True, batch_size=20,force_disable_check_same_thread=True,
)


HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
#STORAGE_DIR = "./storage"  # directory to cache the generated index
DATA_DIR = "./data"  # directory containing the documents to index

def messages_to_prompt(messages):
  prompt = ""
  for message in messages:
    if message.role == 'system':
      prompt += f"<|system|>\n{message.content}</s>\n"
    elif message.role == 'user':
      prompt += f"<|user|>\n{message.content}</s>\n"
    elif message.role == 'assistant':
      prompt += f"<|assistant|>\n{message.content}</s>\n"

  # ensure we start with a system prompt, insert blank if needed
  if not prompt.startswith("<|system|>\n"):
    prompt = "<|system|>\n</s>\n" + prompt

  # add final assistant prompt
  prompt = prompt + "<|assistant|>\n"

  return prompt


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

llm = HuggingFaceLLM(
    model_name="HuggingFaceH4/zephyr-7b-beta",
    tokenizer_name="HuggingFaceH4/zephyr-7b-beta",
    query_wrapper_prompt=PromptTemplate("<|system|>\n</s>\n<|user|>\n{query_str}</s>\n<|assistant|>\n"),
    context_window=3900,
    max_new_tokens=3500,
    model_kwargs={"quantization_config": quantization_config},
    # tokenizer_kwargs={},
    generate_kwargs={"temperature": 0.01, "top_k": 50, "do_sample" : True},
    messages_to_prompt=messages_to_prompt,
    device_map="auto",
)

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5",device=DEVICE)

service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model = embed_model,
    chunk_size=1024,
)

storage_context = StorageContext.from_defaults(
      vector_store=vector_store
)

# And set the service context
set_global_service_context(service_context)

if not os.path.exists(STORAGE_DIR + "/collections/grade3"):
    logger = logging.getLogger("uvicorn")
    logger.info("Creating new index")
    # load the documents and create the index
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    index = VectorStoreIndex.from_documents(documents,storage_context=storage_context,service_context=service_context)
    # store it for later
    index.storage_context.persist(STORAGE_DIR)
    logger.info(f"Finished creating new index. Stored in {STORAGE_DIR}")

def get_index():
    logger = logging.getLogger("uvicorn")
    logger.info(f"Loading index from {STORAGE_DIR}...")
    storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR,vector_store=vector_store)
    index = load_index_from_storage(storage_context,service_context=service_context)
    logger.info(f"Finished loading index from {STORAGE_DIR}")

    return index    
