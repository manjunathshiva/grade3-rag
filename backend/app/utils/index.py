import logging
import os

from llama_index import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
    ServiceContext,
)

from langchain.llms import HuggingFaceHub
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index import set_global_service_context
from llama_index.vector_stores import QdrantVectorStore
from qdrant_client import QdrantClient
STORAGE_DIR = "./qdrant_storage"  # directory to cache the generated index
# creates a persistant index to disk
client = QdrantClient("localhost", port=6333)

# create our vector store with hybrid indexing enabled
# batch_size controls how many nodes are encoded with sparse vectors at once
vector_store = QdrantVectorStore(
    "grade3", client=client, enable_hybrid=True, batch_size=20,force_disable_check_same_thread=True,
)


HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
#STORAGE_DIR = "./storage"  # directory to cache the generated index
DATA_DIR = "./data"  # directory containing the documents to index

llm = HuggingFaceHub(
                    repo_id='HuggingFaceH4/zephyr-7b-beta',
                    model_kwargs={"temperature":0.01, 
                     "do_sample":True, 
                    # "top_k":10, 
                    # "top_p":0.95, 
                    # "num_return_sequences":1, 
                    # "max_length":3900,
                     "max_new_tokens":3500,
                     "context_size":4096,
                     "repetition_penalty":1.3,
                   #  "length_penalty":1.0,
                   #  "early_stopping":False
                    }
    )

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5",device="cpu")

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
