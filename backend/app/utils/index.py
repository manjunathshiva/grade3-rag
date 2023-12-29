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


HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
STORAGE_DIR = "./storage"  # directory to cache the generated index
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
)
# And set the service context
set_global_service_context(service_context)

def get_index():
    logger = logging.getLogger("uvicorn")
    # check if storage already exists
    if not os.path.exists(STORAGE_DIR):
        logger.info("Creating new index")
        # load the documents and create the index
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        index = VectorStoreIndex.from_documents(documents,service_context=service_context)
        # store it for later
        index.storage_context.persist(STORAGE_DIR)
        logger.info(f"Finished creating new index. Stored in {STORAGE_DIR}")
    else:
        # load the existing index
        logger.info(f"Loading index from {STORAGE_DIR}...")
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        index = load_index_from_storage(storage_context,service_context=service_context)
        logger.info(f"Finished loading index from {STORAGE_DIR}")
    return index
