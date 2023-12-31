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
from llama_index.llms import LlamaCPP
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index import set_global_service_context
from llama_index.vector_stores import QdrantVectorStore
from qdrant_client import QdrantClient
from llama_index.llms.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)

STORAGE_DIR = "./qdrant_storage"  # directory to cache the generated index
# creates a persistant index to disk
client = QdrantClient("localhost", port=6333)

# create our vector store with hybrid indexing enabled
# batch_size controls how many nodes are encoded with sparse vectors at once
vector_store = QdrantVectorStore(
    "grade3", client=client, enable_hybrid=True, batch_size=20,force_disable_check_same_thread=True,
)

model_path = "/Volumes/FastSSD2/LLM/AIAnytime/manju/Llama2-Medical-Chatbot/models/zephyr-7b-beta.Q8_0.gguf"

HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
#STORAGE_DIR = "./storage"  # directory to cache the generated index
DATA_DIR = "./data"  # directory containing the documents to index

llm = LlamaCPP(
        model_path=model_path,
        temperature=0.01,
        max_new_tokens=7000,
        context_window=8192,
        generate_kwargs={},
        #model_kwargs={"n_gpu_layers": 1},
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
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
    # load the index from disk
    logger = logging.getLogger("uvicorn")
    logger.info(f"Loading index from {STORAGE_DIR}...")
    storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR,vector_store=vector_store)
    index = load_index_from_storage(storage_context,service_context=service_context)
    logger.info(f"Finished loading index from {STORAGE_DIR}")

    return index    
