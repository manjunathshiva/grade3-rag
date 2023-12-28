import logging
import os

from llama_index import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
    ServiceContext,
)
from llama_index.llms import LlamaCPP
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
from llama_index import set_global_service_context
# Import the prompt wrapper...but for llama index
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts import PromptTemplate

# Define a new prompt template
template = """
<|system|>
Answer the question based on the context below. Keep the answer medium and concise. Respond "Unsure about answer" if not sure about the answer.
</s>
<|user|>
Context: {context_str}
Question: {query_str}
Answer:
</s>
<|assistant|>
"""
# Throw together the query wrapper


model_path = "/Volumes/FastSSD2/LLM/AIAnytime/manju/Llama2-Medical-Chatbot/models/zephyr-7b-beta.Q8_0.gguf"


STORAGE_DIR = "./storage"  # directory to cache the generated index
DATA_DIR = "./data"  # directory containing the documents to index

#llm =  HuggingFaceLLM(
#    model_name="Deci/DeciLM-6b-instruct",
#    tokenizer_name="Deci/DeciLM-6b-instruct",
#    query_wrapper_prompt=PromptTemplate("<|system|>\n</s>\n<|user|>\n{query_str}</s>\n<|assistant|>\n"),
    # query_wrapper_prompt=PromptTemplate(template),
#    context_window=4096,
#    max_new_tokens=512,
#    model_kwargs={'trust_remote_code':True},
#    generate_kwargs={"temperature": 0.0},
#    device_map="auto",
#)


llm = LlamaCPP(
        #model_path=model_path,
        model_url="https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q4_K_M.gguf",
        temperature=0.01,
        max_new_tokens=7000,
        context_window=8192,
        generate_kwargs={},
        #model_kwargs={"n_gpu_layers": 1},
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
    )

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5",device="cpu")

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
