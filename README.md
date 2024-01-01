This is a [LlamaIndex](https://www.llamaindex.ai/) project bootstrapped with [`create-llama`](https://github.com/run-llama/LlamaIndexTS/tree/main/packages/create-llama).

## Getting Started

First, startup the backend as described in the [backend README](./backend/README.md).

Go to ./backend and run qdrant docker
1)docker pull qdrant/qdrant
2)docker run -d -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant

Second, run the development server of the frontend as described in the [frontend README](./frontend/README.md).

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

For Production, Go to ./frontend
1)npm run build
2)npx serve@latest out -l 80  (-l 80 if you want to run in port 80)


## Note
Default HuggingFace Hub is used so export HUGGINGFACEHUB_API_TOKEN=hf_XXXXX before running the backend
If you want to use LLamaCPP and models locally, the copy index.py and chat.py from root folder to backend (  ./backend/app/api/routers/chat.py and ./backend/app/utils/index.py and update the model_path in index.py where you have donloaded GGUF from Hugging Face )

## Learn More

To learn more about LlamaIndex, take a look at the following resources:

- [LlamaIndex Documentation](https://docs.llamaindex.ai) - learn about LlamaIndex (Python features).
- [LlamaIndexTS Documentation](https://ts.llamaindex.ai) - learn about LlamaIndex (Typescript features).

You can check out [the LlamaIndexTS GitHub repository](https://github.com/run-llama/LlamaIndexTS) - your feedback and contributions are welcome!
