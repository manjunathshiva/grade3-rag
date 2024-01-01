from typing import List

from app.utils.index import get_index
from fastapi import APIRouter, Depends, HTTPException, status
from llama_index import VectorStoreIndex
from llama_index.llms.base import MessageRole
from pydantic import BaseModel
from llama_index.llms import ChatMessage, MessageRole
from llama_index.prompts import ChatPromptTemplate


# Text QA Prompt
chat_text_qa_msgs = [
    ChatMessage(
        role=MessageRole.SYSTEM,
        content=(
            "<|system|>Always answer the question, even if the context isn't helpful.</s>"
        ),
    ),
    ChatMessage(
        role=MessageRole.USER,
        content=(
            "<|user|>Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, "
            "answer the question: {query_str}\n"
            "</s><|assistant|>"
        ),
    ),
]
text_qa_template = ChatPromptTemplate(chat_text_qa_msgs)


chat_router = r = APIRouter()


class _Message(BaseModel):
    role: MessageRole
    content: str


class _ChatData(BaseModel):
    messages: List[_Message]


class _Result(BaseModel):
    result: _Message


@r.post("")
async def chat(
    data: _ChatData,
    index: VectorStoreIndex = Depends(get_index),
) -> _Result:
    # check preconditions and get last message
    if len(data.messages) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No messages provided",
        )
    lastMessage = data.messages.pop()
    if lastMessage.role != MessageRole.USER:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Last message must be from user",
        )

    # query chat engine
    chat_engine = index.as_query_engine(
        #chat_mode="context",
        #text_qa_template=text_qa_template,
        sparse_top_k=12,
        vector_store_query_mode="hybrid",
        similarity_top_k=2,
        verbose=False,
    )
    response = chat_engine.query( lastMessage.content)
    return _Result(
        result=_Message(role=MessageRole.ASSISTANT, content=response.response)
    )

