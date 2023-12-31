from typing import List

from app.utils.index import get_index
from fastapi import APIRouter, Depends, HTTPException, status
from llama_index import VectorStoreIndex
from llama_index.llms.base import MessageRole, ChatMessage
from pydantic import BaseModel


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
        chat_mode="context",
        sparse_top_k=12,
        vector_store_query_mode="hybrid",
        similarity_top_k=2,
        system_prompt=(
         "You are a chatbot, able to have normal interactions, as well as talk"
         " about an Grade 3 Unit Tests, Holidays and Dairy of the School."
        ),
        verbose=False,
    )
    response = chat_engine.query( lastMessage.content)
    return _Result(
        result=_Message(role=MessageRole.ASSISTANT, content=response.response)
    )

