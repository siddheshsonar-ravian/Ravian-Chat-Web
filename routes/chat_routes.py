# routes/chat_routes.py
"""
Chat management API routes
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional

from managers.chat_service import ChatService

router = APIRouter(prefix="/api/chats", tags=["chats"])

# Dependency
def get_chat_service() -> ChatService:
    return ChatService(storage_type="local")


class CreateChatRequest(BaseModel):
    user_id: str


class AddMessageRequest(BaseModel):
    role: str  # "user", "assistant", "system", "tool"
    content: str
    metadata: Optional[dict] = {}


@router.post("/")
async def create_chat(
    request: CreateChatRequest,
    chat_service: ChatService = Depends(get_chat_service)
):
    """Create a new chat"""
    chat = chat_service.create_chat(user_id=request.user_id)
    return chat


@router.get("/user/{user_id}")
async def get_user_chats(
    user_id: str,
    limit: int = 50,
    chat_service: ChatService = Depends(get_chat_service)
):
    """Get all chats for a user"""
    chats = chat_service.get_user_chats(user_id=user_id, limit=limit)
    return {"chats": chats}


@router.get("/{chat_id}")
async def get_chat(
    chat_id: str,
    chat_service: ChatService = Depends(get_chat_service)
):
    """Get a specific chat with full message history"""
    chat = chat_service.get_chat(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@router.post("/{chat_id}/messages")
async def add_message(
    chat_id: str,
    message: AddMessageRequest,
    chat_service: ChatService = Depends(get_chat_service)
):
    """Add a message to a chat"""
    success = chat_service.add_message(chat_id, message.dict())
    if not success:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"success": True}


@router.delete("/{chat_id}")
async def delete_chat(
    chat_id: str,
    chat_service: ChatService = Depends(get_chat_service)
):
    """Delete a chat"""
    success = chat_service.delete_chat(chat_id)
    if not success:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"success": True}


@router.patch("/{chat_id}/title")
async def update_title(
    chat_id: str,
    title: str,
    chat_service: ChatService = Depends(get_chat_service)
):
    """Update chat title"""
    success = chat_service.update_chat_title(chat_id, title)
    if not success:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"success": True}