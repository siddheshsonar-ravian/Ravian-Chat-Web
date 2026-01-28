# managers/chat_service.py
"""
Chat persistence service - abstraction layer for storage
Currently uses localStorage, designed for easy MongoDB migration
"""
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path


class ChatService:
    """
    Chat persistence service
    Storage backend can be easily swapped (localStorage -> MongoDB)
    """

    def __init__(self, storage_type: str = "local"):
        """
        Initialize chat service

        Args:
            storage_type: "local" or "mongodb" (future)
        """
        self.storage_type = storage_type

        if storage_type == "local":
            # For now, use file-based storage (simulating localStorage)
            self.storage_path = Path("data/chats")
            self.storage_path.mkdir(parents=True, exist_ok=True)

    def create_chat(self, user_id: str, chat_id: str = None) -> Dict:
        """
        Create a new chat session

        Args:
            user_id: User identifier
            chat_id: Optional chat ID (if not provided, generates UUID-4)

        Returns:
            Chat object with id, user_id, created_at, messages
        """
        chat_id = chat_id or str(uuid.uuid4())

        chat = {
            "id": chat_id,
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "title": "New Chat",  # Can be auto-generated from first message
            "messages": []
        }

        self._save_chat(chat)
        return chat

    def get_chat(self, chat_id: str) -> Optional[Dict]:
        """Get a specific chat by ID"""
        if self.storage_type == "local":
            chat_file = self.storage_path / f"{chat_id}.json"
            if chat_file.exists():
                with open(chat_file, 'r') as f:
                    return json.load(f)
        return None

    def get_user_chats(self, user_id: str, limit: int = 50) -> List[Dict]:
        """
        Get all chats for a user

        Args:
            user_id: User identifier
            limit: Maximum number of chats to return

        Returns:
            List of chat objects (without full message history)
        """
        if self.storage_type == "local":
            chats = []
            for chat_file in self.storage_path.glob("*.json"):
                try:
                    with open(chat_file, 'r') as f:
                        chat = json.load(f)
                        if chat.get("user_id") == user_id:
                            # Return metadata only (not full messages)
                            chats.append({
                                "id": chat["id"],
                                "user_id": chat["user_id"],
                                "created_at": chat["created_at"],
                                "updated_at": chat["updated_at"],
                                "title": chat.get("title", "New Chat"),
                                "message_count": len(chat.get("messages", []))
                            })
                except Exception as e:
                    print(f"Error reading chat {chat_file}: {e}")

            # Sort by updated_at (most recent first)
            chats.sort(key=lambda x: x["updated_at"], reverse=True)
            return chats[:limit]

        return []

    def add_message(self, chat_id: str, message: Dict) -> bool:
        """
        Add a message to a chat

        Args:
            chat_id: Chat identifier
            message: Message object with role, content, timestamp
        """
        chat = self.get_chat(chat_id)
        if not chat:
            return False

        # Add message
        chat["messages"].append({
            "id": str(uuid.uuid4()),
            "role": message.get("role", "user"),  # "user", "assistant", "system", "tool"
            "content": message.get("content", ""),
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": message.get("metadata", {})
        })

        # Update title from first user message if still "New Chat"
        if chat["title"] == "New Chat" and message.get("role") == "user":
            content = message.get("content", "")
            chat["title"] = content[:50] + ("..." if len(content) > 50 else "")

        # Update timestamp
        chat["updated_at"] = datetime.utcnow().isoformat()

        self._save_chat(chat)
        return True

    def delete_chat(self, chat_id: str) -> bool:
        """Delete a chat"""
        if self.storage_type == "local":
            chat_file = self.storage_path / f"{chat_id}.json"
            if chat_file.exists():
                chat_file.unlink()
                return True
        return False

    def update_chat_title(self, chat_id: str, title: str) -> bool:
        """Update chat title"""
        chat = self.get_chat(chat_id)
        if not chat:
            return False

        chat["title"] = title
        chat["updated_at"] = datetime.utcnow().isoformat()
        self._save_chat(chat)
        return True

    def _save_chat(self, chat: Dict):
        """Internal: Save chat to storage"""
        if self.storage_type == "local":
            chat_file = self.storage_path / f"{chat['id']}.json"
            with open(chat_file, 'w') as f:
                json.dump(chat, f, indent=2)

    # Future MongoDB implementation would go here:
    # def _init_mongodb(self):
    #     from pymongo import MongoClient
    #     self.client = MongoClient(os.getenv("MONGODB_URI"))
    #     self.db = self.client.chat_db
    #     self.chats_collection = self.db.chats