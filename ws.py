"""
WebSocket routes for LangGraph chat with Composio integration
"""

import asyncio
import json
import os
from datetime import datetime

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from dependencies import get_websocket_manager, get_composio_manager, get_chat_service
from managers.chat_agent import ChatAgent
from managers.connection import WebSocketManager
from managers.composio_manager import ComposioManager
from managers.chat_service import ChatService

router = APIRouter()

# User ID for Composio entity - loaded from .env
# This must match the entity_id in Composio where accounts are connected
TEST_USER_ID = os.getenv("COMPOSIO_USER_ID")


@router.websocket("/chat/{chat_id}")
async def run_websocket(
        websocket: WebSocket,
        chat_id: str,
        ws_manager: WebSocketManager = Depends(get_websocket_manager),
        composio_manager: ComposioManager = Depends(get_composio_manager),
        chat_service: ChatService = Depends(get_chat_service),
):
    """WebSocket endpoint for chat communication with Composio tools"""

    # Connect websocket first
    connected = await ws_manager.connect(websocket, chat_id)
    if not connected:
        await websocket.close(code=4002, reason="Failed to establish connection")
        return

    try:
        # Fetch Composio tools for the user
        await ws_manager.send_chat_message(
            chat_id,
            "🔄 Loading your connected apps and tools...",
            source="system"
        )

        composio_tools = await composio_manager.get_tools_for_user(TEST_USER_ID)

        # Create chat agent with Composio tools
        chat = ChatAgent(chat_id=chat_id, composio_tools=composio_tools)

        if composio_tools:
            tool_names = [t.name for t in composio_tools if hasattr(t, 'name')]
            await ws_manager.send_chat_message(
                chat_id,
                f"Loaded {len(composio_tools)} tools: {', '.join(tool_names[:5])}{'...' if len(tool_names) > 5 else ''}",
                source="system"
            )

        await ws_manager.send_chat_message(
            chat_id,
            "✅ Ready! Send a message to start chatting.",
            source="system"
        )

        # Load or create chat in storage
        stored_chat = chat_service.get_chat(chat_id)
        if not stored_chat:
            stored_chat = chat_service.create_chat(TEST_USER_ID, chat_id=chat_id)
            print(f"[WS] Created new chat: {stored_chat['id']}")
        else:
            print(f"[WS] Loaded existing chat with {len(stored_chat.get('messages', []))} messages")

        while True:
            try:
                raw_message = await websocket.receive_text()
                message = json.loads(raw_message)

                if message.get("type") == "start":
                    if message.get("task"):
                        user_message = message.get("task")

                        # Load current chat history
                        current_chat = chat_service.get_chat(chat_id)
                        chat_history = current_chat.get("messages", []) if current_chat else []

                        # Save user message to history
                        chat_service.add_message(chat_id, {
                            "role": "user",
                            "content": user_message
                        })

                        asyncio.create_task(
                            ws_manager.start_chat_stream(
                                session_id=chat_id,
                                initial_message=user_message,
                                chat_agent=chat,
                                chat_history=chat_history,
                                chat_service=chat_service
                            )
                        )
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "error": "Invalid start message format",
                            "timestamp": datetime.utcnow().isoformat(),
                        })

                elif message.get("type") == "stop":
                    reason = message.get("reason") or "User requested stop/cancellation"
                    await ws_manager.stop_session(chat_id, reason=reason)

                elif message.get("type") == "ping":
                    await websocket.send_json(
                        {"type": "pong", "timestamp": datetime.utcnow().isoformat()}
                    )

                elif message.get("type") == "input_response":
                    response = message.get("response")
                    if response is not None:
                        await ws_manager.handle_input_response(chat_id, response)

            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "error": "Invalid message format",
                    "timestamp": datetime.utcnow().isoformat(),
                })

    except WebSocketDisconnect:
        print(f"WebSocket disconnected for {chat_id}")
    except Exception as e:
        print(f"Error in WebSocket handler: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await ws_manager.disconnect(chat_id)