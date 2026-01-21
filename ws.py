"""
WebSocket routes for LangGraph chat with Composio integration
"""

import asyncio
import json
from datetime import datetime

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from dependencies import get_websocket_manager, get_composio_manager
from managers.chat_agent import ChatAgent
from managers.connection import WebSocketManager
from managers.composio_manager import ComposioManager

router = APIRouter()

# Static user ID for testing
TEST_USER_ID = "pg-test-cda9efc8-3909-4d52-b8b5-163a3a8d8daa"


@router.websocket("/chat/{chat_id}")
async def run_websocket(
        websocket: WebSocket,
        chat_id: str,
        ws_manager: WebSocketManager = Depends(get_websocket_manager),
        composio_manager: ComposioManager = Depends(get_composio_manager),
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

        # TEST: Verify tool binding works
        if composio_tools:
            await ws_manager.send_chat_message(
                chat_id,
                "🧪 Running tool binding test...",
                source="system"
            )

            try:
                test_response = await chat.test_tool_binding()

                await ws_manager.send_chat_message(
                    chat_id,
                    f"✅ Test complete!",
                    source="system"
                )

                if hasattr(test_response, 'tool_calls') and test_response.tool_calls:
                    tool_names = [tc.get('name', 'unknown') for tc in test_response.tool_calls]
                    await ws_manager.send_chat_message(
                        chat_id,
                        f"✅ LLM wants to call tools: {', '.join(tool_names)}",
                        source="system"
                    )
                else:
                    await ws_manager.send_chat_message(
                        chat_id,
                        f"⚠️ Warning: No tool calls. Response: {test_response.content[:200]}",
                        source="system"
                    )

            except Exception as test_error:
                await ws_manager.send_chat_message(
                    chat_id,
                    f"❌ Test failed: {str(test_error)}",
                    source="system"
                )
                import traceback
                traceback.print_exc()

        await ws_manager.send_chat_message(
            chat_id,
            "✅ Ready! Send a message to start chatting.",
            source="system"
        )

        while True:
            try:
                raw_message = await websocket.receive_text()
                message = json.loads(raw_message)

                if message.get("type") == "start":
                    if message.get("task"):
                        asyncio.create_task(
                            ws_manager.start_chat_stream(
                                session_id=chat_id,
                                initial_message=message.get("task"),
                                chat_agent=chat
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