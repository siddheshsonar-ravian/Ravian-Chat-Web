import os
from typing import Any, Dict, List, Literal
import uuid
import json

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI

from schema.message import MessagesState


class ChatAgent:
    """
    LangGraph chat agent with Composio tools integration and streaming support
    """

    def __init__(self, chat_id: str, composio_tools: List = None):
        """
        Initialize the chat agent

        Args:
            chat_id: Unique identifier for this chat session
            composio_tools: List of Composio tools to bind to the agent
        """
        self.chat_id = chat_id
        self.composio_tools = composio_tools or []

        # Create a dictionary for fast tool lookup by name
        self.tools_by_name = {}
        for tool in self.composio_tools:
            if hasattr(tool, 'name') and tool.name:
                self.tools_by_name[tool.name] = tool

        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            streaming=True,
            api_key=os.environ.get("OPENAI_API_KEY")
        )

        if self.composio_tools:
            self.llm_with_tools = self.llm.bind_tools(self.composio_tools)
            print(f"[ChatAgent] Bound {len(self.composio_tools)} Composio tools to LLM")
        else:
            self.llm_with_tools = self.llm
            print(f"[ChatAgent] No Composio tools bound")

    async def test_tool_binding(self):
        """Test if tools are properly bound"""
        messages = [
            SystemMessage(content="You are a helpful assistant with access to Gmail tools."),
            HumanMessage(content="List my last 3 emails")
        ]

        print("[TEST] Making test call to LLM with tools...")
        response = await self.llm_with_tools.ainvoke(messages)

        print(f"[TEST] Response content: {response.content[:200] if response.content else 'NO CONTENT'}")
        if hasattr(response, 'tool_calls'):
            print(f"[TEST] Tool calls: {response.tool_calls}")

        return response

    async def run_chat(
            self,
            session_id: str,
            message: str,
            ws_manager,
            mode: str = "chat"
    ) -> None:
        """
        Run the chat with streaming to WebSocket and tool execution
        """
        try:
            if mode == "task":
                system_content = (
                    "You are a helpful AI task execution assistant. You excel at breaking down complex tasks, "
                    "creating step-by-step plans, and executing multi-step workflows. When given a task:\n"
                    "1. Analyze the task requirements\n"
                    "2. Break it down into clear steps\n"
                    "3. Execute each step methodically\n"
                    "4. Provide progress updates\n"
                    "5. Summarize the results\n\n"
                    "Be thorough, systematic, and detail-oriented in your approach."
                )
            else:
                system_content = (
                    "You are a helpful AI assistant with access to various tools and integrations. "
                    "You can help with tasks including answering questions, providing explanations, "
                    "writing code, analyzing data, and interacting with external services like Gmail, YouTube, and GitHub. "
                    "\n\n"
                    "IMPORTANT: When you use tools:\n"
                    "1. If a tool returns an error, explain the error to the user clearly\n"
                    "2. Do NOT say you don't have access if a tool is available - instead try using it\n"
                    "3. If a tool fails, check the error message and try to fix the parameters\n"
                    "4. Tool results are returned as JSON - parse them and present information clearly\n"
                    "5. For Gmail: Use GMAIL_FETCH_EMAILS to check emails, with appropriate parameters like max_results\n"
                    "\n"
                    "Be concise, clear, and helpful in your responses."
                )

            system_msg = SystemMessage(content=system_content)
            user_msg = HumanMessage(content=message)

            print(f"[ChatAgent] Running in {mode.upper()} mode")

            if self.composio_tools:
                await self._run_with_tools(session_id, system_msg, user_msg, ws_manager)
            else:
                await self._stream_response(session_id, [system_msg, user_msg], ws_manager)

        except Exception as e:
            await ws_manager.send_chat_message(
                session_id,
                f"Error: {str(e)}",
                source="system"
            )
            raise

    async def _run_with_tools(
            self,
            session_id: str,
            system_msg: SystemMessage,
            user_msg: HumanMessage,
            ws_manager
    ) -> None:
        """Run chat with tool execution support AND real streaming"""
        messages = [system_msg, user_msg]
        max_iterations = 10
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            if ws_manager.is_session_stopped(session_id):
                break

            full_response = ""
            tool_calls_accumulator = {}

            print(f"[ChatAgent] Starting LLM streaming (iteration {iteration})...")

            async for chunk in self.llm_with_tools.astream(messages):
                if ws_manager.is_session_stopped(session_id):
                    break

                # Accumulate content
                if chunk.content:
                    full_response += chunk.content
                    await ws_manager.send_chat_message(
                        session_id,
                        chunk.content,
                        source="assistant",
                        is_streaming=True
                    )

                # Handle tool_call_chunks (streaming tool calls)
                if hasattr(chunk, 'tool_call_chunks') and chunk.tool_call_chunks:
                    for tc_chunk in chunk.tool_call_chunks:
                        index = tc_chunk.get('index', 0)
                        if index not in tool_calls_accumulator:
                            tool_calls_accumulator[index] = {
                                'name': '',
                                'args': '',
                                'id': tc_chunk.get('id'),
                                'type': 'tool_call'
                            }

                        if 'name' in tc_chunk and tc_chunk['name']:
                            tool_calls_accumulator[index]['name'] += tc_chunk['name']

                        if 'args' in tc_chunk and tc_chunk['args']:
                            tool_calls_accumulator[index]['args'] += tc_chunk['args']

                        if 'id' in tc_chunk and tc_chunk['id']:
                            tool_calls_accumulator[index]['id'] = tc_chunk['id']

            # Build tool_calls from accumulator
            tool_calls = []
            if tool_calls_accumulator:
                print(f"[ChatAgent] Building tool_calls from accumulator: {tool_calls_accumulator}")
                for idx in sorted(tool_calls_accumulator.keys()):
                    tc = tool_calls_accumulator[idx]
                    try:
                        args = json.loads(tc['args']) if tc['args'] else {}
                        tool_calls.append({
                            'name': tc['name'],
                            'args': args,
                            'id': tc['id'],
                            'type': 'tool_call'
                        })
                    except json.JSONDecodeError as e:
                        print(f"[ChatAgent] Error parsing tool args: {e}")

            print(f"[ChatAgent] Final tool_calls: {tool_calls}")

            # Create AI message
            ai_message = AIMessage(content=full_response)
            if tool_calls:
                ai_message.tool_calls = tool_calls
            messages.append(ai_message)

            # Execute tools if present
            if tool_calls:
                tool_names = [tc.get("name", "unknown") for tc in tool_calls if tc.get("name")]

                if not tool_names:
                    print(f"[ChatAgent] No valid tool names")
                    break

                await ws_manager.send_chat_message(
                    session_id,
                    f"🔧 Executing tools: {', '.join(tool_names)}",
                    source="system"
                )

                for tool_call in tool_calls:
                    tool_name = tool_call.get("name")
                    tool_args = tool_call.get("args", {})
                    tool_call_id = tool_call.get("id") or f"call_{uuid.uuid4().hex[:8]}"

                    if not tool_name:
                        continue

                    print(f"[ChatAgent] Executing: {tool_name} with {tool_args}")

                    try:
                        if tool_name not in self.tools_by_name:
                            result = f"Error: Tool '{tool_name}' not found"
                        else:
                            tool = self.tools_by_name[tool_name]
                            result = tool.invoke(tool_args)

                        print(f"[ChatAgent] Tool result: {str(result)[:500]}...")

                        tool_message = ToolMessage(
                            content=str(result),
                            tool_call_id=tool_call_id,
                            name=tool_name
                        )
                        messages.append(tool_message)

                        result_display = str(result)[:1000]
                        if len(str(result)) > 1000:
                            result_display += f"\n... (truncated)"

                        await ws_manager.send_chat_message(
                            session_id,
                            f"📊 Tool Output from '{tool_name}':\n{result_display}",
                            source="tool"
                        )

                    except Exception as e:
                        error_msg = f"Error executing {tool_name}: {str(e)}"
                        print(f"[ChatAgent] {error_msg}")

                        tool_message = ToolMessage(
                            content=error_msg,
                            tool_call_id=tool_call_id,
                            name=tool_name or "unknown"
                        )
                        messages.append(tool_message)

                        await ws_manager.send_chat_message(
                            session_id,
                            f"❌ {error_msg}",
                            source="tool"
                        )

                continue

            else:
                # No tool calls, complete
                if not ws_manager.is_session_stopped(session_id):
                    await ws_manager.send_chat_message(
                        session_id,
                        full_response,
                        source="assistant",
                        is_streaming=False,
                        is_complete=True
                    )
                break

        if iteration >= max_iterations:
            await ws_manager.send_chat_message(
                session_id,
                "⚠️ Maximum iterations reached.",
                source="system"
            )

    async def _stream_response(
            self,
            session_id: str,
            messages: List,
            ws_manager
    ) -> None:
        """Stream response without tool execution"""
        full_response = ""
        async for chunk in self.llm.astream(messages):
            if ws_manager.is_session_stopped(session_id):
                break

            if chunk.content:
                full_response += chunk.content
                await ws_manager.send_chat_message(
                    session_id,
                    chunk.content,
                    source="assistant",
                    is_streaming=True
                )

        if not ws_manager.is_session_stopped(session_id):
            await ws_manager.send_chat_message(
                session_id,
                full_response,
                source="assistant",
                is_streaming=False,
                is_complete=True
            )