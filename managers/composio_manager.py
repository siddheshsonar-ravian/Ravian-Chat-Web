"""
Composio Manager for handling tool integrations and user connections
"""
import os
import json
import logging
from typing import List, Optional, Any, Dict, Callable

from composio import Composio
from langchain_core.tools import StructuredTool
from pydantic import create_model, Field


class ComposioToolWrapper:
    """Wrapper to convert Composio tools to LangChain StructuredTool format"""

    def __init__(self, tool_dict: Dict[str, Any], app_name: Optional[str] = None):
        """
        Initialize tool wrapper

        Args:
            tool_dict: Raw tool definition from Composio API
            app_name: Optional app name to associate with the tool
        """
        self.tool_dict = tool_dict
        self.app_name = app_name

        # Composio tools come in OpenAI function format:
        # {'function': {'name': '...', 'description': '...', 'parameters': {...}}, 'type': 'function'}
        if 'function' in tool_dict and isinstance(tool_dict['function'], dict):
            function_def = tool_dict['function']
            self.slug = function_def.get('name', '')
            self.description = function_def.get('description', '')
            self.parameters = function_def.get('parameters', {})
        else:
            # Fallback for other formats
            self.slug = tool_dict.get('name', '')
            self.description = tool_dict.get('description', '')
            self.parameters = tool_dict.get('parameters', {})

        if not self.slug:
            print(f"[ComposioToolWrapper] WARNING: No slug found in tool_dict: {list(tool_dict.keys())}")
        else:
            print(f"[ComposioToolWrapper] Initialized: {self.slug}")

    def to_langchain_tool(self, composio_client: Composio, entity_id: str = "default") -> StructuredTool:
        """
        Convert Composio tool to LangChain StructuredTool

        Args:
            composio_client: Composio client instance for executing actions
            entity_id: Entity ID for tool execution

        Returns:
            StructuredTool: LangChain-compatible tool
        """
        # Ensure we have a valid name
        if not self.slug or self.slug == 'unknown_tool':
            raise ValueError(f"Invalid tool slug: {self.slug}")

        # Ensure we have a description
        if not self.description:
            self.description = f"Tool: {self.slug}"

        # Extract parameters
        properties = self.parameters.get('properties', {})
        required = self.parameters.get('required', [])

        # Create Pydantic model for tool arguments
        field_definitions = {}
        for param_name, param_info in properties.items():
            param_type = self._get_python_type(param_info)
            param_description = param_info.get('description', '')

            # Determine if field is required
            is_required = param_name in required

            # Handle default values
            if 'default' in param_info and param_info['default'] is not None:
                default_value = param_info['default']
            elif is_required:
                default_value = ...  # Required field
            else:
                default_value = None  # Optional field

            field_definitions[param_name] = (
                param_type,
                Field(default=default_value, description=param_description)
            )

        # Create dynamic Pydantic model
        if field_definitions:
            ArgsSchema = create_model(
                f"{self.slug}Args",
                **field_definitions
            )
        else:
            # If no parameters, create empty model
            ArgsSchema = create_model(f"{self.slug}Args")

        # Store slug for closure
        tool_slug = self.slug
        tool_description = self.description
        logger = logging.getLogger(__name__)

        # Create the tool function
        def tool_func(**kwargs) -> str:
            """Execute the Composio action"""
            try:
                logger.info(f"🔧 Executing tool: {tool_slug}")
                logger.info(f"📝 Parameters: {kwargs}")

                # Remove None values from kwargs
                filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}

                logger.info(f"📝 Filtered parameters: {filtered_kwargs}")

                # Extract toolkit name from slug
                toolkit_name = tool_slug.split('_')[0].lower()

                # Execute with toolkit version
                result = composio_client.tools.execute(
                    slug=tool_slug,
                    user_id=entity_id,
                    arguments=filtered_kwargs,
                )

                logger.info(f"✅ Tool {tool_slug} executed successfully")

                # Extract result data
                result_data = None
                if hasattr(result, 'data'):
                    result_data = result.data
                elif hasattr(result, 'response_data'):
                    result_data = result.response_data
                elif isinstance(result, dict):
                    result_data = result
                else:
                    result_data = {"result": str(result)}

                # Convert to string

                if isinstance(result_data, dict):
                    result_str = json.dumps(result_data, indent=2)
                else:
                    result_str = str(result_data)

                # TRUNCATE if too large (limit to ~10K chars to avoid context overflow)
                MAX_LENGTH = 10000
                if len(result_str) > MAX_LENGTH:
                    result_str = result_str[
                                     :MAX_LENGTH] + f"\n\n... (truncated {len(result_str) - MAX_LENGTH} characters)"
                    logger.warning(f"Tool output truncated from {len(result_str)} to {MAX_LENGTH} chars")

                return result_str

            except Exception as e:
                error_msg = str(e)
                logger.error(f"❌ Error executing {tool_slug}: {error_msg}")
                logger.error(f"Full error details:", exc_info=True)

                # Return detailed error for LLM to understand
                error_response = {
                    "error": error_msg,
                    "tool": tool_slug,
                    "params_attempted": filtered_kwargs,
                    "success": False
                }
                return json.dumps(error_response, indent=2)

        # Create StructuredTool with explicit name and description
        tool = StructuredTool(
            name=self.slug,
            description=tool_description,
            func=tool_func,
            args_schema=ArgsSchema
        )

        return tool

    def _get_python_type(self, param_info: Dict[str, Any]) -> type:
        """
        Convert JSON schema type to Python type

        Args:
            param_info: Parameter information from tool definition

        Returns:
            Python type
        """

        param_type = param_info.get('type', 'string')

        # Handle anyOf (union types)
        if 'anyOf' in param_info:
            types = []
            has_null = False

            for option in param_info['anyOf']:
                opt_type = option.get('type')
                if opt_type == 'null':
                    has_null = True
                elif opt_type:
                    types.append(self._map_json_type_to_python(opt_type))

            if types:
                # If multiple non-null types, use the first one
                python_type = types[0] if types else str

                # Make it optional if null is in anyOf
                if has_null:
                    return Optional[python_type]
                return python_type

        # Map single type
        python_type = self._map_json_type_to_python(param_type)

        return python_type

    def _map_json_type_to_python(self, json_type: str) -> type:
        """Map JSON schema type to Python type"""
        type_mapping = {
            'string': str,
            'integer': int,
            'number': float,
            'boolean': bool,
            'array': list,
            'object': dict,
        }
        return type_mapping.get(json_type, str)


class ComposioManager:
    """
    Manager for Composio integrations
    Handles fetching user connected apps and managing tools
    """

    def __init__(self):
        """Initialize the Composio manager"""
        self.composio = Composio(api_key=os.getenv("COMPOSIO_API_KEY"), toolkit_versions={'GMAIL': '20251027_00'})
        self._user_tools_cache = {}
        self.logger = logging.getLogger(__name__)

        # Set up logging if not already configured
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def create_tool_wrappers(
        self,
        raw_tools: List[Dict[str, Any]],
        app_name: Optional[str] = None
    ) -> List[ComposioToolWrapper]:
        """
        Convert raw tool definitions into wrapper objects.

        Args:
            raw_tools: Raw tool definitions from Composio API
            app_name: Optional app name to associate with tools

        Returns:
            List of wrapped tool objects
        """
        tools = [ComposioToolWrapper(tool, app_name) for tool in raw_tools]

        self.logger.info(f"Created {len(tools)} tool wrappers for {app_name}")
        if self.logger.isEnabledFor(logging.DEBUG):
            for tool in tools:
                self.logger.debug(f"Tool: {tool.slug} - {tool.description}")

        return tools

    async def get_tools_for_user(self, user_id: str = "default") -> List[StructuredTool]:
        """
        Get all available LangChain-compatible tools for a user

        Args:
            user_id: User identifier

        Returns:
            List of LangChain StructuredTool objects
        """
        if user_id in self._user_tools_cache:
            self.logger.info(f"Returning cached tools for user {user_id}")
            return self._user_tools_cache[user_id]

        try:
            # Get connected accounts for the user
            connections = self.composio.connected_accounts.list(user_ids=[user_id])

            if not connections.items:
                self.logger.warning(f"⚠️ No connected accounts found for user {user_id}")
                return []

            self.logger.info(f"📱 Found {len(connections.items)} connected accounts")

            # Log connected apps
            for conn in connections.items:
                app_name = conn.toolkit.slug if conn.toolkit else "unknown"
                status = "✅ Active" if conn.status == "ACTIVE" else f"⚠️ {conn.status}"
                self.logger.info(f"  - {app_name}: {status}")

            all_langchain_tools = []

            for conn in connections.items:
                app_name = conn.toolkit.slug.upper() if conn.toolkit else None

                if not app_name:
                    self.logger.warning(f"Skipping connection with no app name")
                    continue

                # Check connection status
                if conn.status != "ACTIVE":
                    self.logger.warning(f"⚠️ Skipping {app_name} - status: {conn.status}")
                    continue

                self.logger.info(f"🔄 Fetching tools for app: {app_name}")

                try:
                    # Get raw tools from Composio - returns list of dicts
                    raw_tools_response = self.composio.tools.get(
                        user_id=user_id,
                        toolkits=[app_name]
                    )

                    # Handle different response formats
                    if isinstance(raw_tools_response, list):
                        raw_tools = raw_tools_response
                    elif hasattr(raw_tools_response, 'items'):
                        raw_tools = raw_tools_response.items
                    else:
                        self.logger.warning(f"Unexpected tools format for {app_name}")
                        continue

                    if not raw_tools:
                        self.logger.warning(f"No tools found for {app_name}")
                        continue

                    self.logger.info(f"📦 Found {len(raw_tools)} raw tools for {app_name}")

                    # Wrap tools
                    tool_wrappers = self.create_tool_wrappers(raw_tools, app_name)

                    # Convert to LangChain tools
                    for wrapper in tool_wrappers:
                        try:
                            langchain_tool = wrapper.to_langchain_tool(
                                composio_client=self.composio,
                                entity_id=user_id
                            )
                            all_langchain_tools.append(langchain_tool)
                            self.logger.debug(f"  ✅ Created tool: {wrapper.slug}")
                        except Exception as tool_error:
                            self.logger.error(
                                f"  ❌ Error creating LangChain tool for {wrapper.slug}: {tool_error}"
                            )
                            continue

                except Exception as app_error:
                    self.logger.error(f"❌ Error processing app {app_name}: {app_error}")
                    import traceback
                    traceback.print_exc()
                    continue

            if all_langchain_tools:
                self._user_tools_cache[user_id] = all_langchain_tools
                self.logger.info(
                    f"✅ Successfully loaded {len(all_langchain_tools)} total tools for user {user_id}"
                )
            else:
                self.logger.warning(f"⚠️ No tools were successfully loaded for user {user_id}")

            return all_langchain_tools

        except Exception as e:
            self.logger.error(f"❌ Error fetching tools for user {user_id}: {e}")
            import traceback
            traceback.print_exc()
            return []

    def clear_cache(self, user_id: Optional[str] = None):
        """Clear the tools cache"""
        if user_id:
            self._user_tools_cache.pop(user_id, None)
            self.logger.info(f"Cleared cache for user {user_id}")
        else:
            self._user_tools_cache.clear()
            self.logger.info("Cleared all tool caches")