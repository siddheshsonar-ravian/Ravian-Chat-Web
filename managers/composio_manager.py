"""
Composio Manager - Simplified using official Composio LangChain provider
This handles ALL versioning automatically - no manual version management needed!
"""
import os
import logging
from typing import List

from composio import Composio
try:
    from composio_langchain import LangchainProvider
    LANGCHAIN_PROVIDER_AVAILABLE = True
except ImportError:
    # Fallback if provider not available
    LANGCHAIN_PROVIDER_AVAILABLE = False
    print("Warning: composio_langchain not available, using basic provider")


class ComposioManager:
    """
    Simplified Composio manager using official Composio + LangChain provider

    Benefits:
    - Automatic toolkit versioning (no manual version management)
    - Automatic tool conversion to LangChain format
    - Connected account detection
    - Built-in error handling
    """

    def __init__(self):
        """Initialize the Composio manager"""
        self.api_key = os.getenv("COMPOSIO_API_KEY")

        if not self.api_key:
            raise ValueError("COMPOSIO_API_KEY environment variable is required")

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

        # Initialize Composio client with LangChain provider
        # This automatically handles tool conversion to LangChain format
        if LANGCHAIN_PROVIDER_AVAILABLE:
            self.composio = Composio(
                api_key=self.api_key,
                provider=LangchainProvider()
            )
            self.logger.info("✅ ComposioManager initialized with LangChain provider")
        else:
            self.composio = Composio(api_key=self.api_key)
            self.logger.info("✅ ComposioManager initialized (basic mode)")

    async def get_tools_for_user(self, user_id: str) -> List:
        """
        Get all LangChain-compatible tools for a user

        This automatically handles:
        - Toolkit versioning (uses latest compatible versions automatically)
        - Tool conversion to LangChain format (via LangchainProvider)
        - Connected account detection (only returns tools for active connections)
        - Error handling and logging

        Args:
            user_id: User identifier (entity_id in Composio)

        Returns:
            List of LangChain-compatible tools ready to use with ChatOpenAI.bind_tools()
        """
        try:
            self.logger.info(f"🔄 Fetching tools for user: {user_id}")

            # Get connected accounts for the user to determine which apps are available
            # Note: In Composio, connections can be filtered by entity_id OR we can get all
            try:
                # Try getting connections for specific entity first
                connections = self.composio.connected_accounts.list(user_ids=[user_id])
            except Exception as e:
                self.logger.warning(f"Could not get connections for user_id {user_id}: {e}")
                # Fallback: Get all connections and filter later
                try:
                    connections = self.composio.connected_accounts.list()
                    self.logger.info(f"📡 Fetched all connections, will filter for user {user_id}")
                except Exception as e2:
                    self.logger.error(f"Could not get any connections: {e2}")
                    return []

            if not connections or not connections.items:
                self.logger.warning(f"⚠️ No connected accounts found for user {user_id}")
                self.logger.info("💡 User needs to connect their accounts via Composio dashboard")
                self.logger.info(f"💡 Or check if entity_id '{user_id}' is correct")
                return []

            self.logger.info(f"📱 Found {len(connections.items)} connected accounts")

            # Log connected apps with detailed info
            connected_toolkits = []
            for conn in connections.items:
                toolkit_name = conn.toolkit.slug.upper() if conn.toolkit else "unknown"

                # Log detailed connection info for debugging
                conn_id = getattr(conn, 'id', 'unknown')
                conn_entity = getattr(conn, 'entity_id', 'unknown')

                # Check status - could be "ACTIVE", "ENABLED", or other values
                status = getattr(conn, 'status', 'UNKNOWN')
                is_active = status in ["ACTIVE", "ENABLED"]

                status_emoji = "✅" if is_active else "⚠️"
                self.logger.info(f"  - {toolkit_name}: {status_emoji} {status} (id={conn_id}, entity={conn_entity})")

                # Accept both ACTIVE and ENABLED status
                if is_active and toolkit_name != "unknown":
                    connected_toolkits.append(toolkit_name)

            if not connected_toolkits:
                self.logger.warning(f"⚠️ No active connections found for user {user_id}")
                return []

            # Get tools for all connected & active toolkits
            # The composio.tools.get() method automatically:
            # 1. Handles toolkit versioning (uses latest compatible versions)
            # 2. Converts tools to LangChain format (via LangchainProvider)
            # 3. Returns only tools for the specified user
            self.logger.info(f"🔄 Fetching tools for toolkits: {', '.join(connected_toolkits)}")

            tools = self.composio.tools.get(
                user_id=user_id,
                toolkits=connected_toolkits
            )

            self.logger.info(f"✅ Successfully loaded {len(tools)} tools for user {user_id}")

            if tools:
                # Log sample tool names for debugging
                tool_names = [
                    tool.name if hasattr(tool, 'name')
                    else getattr(tool, 'slug', str(tool))
                    for tool in tools[:5]
                ]
                self.logger.info(f"📦 Sample tools: {', '.join(tool_names)}")
                if len(tools) > 5:
                    self.logger.info(f"   ... and {len(tools) - 5} more")
            else:
                self.logger.warning(f"⚠️ No tools returned despite having connections")

            return tools

        except Exception as e:
            self.logger.error(f"❌ Error fetching tools for user {user_id}: {e}")
            import traceback
            traceback.print_exc()

            # Return empty list instead of raising exception
            # This allows the app to continue without Composio tools
            return []

    def clear_cache(self, user_id: str = None):
        """
        Clear cache (kept for API compatibility)
        Note: Composio handles caching internally
        """
        if user_id:
            self.logger.info(f"Cache clear requested for user {user_id}")
        else:
            self.logger.info("Cache clear requested for all users")
