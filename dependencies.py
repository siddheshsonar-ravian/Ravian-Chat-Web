"""
Dependency injection for FastAPI application
Manages lifecycle of managers
"""
from managers.composio_manager import ComposioManager
from managers.connection import WebSocketManager

# Global manager instances
_websocket_manager = None
_composio_manager = None


async def init_managers() -> None:
    """Initialize all manager instances"""
    global _websocket_manager, _composio_manager
    _websocket_manager = WebSocketManager()
    print("Managers initialized successfully")


async def cleanup_managers() -> None:
    """Cleanup all manager instances"""
    global _websocket_manager, _composio_manager
    if _websocket_manager:
        await _websocket_manager.cleanup()
        _websocket_manager = None
    _composio_manager = None
    print("Managers cleaned up successfully")


def get_websocket_manager() -> WebSocketManager:
    """
    Dependency to get the WebSocket manager instance

    Returns:
        WebSocketManager: Global WebSocket manager instance
    """
    global _websocket_manager
    if _websocket_manager is None:
        raise RuntimeError("WebSocket manager not initialized. Call init_managers() first.")
    return _websocket_manager


def get_composio_manager() -> ComposioManager:
    """
    Get or create the Composio manager singleton

    Returns:
        ComposioManager instance
    """
    global _composio_manager
    if _composio_manager is None:
        _composio_manager = ComposioManager()
    return _composio_manager
