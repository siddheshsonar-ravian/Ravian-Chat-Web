"""
LangGraph WebSocket Chat Application
Main FastAPI application - compatible with Langraph WebSocket protocol
"""

import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

# Load .env BEFORE importing modules that read env vars at import time
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware import Middleware
from fastapi.responses import HTMLResponse
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse

import ws
from dependencies import init_managers, cleanup_managers
from routes.chat_routes import router

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*']
    )
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for the FastAPI application
    Handles startup and shutdown events
    """
    os.environ["HOME"] = os.path.expanduser("~")
    try:
        await init_managers()
    except Exception as e:
        print(f"Error initializing managers: {e}")
        raise

    yield  # App is running and handling requests here

    # Shutdown: Close connections and cleanup
    try:
        await cleanup_managers()
    except Exception as e:
        print(f"Error during cleanup: {e}")
        raise


app = FastAPI(
    title="LangGraph WebSocket Chat",
    description="WebSocket chat using LangGraph - compatible with Langraph protocol",
    version="1.0.0",
    redoc_url=None,
    lifespan=lifespan,
    middleware=middleware
)

app.include_router(router)

app.include_router(
    ws.router,
    prefix="/ws",
    tags=["websocket"],
    responses={404: {"description": "Not found"}},
)


@app.get(path='/docs', include_in_schema=False)
async def docs_redirect():
    return RedirectResponse(url="/redoc")


@app.get("/", response_class=HTMLResponse)
async def serve_html():
    """
    Serve the HTML WebSocket client UI
    (You can copy the index.html from the Langraph project)
    """
    try:
        html_content = Path("templates/index.html").read_text(encoding="utf-8")
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(
            content="""
            <html>
                <head><title>LangGraph WebSocket Chat</title></head>
                <body>
                    <h1>LangGraph WebSocket Chat</h1>
                    <p>WebSocket endpoint: ws://localhost:8000/ws/chat/{chat_id}</p>
                    <p>To use the UI, copy the index.html from the Langraph project to templates/index.html</p>
                    <h2>API Endpoints:</h2>
                    <ul>
                        <li>POST /chats/new_chat - Create new chat session</li>
                        <li>GET /health - Health check</li>
                        <li>WS /ws/chat/{chat_id} - WebSocket connection</li>
                    </ul>
                </body>
            </html>
            """
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "message": "LangGraph service is running"}


@app.post("/chats/new_chat")
async def new_chat():
    """Create a new chat session"""
    return {"status": "ok", "chat_id": str(uuid.uuid4())}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
