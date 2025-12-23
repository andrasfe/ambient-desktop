"""Chat API endpoints with WebSocket support."""

import uuid
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel

from ..services.websocket import ws_manager, EventType
from ..agents.graph import orchestrator

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatMessage(BaseModel):
    """A chat message."""
    message: str
    session_id: Optional[str] = "default"


class ChatResponse(BaseModel):
    """A chat response."""
    response: str
    session_id: str


@router.get("/history")
async def get_history():
    """Get chat history.

    Note: History persistence is handled by LangGraph checkpointer; this endpoint currently
    returns an empty placeholder for UI/testing.
    """
    return {"messages": [], "length": 0}


@router.websocket("/ws")
async def chat_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time chat."""
    client_id = str(uuid.uuid4())
    session_id = f"session-{client_id}"
    await ws_manager.connect(websocket, client_id)
    
    try:
        # Send connection confirmation
        await ws_manager.send_event(
            client_id,
            EventType.SYSTEM_STATUS,
            {"status": "connected", "client_id": client_id, "session_id": session_id},
        )
        
        while True:
            # Receive message
            data = await websocket.receive_json()
            message = data.get("message", "")
            msg_session = data.get("session_id", session_id)
            
            if not message:
                continue
            
            # Note: Don't broadcast user message here - frontend already adds it locally
            # This prevents duplicate messages in the chat
            
            # Process message through LangGraph orchestrator
            try:
                import asyncio
                import traceback as tb
                print(f"[CHAT-WS] ===== NEW MESSAGE =====")
                print(f"[CHAT-WS] Client: {client_id}")
                print(f"[CHAT-WS] Session: {msg_session}")
                print(f"[CHAT-WS] Message: {message[:100]}...")
                print(f"[CHAT-WS] Calling orchestrator.process_message...")
                
                # Process with timeout (5 minutes max for complex browser tasks)
                try:
                    full_response = await asyncio.wait_for(
                        orchestrator.process_message(message, msg_session),
                        timeout=300.0  # 5 minute timeout
                    )
                    print(f"[CHAT-WS] ✅ Got response: {len(full_response)} chars")
                except asyncio.TimeoutError:
                    full_response = "⏱️ Request timed out after 5 minutes. The operation may still be running in the background. Please try again with a simpler request."
                    print(f"[CHAT-WS] ⏱️ TIMEOUT after 300 seconds")
                except Exception as inner_e:
                    print(f"[CHAT-WS] ❌ Exception in process_message: {inner_e}")
                    print(f"[CHAT-WS] Traceback: {tb.format_exc()}")
                    full_response = f"Error processing message: {str(inner_e)}"
                
                print(f"[CHAT-WS] Response preview: {full_response[:200]}...")
                # Note: Response is already broadcast from summarize_node, no need to send again
                
            except Exception as e:
                import traceback
                error_msg = f"Chat error: {str(e)}\n{traceback.format_exc()}"
                print(f"[CHAT] ERROR: {error_msg}")
                await ws_manager.send_event(
                    client_id,
                    EventType.ERROR,
                    {"message": str(e)},
                )
                await ws_manager.send_event(
                    client_id,
                    EventType.CHAT_MESSAGE,
                    {
                        "role": "assistant", 
                        "content": f"Error: {str(e)}",
                        "session_id": msg_session,
                    },
                )
    except WebSocketDisconnect:
        await ws_manager.disconnect(client_id)


@router.post("/message", response_model=ChatResponse)
async def send_message(message: ChatMessage):
    """Send a chat message and get a response."""
    try:
        response = await orchestrator.process_message(
            message.message,
            message.session_id or "default",
        )
        return ChatResponse(
            response=response,
            session_id=message.session_id or "default",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions")
async def list_sessions():
    """List active chat sessions."""
    return {
        "sessions": list(orchestrator.sessions.keys()),
        "count": len(orchestrator.sessions),
    }


@router.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear a chat session."""
    if session_id in orchestrator.sessions:
        del orchestrator.sessions[session_id]
        return {"cleared": True, "session_id": session_id}
    raise HTTPException(status_code=404, detail="Session not found")
