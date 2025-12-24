"""Chat API endpoints with WebSocket support."""

import asyncio
import uuid
from typing import Optional, Dict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel

from ..services.websocket import ws_manager, EventType
from ..agents.graph import orchestrator

router = APIRouter(prefix="/chat", tags=["chat"])

# Track active requests for cancellation
active_requests: Dict[str, asyncio.Event] = {}  # session_id -> cancel_event
# Track active processing tasks
active_tasks: Dict[str, asyncio.Task] = {}  # session_id -> task


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


async def cleanup_all_agents():
    """Broadcast removal of all agents to clean up the UI."""
    # Import here to avoid circular imports
    from ..database import get_session
    from ..models.agent import Agent, AgentStatus
    
    try:
        async with get_session() as session:
            from sqlalchemy import select
            result = await session.execute(
                select(Agent).where(Agent.status.in_([AgentStatus.BUSY, AgentStatus.IDLE]))
            )
            agents = result.scalars().all()
            
            for agent in agents:
                # Mark as stopped in DB
                agent.status = AgentStatus.STOPPED
                # Broadcast removal to UI
                await ws_manager.broadcast(
                    EventType.AGENT_REMOVED,
                    {"id": str(agent.id)},
                )
            
            await session.commit()
            print(f"[CLEANUP] Stopped {len(agents)} agents")
    except Exception as e:
        print(f"[CLEANUP] Error cleaning up agents: {e}")


async def process_and_respond(
    client_id: str,
    message: str,
    msg_session: str,
    cancel_event: asyncio.Event,
):
    """Process a message and send the response."""
    import traceback as tb
    
    was_cancelled = False
    
    try:
        print(f"[CHAT-WS] ===== NEW MESSAGE =====")
        print(f"[CHAT-WS] Client: {client_id}")
        print(f"[CHAT-WS] Session: {msg_session}")
        print(f"[CHAT-WS] Message: {message[:100]}...")
        print(f"[CHAT-WS] Calling orchestrator.process_message...")
        
        # Process with timeout (5 minutes max for complex browser tasks)
        try:
            full_response = await asyncio.wait_for(
                orchestrator.process_message(message, msg_session, cancel_event),
                timeout=300.0  # 5 minute timeout
            )
            
            if cancel_event.is_set():
                print(f"[CHAT-WS] 🛑 Request was cancelled")
                was_cancelled = True
                return
            
            print(f"[CHAT-WS] ✅ Got response: {len(full_response)} chars")
        except asyncio.TimeoutError:
            full_response = "⏱️ Request timed out after 5 minutes. Please try again with a simpler request."
            print(f"[CHAT-WS] ⏱️ TIMEOUT after 300 seconds")
        except asyncio.CancelledError:
            print(f"[CHAT-WS] 🛑 Request was cancelled (CancelledError)")
            was_cancelled = True
            return
        except Exception as inner_e:
            print(f"[CHAT-WS] ❌ Exception in process_message: {inner_e}")
            print(f"[CHAT-WS] Traceback: {tb.format_exc()}")
            full_response = f"Error processing message: {str(inner_e)}"
        
        if cancel_event.is_set():
            was_cancelled = True
            return
        
        print(f"[CHAT-WS] Response preview: {full_response[:200]}...")
        if full_response and not full_response.startswith("{"):
            try:
                await ws_manager.send_event(
                    client_id,
                    EventType.CHAT_MESSAGE,
                    {
                        "role": "assistant",
                        "content": full_response,
                        "session_id": msg_session,
                    },
                )
                print(f"[CHAT-WS] ✅ Message sent")
            except Exception as send_err:
                print(f"[CHAT-WS] Send failed: {send_err}")
                
    except Exception as e:
        import traceback
        error_msg = f"Chat error: {str(e)}\n{traceback.format_exc()}"
        print(f"[CHAT] ERROR: {error_msg}")
        try:
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
        except Exception:
            pass
    finally:
        # Clean up
        if msg_session in active_requests:
            del active_requests[msg_session]
        
        # If cancelled, make sure all agents are stopped
        if was_cancelled or cancel_event.is_set():
            await cleanup_all_agents()
        if msg_session in active_tasks:
            del active_tasks[msg_session]


@router.websocket("/ws")
async def chat_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time chat."""
    client_id = str(uuid.uuid4())
    session_id = f"session-{client_id}"
    await ws_manager.connect(websocket, client_id)
    
    async def keepalive():
        """Send periodic pings to keep the connection alive."""
        while True:
            try:
                await asyncio.sleep(30)  # Ping every 30 seconds
                await websocket.send_json({"type": "ping"})
            except Exception:
                break
    
    keepalive_task = asyncio.create_task(keepalive())
    
    try:
        # Send connection confirmation
        await ws_manager.send_event(
            client_id,
            EventType.SYSTEM_STATUS,
            {"status": "connected", "client_id": client_id, "session_id": session_id},
        )
        
        while True:
            # Receive message with timeout to allow keepalive checks
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=60)
            except asyncio.TimeoutError:
                # No message received, continue to allow keepalive
                continue
            
            # Handle ping response
            if data.get("type") == "pong":
                continue
            
            # Handle cancel action
            action = data.get("action")
            if action == "cancel":
                cancel_session = data.get("session_id", session_id)
                print(f"[CHAT-WS] 🛑 Received cancel for session: {cancel_session}")
                if cancel_session in active_requests:
                    print(f"[CHAT-WS] 🛑 Cancelling request...")
                    active_requests[cancel_session].set()
                    orchestrator.cancel_session(cancel_session)
                    # Cancel the task if it exists
                    if cancel_session in active_tasks:
                        active_tasks[cancel_session].cancel()
                    await ws_manager.send_event(
                        client_id,
                        EventType.CHAT_CANCELLED,
                        {"session_id": cancel_session},
                    )
                continue
            
            message = data.get("message", "")
            msg_session = data.get("session_id", session_id)
            
            if not message:
                continue
            
            # Create cancellation event for this request
            cancel_event = asyncio.Event()
            active_requests[msg_session] = cancel_event
            
            # Process in background task so we can still receive cancel messages
            task = asyncio.create_task(
                process_and_respond(client_id, message, msg_session, cancel_event)
            )
            active_tasks[msg_session] = task
            
    except WebSocketDisconnect:
        print(f"[CHAT-WS] Client {client_id} disconnected")
    finally:
        keepalive_task.cancel()
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
