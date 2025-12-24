import { useEffect, useRef, useCallback } from 'react';
import { useAgentStore } from '../stores/agentStore';

const WS_URL = `ws://${window.location.hostname}:8000/chat/ws`;
const RESPONSE_TIMEOUT = 30 * 60 * 1000; // 30 minutes - same as backend timeout for large extractions

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number>();
  const responseTimeoutRef = useRef<number>();
  const {
    setConnected,
    addMessage,
    updateStreamingContent,
    clearStreaming,
    updateAgent,
    removeAgent,
    addLog,
    updateTask,
    setProcessing,
  } = useAgentStore();
  
  // Clear any pending timeout
  const clearResponseTimeout = useCallback(() => {
    if (responseTimeoutRef.current) {
      window.clearTimeout(responseTimeoutRef.current);
      responseTimeoutRef.current = undefined;
    }
  }, []);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      console.log('WebSocket connected');
      setConnected(true);
    };

    ws.onclose = (event) => {
      console.log('WebSocket disconnected:', event.code, event.reason);
      setConnected(false);
      setProcessing(false); // Reset processing state on disconnect
      clearResponseTimeout();
      // Reconnect after 3 seconds
      reconnectTimeoutRef.current = window.setTimeout(connect, 3000);
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        // Handle ping/pong keepalive
        if (data.type === 'ping') {
          ws.send(JSON.stringify({ type: 'pong' }));
          return;
        }
        console.log('[WS] Received:', data.type, data.data?.content?.slice?.(0, 100) || '');
        handleMessage(data);
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e);
      }
    };

    wsRef.current = ws;
  }, [setConnected, setProcessing, clearResponseTimeout]);

  const handleMessage = useCallback((data: { type: string; data: Record<string, unknown> }) => {
    try {
      switch (data.type) {
        case 'system:status':
          if (data.data?.client_id) {
            setConnected(true, data.data.client_id as string);
          }
          break;

        case 'chat:message':
          clearResponseTimeout();
          addMessage({
            role: (data.data?.role as 'user' | 'assistant') || 'assistant',
            content: String(data.data?.content || ''),
          });
          setProcessing(false);
          break;

        case 'chat:cancelled':
          clearResponseTimeout();
          addMessage({
            role: 'assistant',
            content: '🛑 Request cancelled by user.',
          });
          setProcessing(false);
          break;

        case 'chat:stream':
          updateStreamingContent(String(data.data?.content || ''));
          break;

        case 'chat:stream_end':
          // Just clear streaming state - the actual message comes via chat:message
          // to avoid duplicates
          clearStreaming();
          break;

        case 'agent:created':
        case 'agent:update':
          if (data.data?.id) {
            updateAgent({
              id: data.data.id as string,
              type: (data.data.type as string) || 'unknown',
              name: (data.data.name as string) || 'Agent',
              status: (data.data.status as Agent['status']) || 'idle',
              summary: data.data.summary as string | undefined,
              progress: data.data.progress as number | undefined,
              metadata: (data.data.metadata as Record<string, unknown>) || {},
            });
          }
          break;

      case 'agent:removed':
        if (data.data?.id) {
          removeAgent(data.data.id as string);
        }
        break;

      case 'log:entry':
        addLog({
          agentId: data.data?.agent_id as string | undefined,
          taskId: data.data?.task_id as string | undefined,
          level: (data.data?.level as LogEntry['level']) || 'info',
          category: (data.data?.category as string) || 'general',
          message: String(data.data?.message || ''),
          details: data.data?.details as Record<string, unknown> | undefined,
        });
        break;

      case 'task:created':
      case 'task:update':
      case 'task:completed':
      case 'task:failed':
        if (data.data?.id) {
          updateTask({
            id: data.data.id as string,
            name: data.data.name as string | undefined,
            status: (data.data.status as Task['status']) || 'pending',
            result: data.data.result as Record<string, unknown> | undefined,
            error: data.data.error as string | undefined,
          });
        }
        break;

      case 'error':
        clearResponseTimeout();
        addLog({
          level: 'error',
          category: 'system',
          message: String(data.data?.message || 'Unknown error'),
        });
        setProcessing(false);
        break;
      }
    } catch (e) {
      console.error('Error handling WebSocket message:', e, data);
      // Don't crash - just log the error
    }
  }, [addMessage, updateStreamingContent, clearStreaming, updateAgent, removeAgent, addLog, updateTask, setConnected, setProcessing, clearResponseTimeout]);

  const sendMessage = useCallback((message: string, context?: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      const sessionId = useAgentStore.getState().activeSessionId;
      setProcessing(true);
      
      // Set a timeout to reset processing state if no response comes
      clearResponseTimeout();
      responseTimeoutRef.current = window.setTimeout(() => {
        console.warn('Response timeout - resetting processing state');
        setProcessing(false);
        addMessage({
          role: 'assistant',
          content: '⏱️ Request timed out. Please try again.',
        });
      }, RESPONSE_TIMEOUT);
      
      wsRef.current.send(JSON.stringify({ message, context, session_id: sessionId }));
      addMessage({ role: 'user', content: message });
    }
  }, [addMessage, setProcessing, clearResponseTimeout]);

  const cancelRequest = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      const sessionId = useAgentStore.getState().activeSessionId;
      wsRef.current.send(JSON.stringify({ action: 'cancel', session_id: sessionId }));
    }
  }, []);

  const disconnect = useCallback(() => {
    clearResponseTimeout();
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    wsRef.current?.close();
    wsRef.current = null;
  }, [clearResponseTimeout]);

  useEffect(() => {
    connect();
    return () => disconnect();
  }, [connect, disconnect]);

  return { sendMessage, cancelRequest, connect, disconnect };
}

// Import types
import type { Agent, LogEntry, Task } from '../stores/agentStore';

