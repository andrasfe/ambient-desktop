import { useEffect, useRef, useCallback } from 'react';
import { useAgentStore } from '../stores/agentStore';

const WS_URL = `ws://${window.location.hostname}:8000/chat/ws`;

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number>();
  const {
    setConnected,
    addMessage,
    updateStreamingContent,
    clearStreaming,
    updateAgent,
    removeAgent,
    addLog,
    updateTask,
  } = useAgentStore();

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      console.log('WebSocket connected');
      setConnected(true);
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setConnected(false);
      // Reconnect after 3 seconds
      reconnectTimeoutRef.current = window.setTimeout(connect, 3000);
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleMessage(data);
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e);
      }
    };

    wsRef.current = ws;
  }, [setConnected]);

  const handleMessage = useCallback((data: { type: string; data: Record<string, unknown> }) => {
    switch (data.type) {
      case 'system:status':
        if (data.data.client_id) {
          setConnected(true, data.data.client_id as string);
        }
        break;

      case 'chat:message':
        addMessage({
          role: data.data.role as 'user' | 'assistant',
          content: data.data.content as string,
        });
        break;

      case 'chat:stream':
        updateStreamingContent(data.data.content as string);
        break;

      case 'chat:stream_end':
        // Just clear streaming state - the actual message comes via chat:message
        // to avoid duplicates
        clearStreaming();
        break;

      case 'agent:created':
      case 'agent:update':
        updateAgent({
          id: data.data.id as string,
          type: data.data.type as string,
          name: data.data.name as string,
          status: data.data.status as Agent['status'],
          summary: data.data.summary as string | undefined,
          progress: data.data.progress as number | undefined,
          metadata: (data.data.metadata as Record<string, unknown>) || {},
        });
        break;

      case 'agent:removed':
        removeAgent(data.data.id as string);
        break;

      case 'log:entry':
        addLog({
          agentId: data.data.agent_id as string | undefined,
          taskId: data.data.task_id as string | undefined,
          level: data.data.level as LogEntry['level'],
          category: data.data.category as string,
          message: data.data.message as string,
          details: data.data.details as Record<string, unknown> | undefined,
        });
        break;

      case 'task:created':
      case 'task:update':
      case 'task:completed':
      case 'task:failed':
        updateTask({
          id: data.data.id as string,
          name: data.data.name as string | undefined,
          status: data.data.status as Task['status'],
          result: data.data.result as Record<string, unknown> | undefined,
          error: data.data.error as string | undefined,
        });
        break;

      case 'error':
        addLog({
          level: 'error',
          category: 'system',
          message: data.data.message as string,
        });
        break;
    }
  }, [addMessage, updateStreamingContent, clearStreaming, updateAgent, removeAgent, addLog, updateTask, setConnected]);

  const sendMessage = useCallback((message: string, context?: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      const sessionId = useAgentStore.getState().activeSessionId;
      wsRef.current.send(JSON.stringify({ message, context, session_id: sessionId }));
      addMessage({ role: 'user', content: message });
    }
  }, [addMessage]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    wsRef.current?.close();
    wsRef.current = null;
  }, []);

  useEffect(() => {
    connect();
    return () => disconnect();
  }, [connect, disconnect]);

  return { sendMessage, connect, disconnect };
}

// Import types
import type { Agent, LogEntry, Task } from '../stores/agentStore';

