import { useEffect } from 'react';
import { useWebSocket } from './hooks/useWebSocket';
import { useAgentStore } from './stores/agentStore';
import { Chat } from './components/Chat';
import { ChatSidebar } from './components/ChatSidebar';
import { ActivityLog } from './components/ActivityLog';
import { AgentPanel } from './components/AgentPanel';
import { Header } from './components/Header';
import { StatusBar } from './components/StatusBar';

function App() {
  const { sendMessage, cancelRequest } = useWebSocket();
  const connected = useAgentStore((state) => state.connected);
  const sessions = useAgentStore((state) => state.sessions);
  const activeSessionId = useAgentStore((state) => state.activeSessionId);
  const createSession = useAgentStore((state) => state.createSession);

  // Ensure there's always at least one session
  useEffect(() => {
    if (sessions.length === 0) {
      createSession('New Chat');
    } else if (!activeSessionId) {
      useAgentStore.getState().switchSession(sessions[0].id);
    }
  }, [sessions, activeSessionId, createSession]);

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      <Header />
      
      <main className="flex-1 flex overflow-hidden">
        {/* Sidebar - Session List */}
        <div className="w-64 shrink-0 border-r border-void-800">
          <ChatSidebar />
        </div>

        {/* Center Panel - Chat */}
        <div className="flex-1 flex flex-col border-r border-void-800 min-w-0">
          <Chat onSendMessage={sendMessage} onCancelRequest={cancelRequest} />
        </div>

        {/* Right Panel - Logs & Agents */}
        <div className="w-80 shrink-0 flex flex-col">
          {/* Agent Status */}
          <div className="h-1/3 border-b border-void-800 overflow-hidden">
            <AgentPanel />
          </div>
          
          {/* Activity Log */}
          <div className="flex-1 overflow-hidden">
            <ActivityLog />
          </div>
        </div>
      </main>

      <StatusBar connected={connected} />
    </div>
  );
}

export default App;

