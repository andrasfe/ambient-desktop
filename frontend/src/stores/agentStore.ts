import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  streaming?: boolean;
}

export interface ChatSession {
  id: string;
  name: string;
  messages: Message[];
  createdAt: Date;
  updatedAt: Date;
}

export interface Agent {
  id: string;
  type: string;
  name: string;
  status: 'idle' | 'busy' | 'error' | 'stopped';
  summary?: string;
  progress?: number;
  currentTaskId?: string;
  metadata: Record<string, unknown>;
}

export interface LogEntry {
  id: string;
  agentId?: string;
  taskId?: string;
  level: 'debug' | 'info' | 'warn' | 'error';
  category: string;
  message: string;
  details?: Record<string, unknown>;
  timestamp: Date;
}

export interface Task {
  id: string;
  name: string;
  agentType: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  priority: number;
  description?: string;
  result?: Record<string, unknown>;
  error?: string;
  createdAt: Date;
}

interface AgentStore {
  // Connection
  connected: boolean;
  clientId: string | null;
  setConnected: (connected: boolean, clientId?: string) => void;

  // Sessions
  sessions: ChatSession[];
  activeSessionId: string | null;
  createSession: (name?: string) => string;
  switchSession: (sessionId: string) => void;
  renameSession: (sessionId: string, name: string) => void;
  deleteSession: (sessionId: string) => void;
  getActiveSession: () => ChatSession | null;

  // Messages (for active session)
  messages: Message[];
  streamingContent: string;
  addMessage: (message: Omit<Message, 'id' | 'timestamp'>) => void;
  updateStreamingContent: (content: string) => void;
  clearStreaming: () => void;
  clearMessages: () => void;

  // Agents
  agents: Agent[];
  setAgents: (agents: Agent[]) => void;
  updateAgent: (agent: Partial<Agent> & { id: string }) => void;
  removeAgent: (id: string) => void;

  // Logs
  logs: LogEntry[];
  addLog: (log: Omit<LogEntry, 'id' | 'timestamp'>) => void;
  clearLogs: () => void;

  // Tasks
  tasks: Task[];
  setTasks: (tasks: Task[]) => void;
  updateTask: (task: Partial<Task> & { id: string }) => void;
}

// Helper to create a new session
const createNewSession = (name?: string): ChatSession => ({
  id: crypto.randomUUID(),
  name: name || `Chat ${new Date().toLocaleDateString()}`,
  messages: [],
  createdAt: new Date(),
  updatedAt: new Date(),
});

export const useAgentStore = create<AgentStore>()(
  persist(
    (set, get) => ({
      // Connection
      connected: false,
      clientId: null,
      setConnected: (connected, clientId) => set({ connected, clientId: clientId || null }),

      // Sessions
      sessions: [],
      activeSessionId: null,
      
      createSession: (name) => {
        const newSession = createNewSession(name);
        set((state) => ({
          sessions: [newSession, ...state.sessions],
          activeSessionId: newSession.id,
          messages: [],
          streamingContent: '',
        }));
        return newSession.id;
      },
      
      switchSession: (sessionId) => {
        const state = get();
        const session = state.sessions.find((s) => s.id === sessionId);
        if (session) {
          // Save current messages to current session before switching
          if (state.activeSessionId) {
            set((s) => ({
              sessions: s.sessions.map((sess) =>
                sess.id === s.activeSessionId
                  ? { ...sess, messages: s.messages, updatedAt: new Date() }
                  : sess
              ),
            }));
          }
          set({
            activeSessionId: sessionId,
            messages: session.messages,
            streamingContent: '',
          });
        }
      },
      
      renameSession: (sessionId, name) => {
        set((state) => ({
          sessions: state.sessions.map((s) =>
            s.id === sessionId ? { ...s, name, updatedAt: new Date() } : s
          ),
        }));
      },
      
      deleteSession: (sessionId) => {
        set((state) => {
          const newSessions = state.sessions.filter((s) => s.id !== sessionId);
          const wasActive = state.activeSessionId === sessionId;
          
          if (wasActive) {
            // Switch to another session or create new one
            const nextSession = newSessions[0];
            if (nextSession) {
              return {
                sessions: newSessions,
                activeSessionId: nextSession.id,
                messages: nextSession.messages,
              };
            } else {
              // No sessions left, create a new one
              const newSession = createNewSession();
              return {
                sessions: [newSession],
                activeSessionId: newSession.id,
                messages: [],
              };
            }
          }
          return { sessions: newSessions };
        });
      },
      
      getActiveSession: () => {
        const state = get();
        return state.sessions.find((s) => s.id === state.activeSessionId) || null;
      },

      // Messages (for active session)
      messages: [],
      streamingContent: '',
      addMessage: (message) => set((state) => {
        const newMessage = {
          ...message,
          id: crypto.randomUUID(),
          timestamp: new Date(),
        };
        const newMessages = [...state.messages, newMessage];
        
        // If no active session, create one
        if (!state.activeSessionId || !state.sessions.find((s) => s.id === state.activeSessionId)) {
          const newSession = {
            id: crypto.randomUUID(),
            name: `Chat ${new Date().toLocaleDateString()}`,
            messages: newMessages,
            createdAt: new Date(),
            updatedAt: new Date(),
          };
          return {
            messages: newMessages,
            sessions: [newSession, ...state.sessions],
            activeSessionId: newSession.id,
          };
        }
        
        // Update the active session
        return {
          messages: newMessages,
          sessions: state.sessions.map((s) =>
            s.id === state.activeSessionId
              ? { ...s, messages: newMessages, updatedAt: new Date() }
              : s
          ),
        };
      }),
      updateStreamingContent: (content) => set((state) => ({
        streamingContent: state.streamingContent + content,
      })),
      clearStreaming: () => set({ streamingContent: '' }),
      clearMessages: () => set((state) => ({
        messages: [],
        sessions: state.sessions.map((s) =>
          s.id === state.activeSessionId
            ? { ...s, messages: [], updatedAt: new Date() }
            : s
        ),
      })),

      // Agents
      agents: [],
      setAgents: (agents) => set({ agents }),
      updateAgent: (agent) => set((state) => {
        const exists = state.agents.some((a) => a.id === agent.id);
        if (exists) {
          // Update existing agent
          return {
            agents: state.agents.map((a) =>
              a.id === agent.id ? { ...a, ...agent } : a
            ),
          };
        } else {
          // Add new agent
          return {
            agents: [
              ...state.agents,
              {
                id: agent.id,
                type: agent.type || 'custom',
                name: agent.name || 'Unknown Agent',
                status: agent.status || 'idle',
                metadata: agent.metadata || {},
                ...agent,
              } as Agent,
            ],
          };
        }
      }),
      removeAgent: (id) => set((state) => ({
        agents: state.agents.filter((a) => a.id !== id),
      })),

      // Logs
      logs: [],
      addLog: (log) => set((state) => ({
        logs: [
          {
            ...log,
            id: crypto.randomUUID(),
            timestamp: new Date(),
          },
          ...state.logs,
        ].slice(0, 500), // Keep last 500 logs
      })),
      clearLogs: () => set({ logs: [] }),

      // Tasks
      tasks: [],
      setTasks: (tasks) => set({ tasks }),
      updateTask: (task) => set((state) => ({
        tasks: state.tasks.map((t) =>
          t.id === task.id ? { ...t, ...task } : t
        ),
      })),
    }),
    {
      name: 'ambient-agent-storage',
      // Persist sessions and logs
      partialize: (state) => ({
        sessions: state.sessions,
        activeSessionId: state.activeSessionId,
        messages: state.messages,
        logs: state.logs.slice(0, 100),
      }),
      // Deserialize dates back to Date objects
      onRehydrateStorage: () => (state) => {
        if (state) {
          state.sessions = state.sessions.map((session) => ({
            ...session,
            createdAt: new Date(session.createdAt),
            updatedAt: new Date(session.updatedAt),
            messages: session.messages.map((msg) => ({
              ...msg,
              timestamp: new Date(msg.timestamp),
            })),
          }));
          state.messages = state.messages.map((msg) => ({
            ...msg,
            timestamp: new Date(msg.timestamp),
          }));
          state.logs = state.logs.map((log) => ({
            ...log,
            timestamp: new Date(log.timestamp),
          }));
          
          // Ensure there's always an active session
          if (!state.activeSessionId || !state.sessions.find(s => s.id === state.activeSessionId)) {
            if (state.sessions.length > 0) {
              state.activeSessionId = state.sessions[0].id;
              state.messages = state.sessions[0].messages;
            } else {
              // No sessions at all, create a default one
              const newSession = {
                id: crypto.randomUUID(),
                name: `Chat ${new Date().toLocaleDateString()}`,
                messages: [],
                createdAt: new Date(),
                updatedAt: new Date(),
              };
              state.sessions = [newSession];
              state.activeSessionId = newSession.id;
              state.messages = [];
            }
          }
        }
      },
    }
  )
);

