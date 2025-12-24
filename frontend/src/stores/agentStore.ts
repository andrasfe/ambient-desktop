import { create } from 'zustand';
import { persist } from 'zustand/middleware';

// UUID fallback for browsers that don't support crypto.randomUUID
function generateUUID(): string {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  // Fallback for older browsers
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

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

  // Processing state
  isProcessing: boolean;
  setProcessing: (processing: boolean) => void;

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
  id: generateUUID(),
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

      // Processing state
      isProcessing: false,
      setProcessing: (processing) => set({ isProcessing: processing }),

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
        try {
          // Sanitize content - limit length to prevent localStorage issues
          const MAX_MESSAGE_LENGTH = 50000;
          const sanitizedContent = typeof message.content === 'string' 
            ? message.content.slice(0, MAX_MESSAGE_LENGTH)
            : String(message.content || '');
          
          const newMessage = {
            ...message,
            content: sanitizedContent,
            id: generateUUID(),
            timestamp: new Date(),
          };
          const newMessages = [...state.messages, newMessage];
          
          // If no active session, create one
          if (!state.activeSessionId || !state.sessions.find((s) => s.id === state.activeSessionId)) {
            const newSession = {
              id: generateUUID(),
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
        } catch (e) {
          console.error('Error adding message:', e);
          // Return unchanged state on error
          return state;
        }
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
                type: agent.type || 'custom',
                name: agent.name || 'Unknown Agent',
                status: agent.status || 'idle',
                metadata: agent.metadata || {},
                ...agent,  // id comes from agent
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
            id: generateUUID(),
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
      // Custom storage that handles errors gracefully
      storage: {
        getItem: (name) => {
          try {
            const value = localStorage.getItem(name);
            return value ? JSON.parse(value) : null;
          } catch (e) {
            console.error('Error reading from localStorage:', e);
            return null;
          }
        },
        setItem: (name, value) => {
          try {
            localStorage.setItem(name, JSON.stringify(value));
          } catch (e) {
            console.error('Error writing to localStorage:', e);
            // If storage is full, try to clear old data
            if (e instanceof DOMException && e.name === 'QuotaExceededError') {
              try {
                localStorage.removeItem(name);
                console.warn('Cleared localStorage due to quota exceeded');
              } catch {
                // Ignore
              }
            }
          }
        },
        removeItem: (name) => {
          try {
            localStorage.removeItem(name);
          } catch (e) {
            console.error('Error removing from localStorage:', e);
          }
        },
      },
      // Persist sessions and logs
      partialize: (state) => ({
        sessions: state.sessions.slice(0, 10), // Keep only last 10 sessions
        activeSessionId: state.activeSessionId,
        messages: state.messages.slice(-100), // Keep only last 100 messages
        logs: state.logs.slice(0, 100),
      }),
      // Deserialize dates back to Date objects
      onRehydrateStorage: () => (state) => {
        if (state) {
          try {
            state.sessions = (state.sessions || []).map((session) => ({
              ...session,
              createdAt: new Date(session.createdAt),
              updatedAt: new Date(session.updatedAt),
              messages: (session.messages || []).map((msg) => ({
                ...msg,
                timestamp: new Date(msg.timestamp),
              })),
            }));
            state.messages = (state.messages || []).map((msg) => ({
              ...msg,
              timestamp: new Date(msg.timestamp),
            }));
            state.logs = (state.logs || []).map((log) => ({
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
                  id: generateUUID(),
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
          } catch (e) {
            console.error('Failed to rehydrate state, resetting to defaults:', e);
            // Don't reload - just reset to safe defaults
            state.sessions = [];
            state.activeSessionId = null;
            state.messages = [];
            state.logs = [];
            // Clear corrupted storage
            try {
              localStorage.removeItem('ambient-agent-storage');
            } catch (storageErr) {
              console.error('Failed to clear storage:', storageErr);
            }
          }
        }
      },
    }
  )
);

