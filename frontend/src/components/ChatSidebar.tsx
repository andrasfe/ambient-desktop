import { useState } from 'react';
import { Plus, MessageSquare, Trash2, Edit2, Check, X, AlertTriangle } from 'lucide-react';
import { useAgentStore, ChatSession } from '../stores/agentStore';
import clsx from 'clsx';

export function ChatSidebar() {
  const sessions = useAgentStore((state) => state.sessions);
  const activeSessionId = useAgentStore((state) => state.activeSessionId);
  const createSession = useAgentStore((state) => state.createSession);
  const switchSession = useAgentStore((state) => state.switchSession);
  const renameSession = useAgentStore((state) => state.renameSession);
  const deleteSession = useAgentStore((state) => state.deleteSession);

  const [editingId, setEditingId] = useState<string | null>(null);
  const [editName, setEditName] = useState('');
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);

  const handleNewChat = () => {
    createSession();
  };

  const handleStartRename = (session: ChatSession) => {
    setEditingId(session.id);
    setEditName(session.name);
  };

  const handleSaveRename = () => {
    if (editingId && editName.trim()) {
      renameSession(editingId, editName.trim());
    }
    setEditingId(null);
    setEditName('');
  };

  const handleCancelRename = () => {
    setEditingId(null);
    setEditName('');
  };

  const handleDelete = (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setDeleteConfirmId(sessionId);
  };

  const confirmDelete = () => {
    if (deleteConfirmId) {
      deleteSession(deleteConfirmId);
      setDeleteConfirmId(null);
    }
  };

  const cancelDelete = () => {
    setDeleteConfirmId(null);
  };

  // Group sessions by date
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);
  const lastWeek = new Date(today);
  lastWeek.setDate(lastWeek.getDate() - 7);

  const groupedSessions = {
    today: sessions.filter((s) => new Date(s.updatedAt) >= today),
    yesterday: sessions.filter((s) => {
      const d = new Date(s.updatedAt);
      return d >= yesterday && d < today;
    }),
    lastWeek: sessions.filter((s) => {
      const d = new Date(s.updatedAt);
      return d >= lastWeek && d < yesterday;
    }),
    older: sessions.filter((s) => new Date(s.updatedAt) < lastWeek),
  };

  const renderSessionItem = (session: ChatSession) => {
    const isActive = session.id === activeSessionId;
    const isEditing = editingId === session.id;
    const preview = session.messages[0]?.content?.slice(0, 30) || 'New conversation';

    return (
      <div
        key={session.id}
        onClick={() => !isEditing && switchSession(session.id)}
        className={clsx(
          'group flex items-center gap-2 px-3 py-2.5 rounded-lg cursor-pointer transition-all',
          isActive
            ? 'bg-neon-cyan/10 border border-neon-cyan/30'
            : 'hover:bg-void-800/50 border border-transparent'
        )}
      >
        <MessageSquare className={clsx(
          'w-4 h-4 shrink-0',
          isActive ? 'text-neon-cyan' : 'text-void-500'
        )} />
        
        <div className="flex-1 min-w-0">
          {isEditing ? (
            <div className="flex items-center gap-1">
              <input
                type="text"
                value={editName}
                onChange={(e) => setEditName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') handleSaveRename();
                  if (e.key === 'Escape') handleCancelRename();
                }}
                className="flex-1 bg-void-900 border border-void-700 rounded px-2 py-0.5 text-sm focus:border-neon-cyan focus:outline-none"
                autoFocus
                onClick={(e) => e.stopPropagation()}
              />
              <button
                onClick={(e) => { e.stopPropagation(); handleSaveRename(); }}
                className="p-1 text-neon-cyan hover:bg-neon-cyan/20 rounded"
              >
                <Check className="w-3 h-3" />
              </button>
              <button
                onClick={(e) => { e.stopPropagation(); handleCancelRename(); }}
                className="p-1 text-void-400 hover:bg-void-700 rounded"
              >
                <X className="w-3 h-3" />
              </button>
            </div>
          ) : (
            <>
              <p className={clsx(
                'text-sm font-medium truncate',
                isActive ? 'text-void-100' : 'text-void-300'
              )}>
                {session.name}
              </p>
              <p className="text-xs text-void-500 truncate">
                {preview}
              </p>
            </>
          )}
        </div>

        {!isEditing && (
          <div className="flex items-center gap-0.5 shrink-0">
            <button
              onClick={(e) => { e.stopPropagation(); handleStartRename(session); }}
              className="p-1.5 text-void-500 hover:text-void-200 hover:bg-void-700 rounded opacity-0 group-hover:opacity-100 transition-opacity"
              title="Rename"
            >
              <Edit2 className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={(e) => handleDelete(session.id, e)}
              className="p-1.5 text-void-500 hover:text-red-400 hover:bg-red-400/10 rounded"
              title="Delete chat"
            >
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          </div>
        )}
      </div>
    );
  };

  const renderSection = (title: string, items: ChatSession[]) => {
    if (items.length === 0) return null;
    return (
      <div className="mb-4">
        <h3 className="text-xs font-medium text-void-500 uppercase tracking-wider px-3 mb-2">
          {title}
        </h3>
        <div className="space-y-1">
          {items.map(renderSessionItem)}
        </div>
      </div>
    );
  };

  const sessionToDelete = sessions.find((s) => s.id === deleteConfirmId);

  return (
    <div className="h-full flex flex-col bg-void-950 relative">
      {/* Delete Confirmation Modal */}
      {deleteConfirmId && (
        <div className="absolute inset-0 bg-void-950/90 z-10 flex items-center justify-center p-4">
          <div className="bg-void-900 border border-void-700 rounded-lg p-4 max-w-xs w-full shadow-xl">
            <div className="flex items-center gap-2 text-red-400 mb-3">
              <AlertTriangle className="w-5 h-5" />
              <span className="font-medium">Delete Chat?</span>
            </div>
            <p className="text-sm text-void-300 mb-4">
              Delete "<span className="text-void-100">{sessionToDelete?.name}</span>"? 
              This cannot be undone.
            </p>
            <div className="flex gap-2">
              <button
                onClick={cancelDelete}
                className="flex-1 px-3 py-2 text-sm bg-void-800 hover:bg-void-700 rounded-lg transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={confirmDelete}
                className="flex-1 px-3 py-2 text-sm bg-red-500/20 hover:bg-red-500/30 text-red-400 border border-red-500/30 rounded-lg transition-colors"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}

      {/* New Chat Button */}
      <div className="p-3 border-b border-void-800">
        <button
          onClick={handleNewChat}
          className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-neon-cyan/10 hover:bg-neon-cyan/20 border border-neon-cyan/30 rounded-lg text-neon-cyan font-medium transition-all"
        >
          <Plus className="w-4 h-4" />
          New Chat
        </button>
      </div>

      {/* Sessions List */}
      <div className="flex-1 overflow-y-auto p-2">
        {sessions.length === 0 ? (
          <div className="text-center text-void-500 py-8">
            <MessageSquare className="w-8 h-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">No conversations yet</p>
            <p className="text-xs mt-1">Click "New Chat" to start</p>
          </div>
        ) : (
          <>
            {renderSection('Today', groupedSessions.today)}
            {renderSection('Yesterday', groupedSessions.yesterday)}
            {renderSection('Last 7 Days', groupedSessions.lastWeek)}
            {renderSection('Older', groupedSessions.older)}
          </>
        )}
      </div>

      {/* Footer */}
      <div className="p-3 border-t border-void-800 text-xs text-void-500 text-center">
        {sessions.length} conversation{sessions.length !== 1 ? 's' : ''}
      </div>
    </div>
  );
}

