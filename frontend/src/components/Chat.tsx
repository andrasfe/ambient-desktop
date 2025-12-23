import { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Loader2, Copy, Check } from 'lucide-react';
import { useAgentStore } from '../stores/agentStore';
import clsx from 'clsx';

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  return (
    <button
      onClick={handleCopy}
      className="opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded hover:bg-void-700 text-void-500 hover:text-void-300"
      title="Copy message"
    >
      {copied ? (
        <Check className="w-3.5 h-3.5 text-neon-cyan" />
      ) : (
        <Copy className="w-3.5 h-3.5" />
      )}
    </button>
  );
}

interface ChatProps {
  onSendMessage: (message: string) => void;
}

export function Chat({ onSendMessage }: ChatProps) {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messages = useAgentStore((state) => state.messages);
  const streamingContent = useAgentStore((state) => state.streamingContent);
  const connected = useAgentStore((state) => state.connected);
  const activeSession = useAgentStore((state) => 
    state.sessions.find((s) => s.id === state.activeSessionId)
  );

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, streamingContent]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || !connected) return;
    
    onSendMessage(input.trim());
    setInput('');
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-4 py-3 border-b border-void-800 glass">
        <h2 className="font-display font-semibold text-void-200 flex items-center gap-2">
          <Bot className="w-5 h-5 text-neon-cyan" />
          <span className="truncate">{activeSession?.name || 'Agent Chat'}</span>
        </h2>
        {activeSession && (
          <p className="text-xs text-void-500 mt-0.5">
            {activeSession.messages.length} message{activeSession.messages.length !== 1 ? 's' : ''}
          </p>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && !streamingContent && (
          <div className="flex flex-col items-center justify-center h-full text-void-500">
            <Bot className="w-16 h-16 mb-4 opacity-30" />
            <p className="text-center font-mono text-sm">
              Start a conversation to control your computer.
              <br />
              I can browse the web, manage files, and more.
            </p>
          </div>
        )}

        {messages.map((msg) => (
          <div
            key={msg.id}
            className={clsx(
              'flex gap-3 group',
              msg.role === 'user' ? 'flex-row-reverse' : ''
            )}
          >
            <div
              className={clsx(
                'w-8 h-8 rounded-lg flex items-center justify-center shrink-0',
                msg.role === 'user'
                  ? 'bg-neon-pink/20 text-neon-pink'
                  : 'bg-neon-cyan/20 text-neon-cyan'
              )}
            >
              {msg.role === 'user' ? (
                <User className="w-4 h-4" />
              ) : (
                <Bot className="w-4 h-4" />
              )}
            </div>
            <div
              className={clsx(
                'max-w-[80%] rounded-xl px-4 py-3 relative',
                msg.role === 'user'
                  ? 'bg-neon-pink/10 border border-neon-pink/20'
                  : 'glass'
              )}
            >
              <div className="absolute top-2 right-2">
                <CopyButton text={msg.content} />
              </div>
              <p className="text-sm whitespace-pre-wrap font-mono leading-relaxed pr-6">
                {msg.content}
              </p>
              <p className="text-[10px] text-void-500 mt-2">
                {msg.timestamp.toLocaleTimeString()}
              </p>
            </div>
          </div>
        ))}

        {/* Streaming response */}
        {streamingContent && (
          <div className="flex gap-3">
            <div className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0 bg-neon-cyan/20 text-neon-cyan">
              <Loader2 className="w-4 h-4 animate-spin" />
            </div>
            <div className="max-w-[80%] rounded-xl px-4 py-3 glass">
              <p className="text-sm whitespace-pre-wrap font-mono leading-relaxed cursor-blink">
                {streamingContent}
              </p>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="p-4 border-t border-void-800">
        <div className="flex gap-3">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={connected ? "Tell me what to do..." : "Connecting..."}
            disabled={!connected}
            className="input-field flex-1"
          />
          <button
            type="submit"
            disabled={!connected || !input.trim()}
            className={clsx(
              'btn-primary px-6 flex items-center gap-2',
              (!connected || !input.trim()) && 'opacity-50 cursor-not-allowed'
            )}
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </form>
    </div>
  );
}

