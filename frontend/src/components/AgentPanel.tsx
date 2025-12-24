import { Bot, Globe, FileText, Puzzle, Loader2, CheckCircle, XCircle, Pause } from 'lucide-react';
import { useAgentStore } from '../stores/agentStore';
import clsx from 'clsx';

const typeIcons: Record<string, typeof Bot> = {
  coordinator: Bot,
  browser: Globe,
  file: FileText,
  mcp: Puzzle,
  custom: Bot,
};

const statusColors = {
  idle: 'text-void-400',
  busy: 'text-neon-cyan',
  error: 'text-red-400',
  stopped: 'text-void-600',
};

const statusIcons = {
  idle: Pause,
  busy: Loader2,
  error: XCircle,
  stopped: CheckCircle,
};

export function AgentPanel() {
  const agents = useAgentStore((state) => state.agents);
  
  // Filter out stopped and idle agents - only show active ones
  const activeAgents = agents.filter((agent) => 
    agent.status === 'busy' || agent.status === 'error'
  );

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-4 py-3 border-b border-void-800 glass">
        <h2 className="font-display font-semibold text-void-200 flex items-center gap-2">
          <Bot className="w-5 h-5 text-neon-pink" />
          Active Agents
          {activeAgents.length > 0 && (
            <span className="text-xs text-void-500 font-mono">
              ({activeAgents.length})
            </span>
          )}
        </h2>
      </div>

      {/* Agents Grid */}
      <div className="flex-1 overflow-y-auto overflow-x-hidden p-4">
        {activeAgents.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-void-500">
            <Bot className="w-12 h-12 mb-3 opacity-30" />
            <p className="text-sm font-mono">No agents running</p>
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-3">
            {activeAgents.map((agent) => {
              const TypeIcon = typeIcons[agent.type] || Bot;
              const StatusIcon = statusIcons[agent.status];
              const statusColor = statusColors[agent.status];

              return (
                <div
                  key={agent.id}
                  className={clsx(
                    'glass rounded-lg p-3 transition-all duration-200',
                    agent.status === 'busy' && 'neon-border animate-glow'
                  )}
                >
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <TypeIcon className="w-4 h-4 text-neon-cyan" />
                      <span className="font-mono text-xs font-medium truncate max-w-[100px]">
                        {agent.name}
                      </span>
                    </div>
                    <div className={clsx('flex items-center gap-1', statusColor)}>
                      <StatusIcon
                        className={clsx(
                          'w-3.5 h-3.5',
                          agent.status === 'busy' && 'animate-spin'
                        )}
                      />
                      <span className="text-[10px] uppercase font-mono">
                        {agent.status}
                      </span>
                    </div>
                  </div>

                  {agent.summary && (
                    <p className="text-[11px] text-void-400 mb-2 line-clamp-2">
                      {agent.summary}
                    </p>
                  )}

                  {agent.progress !== undefined && agent.status === 'busy' && (
                    <div className="mt-2">
                      <div className="h-1 bg-void-800 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-neon-cyan to-neon-pink transition-all duration-300"
                          style={{ width: `${agent.progress * 100}%` }}
                        />
                      </div>
                      <p className="text-[10px] text-void-500 mt-1 text-right font-mono">
                        {Math.round(agent.progress * 100)}%
                      </p>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

