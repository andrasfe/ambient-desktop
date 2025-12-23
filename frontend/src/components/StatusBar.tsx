import { Wifi, WifiOff, Clock } from 'lucide-react';
import { useAgentStore } from '../stores/agentStore';

interface StatusBarProps {
  connected: boolean;
}

export function StatusBar({ connected }: StatusBarProps) {
  const agents = useAgentStore((state) => state.agents);
  const logs = useAgentStore((state) => state.logs);
  
  const busyAgents = agents.filter((a) => a.status === 'busy').length;
  const errorCount = logs.filter((l) => l.level === 'error').length;

  return (
    <footer className="glass border-t border-void-800 px-4 py-2">
      <div className="flex items-center justify-between text-xs font-mono">
        <div className="flex items-center gap-4">
          {/* Connection Status */}
          <div className="flex items-center gap-2">
            {connected ? (
              <>
                <Wifi className="w-3.5 h-3.5 text-neon-green" />
                <span className="text-neon-green">Connected</span>
              </>
            ) : (
              <>
                <WifiOff className="w-3.5 h-3.5 text-red-400 animate-pulse" />
                <span className="text-red-400">Disconnected</span>
              </>
            )}
          </div>
          
          {/* Agent Status */}
          <div className="flex items-center gap-2 text-void-400">
            <span>Agents: {agents.length}</span>
            {busyAgents > 0 && (
              <span className="text-neon-orange">({busyAgents} busy)</span>
            )}
          </div>
          
          {/* Error Count */}
          {errorCount > 0 && (
            <div className="text-red-400">
              {errorCount} error{errorCount !== 1 ? 's' : ''}
            </div>
          )}
        </div>
        
        <div className="flex items-center gap-2 text-void-400">
          <Clock className="w-3.5 h-3.5" />
          <span>{new Date().toLocaleTimeString()}</span>
        </div>
      </div>
    </footer>
  );
}

