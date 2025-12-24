import { useRef, useEffect } from 'react';
import { Terminal, AlertCircle, Info, AlertTriangle, Bug, Trash2 } from 'lucide-react';
import { useAgentStore } from '../stores/agentStore';
import clsx from 'clsx';

const levelIcons = {
  debug: Bug,
  info: Info,
  warn: AlertTriangle,
  error: AlertCircle,
};

const levelColors = {
  debug: 'text-void-500',
  info: 'text-neon-cyan',
  warn: 'text-neon-orange',
  error: 'text-red-400',
};

export function ActivityLog() {
  const logs = useAgentStore((state) => state.logs);
  const clearLogs = useAgentStore((state) => state.clearLogs);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = 0;
    }
  }, [logs]);

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-4 py-3 border-b border-void-800 glass flex items-center justify-between">
        <h2 className="font-display font-semibold text-void-200 flex items-center gap-2">
          <Terminal className="w-5 h-5 text-neon-green" />
          Activity Log
          {logs.length > 0 && (
            <span className="text-xs text-void-500 font-mono">
              ({logs.length})
            </span>
          )}
        </h2>
        {logs.length > 0 && (
          <button
            onClick={clearLogs}
            className="p-1.5 rounded hover:bg-void-800 text-void-500 hover:text-void-300 transition-colors"
            title="Clear logs"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* Logs */}
      <div ref={containerRef} className="flex-1 overflow-y-auto overflow-x-hidden font-mono text-xs">
        {logs.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-void-500">
            <Terminal className="w-12 h-12 mb-3 opacity-30" />
            <p>No activity yet</p>
          </div>
        ) : (
          <div className="divide-y divide-void-900">
            {logs.map((log) => {
              // Safe lookup with fallback to Info icon
              const level = log.level && levelIcons[log.level] ? log.level : 'info';
              const Icon = levelIcons[level] || Info;
              const colorClass = levelColors[level] || 'text-void-500';
              
              // Safe timestamp handling
              const timestamp = log.timestamp instanceof Date 
                ? log.timestamp 
                : new Date(log.timestamp || Date.now());

              return (
                <div
                  key={log.id}
                  className={clsx(
                    'px-4 py-2 hover:bg-void-900/50 transition-colors',
                    log.level === 'error' && 'bg-red-900/10'
                  )}
                >
                  <div className="flex items-start gap-2">
                    <Icon className={clsx('w-3.5 h-3.5 mt-0.5 shrink-0', colorClass)} />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 min-w-0">
                        <span className={clsx('font-medium shrink-0', colorClass)}>
                          [{log.category || 'general'}]
                        </span>
                        <span className="text-void-400 truncate block">
                          {log.message || ''}
                        </span>
                      </div>
                      {log.details && (
                        <pre className="mt-1 text-[10px] text-void-500 overflow-hidden text-ellipsis whitespace-pre-wrap break-all">
                          {JSON.stringify(log.details, null, 2)}
                        </pre>
                      )}
                    </div>
                    <span className="text-[10px] text-void-600 shrink-0">
                      {timestamp.toLocaleTimeString()}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

