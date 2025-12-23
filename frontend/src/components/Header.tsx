import { Cpu, Zap } from 'lucide-react';

export function Header() {
  return (
    <header className="glass border-b border-void-800 px-6 py-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="relative">
            <Cpu className="w-8 h-8 text-neon-cyan" />
            <Zap className="w-4 h-4 text-neon-pink absolute -top-1 -right-1 animate-pulse" />
          </div>
          <div>
            <h1 className="text-xl font-display font-bold neon-text">
              Ambient Desktop
            </h1>
            <p className="text-xs text-void-400 font-mono">
              Multi-Agent Computer Automation
            </p>
          </div>
        </div>
        
        <div className="flex items-center gap-4">
          <div className="px-3 py-1 rounded-full glass text-xs font-mono text-void-300">
            v1.0.0
          </div>
        </div>
      </div>
    </header>
  );
}

