import React, { Component, ErrorInfo, ReactNode } from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'

// Global error boundary
class GlobalErrorBoundary extends Component<
  { children: ReactNode },
  { hasError: boolean; error: Error | null }
> {
  constructor(props: { children: ReactNode }) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Global error caught:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100vh',
          backgroundColor: '#0a0a0f',
          color: '#e0e0e0',
          fontFamily: 'system-ui, sans-serif',
          padding: '20px',
          textAlign: 'center',
        }}>
          <h1 style={{ color: '#ff6b6b', marginBottom: '20px' }}>Something went wrong</h1>
          <p style={{ marginBottom: '10px', color: '#888' }}>
            {this.state.error?.message || 'An unexpected error occurred'}
          </p>
          <button
            onClick={() => {
              localStorage.removeItem('ambient-agent-storage');
              window.location.reload();
            }}
            style={{
              padding: '12px 24px',
              backgroundColor: '#00d4ff',
              color: '#0a0a0f',
              border: 'none',
              borderRadius: '8px',
              cursor: 'pointer',
              fontSize: '16px',
              fontWeight: 'bold',
              marginTop: '20px',
            }}
          >
            Reset & Reload
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

// Also catch unhandled errors
window.onerror = (message, source, lineno, colno, error) => {
  console.error('Unhandled error:', message, source, lineno, colno, error);
};

window.onunhandledrejection = (event) => {
  console.error('Unhandled promise rejection:', event.reason);
};

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <GlobalErrorBoundary>
      <App />
    </GlobalErrorBoundary>
  </React.StrictMode>,
)

