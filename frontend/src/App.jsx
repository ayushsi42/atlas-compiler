import { useState, useCallback, useEffect } from 'react'
import './App.css'

const API_URL = 'http://localhost:8000'

// SVG Icons
const Icons = {
  Zap: () => (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
    </svg>
  ),
  Code: () => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <polyline points="16 18 22 12 16 6" /><polyline points="8 6 2 12 8 18" />
    </svg>
  ),
  Check: () => (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  ),
  Loader: () => <span className="spinner"></span>,
  Brain: () => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-2z" />
      <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-4.44-2z" />
    </svg>
  ),
  Scale: () => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <line x1="12" y1="3" x2="12" y2="21" /><path d="M5 7l7-4 7 4" /><path d="M5 7v7l7 4 7-4V7" />
    </svg>
  ),
  Rocket: () => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z" />
      <path d="M12 15l-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z" />
    </svg>
  ),
  Play: () => (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
      <polygon points="5 3 19 12 5 21 5 3" />
    </svg>
  ),
  Circle: () => (
    <svg width="8" height="8" viewBox="0 0 24 24" fill="currentColor">
      <circle cx="12" cy="12" r="10" />
    </svg>
  ),
}

const SAMPLE_CODE = `def double_add(x, y):
    """Adds doubles of x and y."""
    return (x * 2) + (y * 2)`

function App() {
  const [code, setCode] = useState(SAMPLE_CODE)
  const [isCompiling, setIsCompiling] = useState(false)
  const [logs, setLogs] = useState([])
  const [stats, setStats] = useState(null)
  const [optimizedCode, setOptimizedCode] = useState('')
  const [assembly, setAssembly] = useState('')
  const [backendStatus, setBackendStatus] = useState('checking')

  useEffect(() => {
    const checkBackend = async () => {
      try {
        const res = await fetch(`${API_URL}/health`, { method: 'GET' })
        setBackendStatus(res.ok ? 'connected' : 'error')
      } catch {
        setBackendStatus('disconnected')
      }
    }
    checkBackend()
  }, [])

  const runCompilation = useCallback(async () => {
    setIsCompiling(true)
    setLogs([])
    setStats(null)
    setOptimizedCode('')
    setAssembly('')

    try {
      const response = await fetch(`${API_URL}/compile`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code }),
      })

      if (!response.ok) throw new Error(`Server error: ${response.status}`)

      const result = await response.json()

      for (const log of result.logs) {
        await new Promise(r => setTimeout(r, 80))
        setLogs(prev => [...prev, { ...log, id: Date.now() + Math.random() }])
      }

      setOptimizedCode(result.optimized_ir || result.original_ir)
      setAssembly(result.assembly || '')
      setStats(result.stats)
      setBackendStatus('connected')
    } catch {
      setBackendStatus('disconnected')
      await runMockCompilation()
    }

    setIsCompiling(false)
  }, [code])

  const runMockCompilation = async () => {
    const mockLogs = [
      { message: 'Backend offline — demo mode', status: 'info' },
      { message: 'Parsing...', status: 'checking' },
      { message: 'Parse complete', status: 'pass' },
      { message: 'Generating IR', status: 'checking' },
      { message: 'Type: (int64, int64) → int64', status: 'info' },
      { message: 'IR complete', status: 'pass' },
      { message: 'Strategy: mul → shl', status: 'info' },
      { message: 'Verified', status: 'pass' },
    ]

    for (const log of mockLogs) {
      await new Promise(r => setTimeout(r, 120))
      setLogs(prev => [...prev, { ...log, id: Date.now() }])
    }

    setOptimizedCode(`define i64 @double_add(i64 %x, i64 %y) {
entry:
  %0 = shl i64 %x, 1
  %1 = shl i64 %y, 1
  %2 = add i64 %0, %1
  ret i64 %2
}`)

    setStats({
      strategy: 'Multiply-to-Shift',
      original_time_us: 0.21,
      compiled_time_us: 0.15,
      real_speedup: '1.4x',
      verify_time_ms: 12.5,
      original_ir_lines: 45,
    })
  }

  const [showAsm, setShowAsm] = useState(false)
  const currentStep = logs.length

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <div className="logo-icon"><Icons.Zap /></div>
            <div className="logo-text">
              <span className="logo-name">Atlas Compiler</span>
              <span className="logo-tagline">Verifiable Neural JIT</span>
            </div>
          </div>
          <div className={`status-dot ${backendStatus}`}>
            <Icons.Circle />
            <span>{backendStatus === 'connected' ? 'Live' : 'Demo'}</span>
          </div>
        </div>
      </header>

      <main className="main">
        <div className="compiler-layout">
          {/* Pipeline Steps */}
          <div className="pipeline">
            {[
              { icon: Icons.Code, label: 'Lift', range: [0, 3] },
              { icon: Icons.Brain, label: 'Optimize', range: [3, 8] },
              { icon: Icons.Scale, label: 'Verify', range: [8, 12] },
              { icon: Icons.Rocket, label: 'Compile', range: [12, 99] },
            ].map((step, i) => (
              <div key={step.label} className="pipeline-step">
                <div className={`pipeline-node ${currentStep >= step.range[0] && currentStep < step.range[1] ? 'active' : ''} ${currentStep >= step.range[1] ? 'done' : ''}`}>
                  <step.icon />
                </div>
                <span className="pipeline-label">{step.label}</span>
                {i < 3 && <div className={`pipeline-line ${currentStep >= step.range[1] ? 'done' : ''}`} />}
              </div>
            ))}
          </div>

          <div className="panels">
            {/* Input Panel */}
            <div className="panel">
              <div className="panel-header">
                <span className="panel-title">Python Source</span>
              </div>
              <textarea
                className="code-input"
                value={code}
                onChange={(e) => setCode(e.target.value)}
                spellCheck={false}
                placeholder="def your_function(x, y):&#10;    return x + y"
              />
              <button className="compile-btn" onClick={runCompilation} disabled={isCompiling || !code.trim()}>
                {isCompiling ? <><Icons.Loader /> Compiling...</> : <><Icons.Play /> Compile</>}
              </button>
            </div>

            {/* Log Panel */}
            <div className="panel">
              <div className="panel-header">
                <span className="panel-title">Log</span>
                {stats && <span className="verified-badge"><Icons.Check /> OK</span>}
              </div>
              <div className="log-container">
                {logs.length === 0 ? (
                  <div className="log-empty">Ready</div>
                ) : (
                  logs.map(log => (
                    <div key={log.id} className={`log-line ${log.status}`}>
                      <span className="log-dot" />
                      {log.message}
                    </div>
                  ))
                )}
              </div>
            </div>

            {/* Output Panel */}
            <div className="panel">
              <div className="panel-header">
                <span className="panel-title">{showAsm ? 'Assembly' : 'LLVM IR'}</span>
                {optimizedCode && (
                  <button className="toggle-btn" onClick={() => setShowAsm(!showAsm)}>
                    {showAsm ? 'IR' : 'ASM'}
                  </button>
                )}
              </div>
              <pre className="code-output">
                {showAsm ? (assembly || '; No assembly') : (optimizedCode || '; Output appears here...')}
              </pre>
            </div>
          </div>

          {/* Stats */}
          {stats && (
            <div className="stats-row">
              <StatCard label="Strategy" value={stats.strategy} />
              <StatCard label="Python" value={`${stats.original_time_us?.toFixed(2)}µs`} />
              <StatCard label="Compiled" value={`${stats.compiled_time_us?.toFixed(2)}µs`} />
              <StatCard label="Speedup" value={stats.real_speedup} highlight />
              <StatCard label="Z3 Verify" value={`${stats.verify_time_ms}ms`} />
              <StatCard label="IR Lines" value={stats.original_ir_lines} />
            </div>
          )}
        </div>
      </main>
    </div>
  )
}

function StatCard({ label, value, highlight }) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className={`stat-value ${highlight ? 'highlight' : ''}`}>{value}</div>
    </div>
  )
}

export default App
