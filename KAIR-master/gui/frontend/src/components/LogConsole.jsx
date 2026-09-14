import { useEffect, useRef, useState, useCallback } from 'react'
import { openLogStream } from '../api/client'

function classifyLine(line) {
  const l = line.toLowerCase()
  if (l.includes('error') || l.includes('traceback') || l.includes('exception')) return 'lv-warn'
  if (l.includes('warn')) return 'lv-warn'
  if (l.includes('complete') || l.includes('saved') || l.includes('done')) return 'lv-ok'
  if (l.startsWith('<') || l.includes('iter:')) return 'lv-step'
  if (line.includes('Delta:') || line.includes('average delta') || line.includes('Δ')) return 'lv-delta'
  if (line.trim().startsWith('LR:') || line.includes('average lr')) return 'lv-lr'
  if (line.trim().startsWith('SR:') || line.includes('average sr')) return 'lv-sr'
  if (line.includes('PREVIEW_READY')) return 'lv-preview'
  return 'lv-info'
}

const PREVIEW_STAGES = [
  { key: 'load_hr',     label: 'HR loaded' },
  { key: 'bands_hr',    label: 'HR bands' },
  { key: 'load_lr',     label: 'LR loaded' },
  { key: 'bands_lr',    label: 'LR bands' },
  { key: 'coreg_a',     label: 'Coreg — ORB' },
  { key: 'coreg_b',     label: 'Coreg — Phase' },
  { key: 'radiometric', label: 'Radiometric' },
  { key: 'patches',     label: 'Sample patches' },
]

const PREVIEW_MARKER_RE = /PREVIEW_READY (\S+) (\S+) (\S+)/

function Lightbox({ preview, label, onClose }) {
  useEffect(() => {
    const handler = (e) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose])

  return (
    <div className="lightbox-overlay" onClick={onClose}>
      <div className="lightbox-box" onClick={(e) => e.stopPropagation()}>
        <div className="lightbox-header">
          <span className="lightbox-title">{label}</span>
          <span className="lightbox-scene">{preview.scene}</span>
          <button className="lightbox-close" onClick={onClose}>✕</button>
        </div>
        <img src={preview.url} alt={label} className="lightbox-img" />
      </div>
    </div>
  )
}

function useElapsedTimer(running) {
  const [elapsed, setElapsed] = useState(0)
  const startRef = useRef(null)

  useEffect(() => {
    if (running) {
      startRef.current = Date.now() - elapsed * 1000
      const id = setInterval(() => {
        setElapsed(Math.floor((Date.now() - startRef.current) / 1000))
      }, 1000)
      return () => clearInterval(id)
    }
  }, [running])  // eslint-disable-line react-hooks/exhaustive-deps

  const reset = useCallback(() => { setElapsed(0); startRef.current = null }, [])
  return { elapsed, reset }
}

function formatElapsed(s) {
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = s % 60
  if (h > 0) return `${h}h ${m}m ${sec}s`
  if (m > 0) return `${m}m ${sec}s`
  return `${sec}s`
}

export default function LogConsole({ domain, jobId, onStop, onPause, onResume, onComplete, onLine, onPreviewsChange, previewStages }) {
  const stagesForStrip = previewStages || PREVIEW_STAGES
  const [lines, setLines] = useState([])
  const [status, setStatus] = useState('pending')
  const [copied, setCopied] = useState(false)
  const [previews, setPreviews] = useState({})
  const [lightbox, setLightbox] = useState(null)
  const logRef = useRef(null)
  const esRef = useRef(null)
  const previewsRef = useRef({})
  const onCompleteRef = useRef(onComplete)
  const onLineRef = useRef(onLine)
  const onPreviewsChangeRef = useRef(onPreviewsChange)
  onCompleteRef.current = onComplete
  onLineRef.current = onLine
  onPreviewsChangeRef.current = onPreviewsChange

  const isLiveStatus = (s) => s === 'running' || s === 'pending'
  const { elapsed, reset: resetTimer } = useElapsedTimer(isLiveStatus(status))

  useEffect(() => {
    if (!jobId) return
    setLines([])
    setStatus('running')
    resetTimer()

    // Restore any previews already cached for this job before the SSE stream
    // replays — gives instant thumbnails on tab-switch or page refresh.
    const cacheKey = `kair_previews_${domain}_${jobId}`
    try {
      const cached = JSON.parse(localStorage.getItem(cacheKey) || '{}')
      if (Object.keys(cached).length > 0) {
        previewsRef.current = cached
        setPreviews(cached)
        onPreviewsChangeRef.current?.(cached)
      } else {
        previewsRef.current = {}
        setPreviews({})
      }
    } catch {
      previewsRef.current = {}
      setPreviews({})
    }

    esRef.current = openLogStream(
      domain,
      jobId,
      (line) => {
        setLines((prev) => [...prev, line])
        if (onLineRef.current) onLineRef.current(line)
        if (domain === 'preprocessing' || domain === 'inference') {
          const match = line.match(PREVIEW_MARKER_RE)
          if (match) {
            const [, filename, stage, scene] = match
            const url = `/api/${domain}/preview/${jobId}/${encodeURIComponent(filename)}`
            const next = { ...previewsRef.current, [stage]: { url, scene } }
            previewsRef.current = next
            setPreviews(next)
            try { localStorage.setItem(cacheKey, JSON.stringify(next)) } catch {}
            onPreviewsChangeRef.current?.(next)
          }
        }
      },
      (s) => {
        setStatus(s)
        if (s.toLowerCase().includes('completed') && onCompleteRef.current) {
          onCompleteRef.current()
        }
      }
    )

    return () => esRef.current?.close()
  }, [jobId, domain])

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [lines])

  if (!jobId) return null

  const isLive = status === 'running' || status === 'pending'
  const isPaused = status === 'paused'
  const visibleStages = stagesForStrip.filter((s) => previews[s.key])

  return (
    <div style={{ marginTop: 20 }}>
      {lightbox && (
        <Lightbox
          preview={previews[lightbox.key]}
          label={lightbox.label}
          onClose={() => setLightbox(null)}
        />
      )}

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <h3 style={{ fontSize: 15, fontWeight: 600, color: 'var(--ink)' }}>Live Output</h3>
          <div className="run-status">
            {isLive && <span className="pulse-dot" />}
            {isPaused && <span style={{ marginRight: 4, fontSize: 10 }}>⏸</span>}
            {status}
          </div>
          {elapsed > 0 && (
            <span className="mono" style={{
              fontSize: 11, color: isLive ? 'var(--cobalt-deep)' : 'var(--ink-3)',
              background: 'var(--surface-2)', border: '1px solid var(--line-2)',
              borderRadius: 4, padding: '2px 7px',
            }}>
              ⏱ {formatElapsed(elapsed)}
            </span>
          )}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span className="mono" style={{ fontSize: 11, color: 'var(--ink-3)' }}>{lines.length} lines</span>
          <button
            className="btn"
            style={{ padding: '4px 10px', fontSize: 11 }}
            disabled={lines.length === 0}
            onClick={() => {
              navigator.clipboard.writeText(lines.join('\n')).then(() => {
                setCopied(true)
                setTimeout(() => setCopied(false), 2000)
              })
            }}
          >
            {copied ? '✓ Copied!' : '⎘ Copy log'}
          </button>
          {(isLive || isPaused) && (onPause || onResume) && (
            <button
              className="btn"
              style={{ padding: '5px 12px', fontSize: 12 }}
              onClick={isPaused ? onResume : onPause}
            >
              {isPaused ? '▶ Resume' : '⏸ Pause'}
            </button>
          )}
          {(isLive || isPaused) && onStop && (
            <button
              className="btn"
              style={{ padding: '5px 12px', fontSize: 12, borderColor: 'var(--bad)', color: 'var(--bad)' }}
              onClick={onStop}
            >
              ✕ Cancel
            </button>
          )}
        </div>
      </div>

      {visibleStages.length > 0 && (
        <div className="preview-strip">
          {visibleStages.map((s) => (
            <button key={s.key} className="preview-thumb" onClick={() => setLightbox(s)}>
              <img src={previews[s.key].url} alt={s.label} />
              <span className="preview-label">{s.label}</span>
              <span className="preview-scene">{previews[s.key].scene}</span>
            </button>
          ))}
        </div>
      )}

      <div className="logstream scroll" ref={logRef}>
        {lines.map((line, i) => (
          <div key={i} className={`ln ${classifyLine(line)}`}>
            {line}
          </div>
        ))}
      </div>
    </div>
  )
}
