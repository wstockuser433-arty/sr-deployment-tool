import { useEffect, useRef, useState } from 'react'
import { getInferenceProgress } from '../api/client'

/**
 * Minimal inline panel that lives directly under the "Run Super-Resolution" button.
 * Shows: status pill, stage, progress bar, patch counter, elapsed, output dir,
 * Stop button. When `done` is true it also renders a compact metrics summary
 * passed via the `metrics` prop.
 *
 * Progress is polled from /api/inference/progress/{jobId} every 2s while running.
 */
export default function InlineInferencePanel({
  jobId,
  running,
  cancelled = false,
  onStop,
  outputDir,
  metrics = null,
  metricsError = '',
  onRetryMetrics = null,
  scale = null,
  patchSize = null,
  overlap = null,
  inputDims = null,          // { width, height } or null
  estimatedPatches = null,   // number or null
  acceptBar = null,          // { psnrGain, ssimFloor } or null
}) {
  const [progress, setProgress] = useState(null)
  const [elapsed, setElapsed] = useState(0)
  const startedAtRef = useRef(null)
  const pollRef = useRef(null)
  const tickRef = useRef(null)

  // Reset on new jobId
  useEffect(() => {
    setProgress(null)
    setElapsed(0)
    startedAtRef.current = jobId ? Date.now() : null
  }, [jobId])

  // Poll progress while running
  useEffect(() => {
    clearInterval(pollRef.current)
    if (!jobId || !running) return

    let cancelled = false
    let unknownCount = 0
    const UNKNOWN_LIMIT = 5      // ~10s of grace before we give up

    const tick = async () => {
      try {
        const r = await getInferenceProgress(jobId)
        if (cancelled) return
        const data = r.data || {}
        setProgress(data)

        if (["completed", "failed", "cancelled"].includes(data.status)) {
          clearInterval(pollRef.current)
          return
        }

        if (data.status === "unknown") {
          unknownCount += 1
          if (unknownCount >= UNKNOWN_LIMIT) {
            // The backend doesn't know this job. This is either a stale ID
            // (survived a restart) or a race during launch that never resolved.
            // Stop polling; the outer JobContext reconciler will clear it too.
            clearInterval(pollRef.current)
          }
        } else {
          unknownCount = 0
        }
      } catch { /* swallow — endpoint is lenient now */ }
    }

    const startDelay = setTimeout(tick, 300)
    pollRef.current = setInterval(tick, 2000)

    return () => {
      cancelled = true
      clearTimeout(startDelay)
      clearInterval(pollRef.current)
    }
  }, [jobId, running])

  // Elapsed ticker
  useEffect(() => {
    if (!running || !startedAtRef.current) { clearInterval(tickRef.current); return }
    tickRef.current = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startedAtRef.current) / 1000))
    }, 1000)
    return () => clearInterval(tickRef.current)
  }, [running, jobId])

  if (!jobId && !running) return null

  // Derive authoritative status from the polled payload. The `running` prop
  // is only a hint; the backend's `status` field wins once we've seen it.
  const backendStatus = progress?.status || null
  const isTerminal = backendStatus && ["completed", "failed", "cancelled"].includes(backendStatus)
  
  // Priority: explicit cancellation flag from parent > backend status > running prop
  const uiStatus =
    cancelled              ? 'cancelled'
    : isTerminal           ? backendStatus       // 'completed' | 'failed' | 'cancelled'
    : running              ? 'running'
    : 'completed'
  
  // const effectiveRunning = running && !isTerminal

  // const pct = effectiveRunning ? Math.min(100, Math.max(0, progress?.percent ?? 0)) : 100

  const STATUS_META = {
    running:   { label: 'Running',   dot: 'var(--cobalt-deep)', pulse: true,  pillColor: 'var(--cobalt-deep)' },
    completed: { label: 'Complete',  dot: 'var(--ok)',          pulse: false, pillColor: 'var(--ok)' },
    cancelled: { label: 'Cancelled', dot: 'var(--ink-3)',       pulse: false, pillColor: 'var(--ink-3)' },
    failed:    { label: 'Failed',    dot: 'var(--bad)',         pulse: false, pillColor: 'var(--bad)' },
  }
  const meta = STATUS_META[uiStatus] || STATUS_META.running


  const pct = uiStatus === 'running'
    ? Math.min(100, Math.max(0, progress?.percent ?? 0))
    : uiStatus === 'cancelled'
      ? Math.min(100, Math.max(0, progress?.percent ?? 0))   // freeze at last value
      : 100

  const fmtTime = (s) => {
    const m = Math.floor(s / 60)
    const r = s % 60
    return `${String(m).padStart(2, '0')}:${String(r).padStart(2, '0')}`
  }

  const stageLabel = uiStatus === 'cancelled'
  ? 'Stopped'
  : {
      coreg: 'Coregistration',
      'stage a': 'ORB alignment',
      'stage b': 'Phase correlation',
      radiometric: 'Radiometric normalisation',
      histogram: 'Histogram matching',
      patch: 'Patch inference',
      infer: 'Patch inference',
      stitch: 'Stitching',
      metric: 'Computing metrics',
      preview: 'Rendering previews',
      done: 'Done',
      complete: 'Done',
    }[progress?.stage] || (uiStatus === 'running' ? 'Running' : 'Complete')

  // Compact metrics grid for the completion state
  const metricsNode = metrics ? (
    <div style={{
      marginTop: 14, padding: '12px 14px',
      borderRadius: 'var(--radius-sm)',
      background: 'var(--bg-2)', border: '1px solid var(--line-2)',
    }}>
      <div style={{
        fontSize: 10, fontWeight: 700, color: 'var(--ink-3)',
        textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 8,
      }}>
        METRICS
      </div>
      <div style={{
        display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(110px, 1fr))',
        gap: 10, fontSize: 12,
      }}>
        {['psnr', 'ssim', 'it_ssim', 'sam', 'uiqi', 'rmse', 'fsim', 'srer'].map(k => {
          const sr = metrics.sr?.[k]
          const d  = metrics.delta?.[k]
          const lowerBetter = k === 'sam' || k === 'rmse'
          const improved = typeof d === 'number' && (lowerBetter ? d < 0 : d > 0)
          const worsened = typeof d === 'number' && (lowerBetter ? d > 0 : d < 0)
          const val = (typeof sr === 'number' && isFinite(sr)) ? sr.toFixed(3) : '—'
          const dv  = (typeof d === 'number' && isFinite(d))
            ? `${d >= 0 ? '+' : ''}${d.toFixed(2)}`
            : null
          return (
            <div key={k}>
              <div style={{ fontSize: 10, color: 'var(--ink-3)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                {k.toUpperCase().replace('_', '-')}
              </div>
              <div style={{ fontFamily: 'var(--font-mono)', color: 'var(--ink-1)', fontWeight: 600 }}>{val}</div>
              {dv && (
                <div style={{
                  fontFamily: 'var(--font-mono)', fontSize: 10,
                  color: improved ? 'var(--ok)' : worsened ? 'var(--bad)' : 'var(--ink-3)',
                }}>
                  Δ {dv}
                </div>
              )}
            </div>
          )
        })}
      </div>
      {acceptBar && (() => {
        const psnrGain = metrics.delta?.psnr
        const srSsim = metrics.sr?.ssim
        if (typeof psnrGain !== 'number' || typeof srSsim !== 'number') return null
        const psnrPass = psnrGain >= acceptBar.psnrGain
        const ssimPass = srSsim >= acceptBar.ssimFloor
        const overall = psnrPass && ssimPass
        return (
          <div style={{
            marginTop: 12, padding: '8px 12px', borderRadius: 'var(--radius-sm)',
            border: `1px solid ${overall ? 'var(--ok)' : 'var(--bad)'}`,
            background: overall ? 'rgba(60,180,100,0.08)' : 'rgba(220,70,70,0.08)',
            fontSize: 11, color: overall ? 'var(--ok)' : 'var(--bad)', fontWeight: 600,
          }}>
            {overall ? '✓ PASS' : '✗ FAIL'} — client acceptance
            <span style={{ color: 'var(--ink-2)', fontWeight: 400, marginLeft: 10 }}>
              PSNR gain {psnrGain >= 0 ? '+' : ''}{psnrGain.toFixed(2)} dB
              {' · '}SSIM {srSsim.toFixed(3)}
            </span>
          </div>
        )
      })()}
    </div>
  ) : metricsError ? (
    <div style={{ marginTop: 12, fontSize: 12, color: 'var(--bad)' }}>
      {metricsError}
      {onRetryMetrics && (
        <button className="btn" style={{ fontSize: 11, padding: '3px 10px', marginLeft: 10 }}
          onClick={onRetryMetrics}>↻ Retry</button>
      )}
    </div>
  ) : null

  return (
    <div style={{
      marginTop: 16,
      padding: '14px 16px',
      borderRadius: 'var(--radius-sm)',
      background: 'var(--surface)',
      border: `1px solid ${running ? 'var(--cobalt-deep)' : 'var(--line-2)'}`,
      transition: 'border-color 0.2s',
    }}>
      {/* Header row — status pill + stop button */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span style={{
              display: 'inline-flex', alignItems: 'center', gap: 6,
              fontSize: 12, fontWeight: 600,
              color: meta.dot,
            }}>
              <span style={{
                display: 'inline-block', width: 8, height: 8, borderRadius: '50%',
                background: meta.dot,
                boxShadow: meta.pulse ? '0 0 0 3px rgba(80,130,220,0.18)' : 'none',
                animation: meta.pulse ? 'pulse 1.4s ease-in-out infinite' : 'none',
              }} />
              {meta.label}
            </span>
          <span style={{ fontSize: 12, color: 'var(--ink-2)' }}>{stageLabel}</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ fontSize: 11, color: 'var(--ink-3)', fontFamily: 'var(--font-mono)' }}>
            {fmtTime(elapsed)}
          </span>
          {uiStatus === 'running' && onStop && (
            <button type="button" onClick={onStop}
              style={{
                fontSize: 12, fontWeight: 600, padding: '5px 14px',
                borderRadius: 'var(--radius-sm)',
                border: '1px solid var(--bad)', background: 'transparent',
                color: 'var(--bad)', cursor: 'pointer',
              }}>
              ■ Stop
            </button>
          )}
        </div>
      </div>

      {/* Progress bar */}
      <div style={{
        height: 6, borderRadius: 3, background: 'var(--bg-2)',
        overflow: 'hidden', position: 'relative',
      }}>
        <div style={{
          height: '100%', width: `${pct}%`,
          background:
            uiStatus === 'running'   ? 'linear-gradient(90deg, var(--cobalt-deep), #7aa9ff)'
          : uiStatus === 'cancelled' ? 'var(--ink-3)'
          : uiStatus === 'failed'    ? 'var(--bad)'
          :                            'var(--ok)',
          transition: 'width 0.4s ease-out',
          borderRadius: 3,
        }} />
      </div>

      {/* Progress text row */}
      <div style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        marginTop: 6, fontSize: 11, color: 'var(--ink-3)', fontFamily: 'var(--font-mono)',
      }}>
        <span>
          {progress?.total > 0
            ? `Patch ${progress.current} / ${progress.total}`
            : estimatedPatches
              ? `~${estimatedPatches.toLocaleString()} patches expected`
              : 'Initialising…'}
        </span>
        <span>{pct.toFixed(1)}%</span>
      </div>

      {/* Context strip — output dir + inference params */}
      <div style={{
        marginTop: 12, paddingTop: 10,
        borderTop: '1px solid var(--line-2)',
        display: 'grid', gridTemplateColumns: 'auto 1fr', rowGap: 4, columnGap: 12,
        fontSize: 11,
      }}>
        {outputDir && (
          <>
            <span style={{ color: 'var(--ink-3)' }}>Output</span>
            <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--ink-2)', wordBreak: 'break-all' }}>
              {outputDir}
            </span>
          </>
        )}
        {(scale || patchSize || overlap) && (
          <>
            <span style={{ color: 'var(--ink-3)' }}>Params</span>
            <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--ink-2)' }}>
              {scale && <>×{scale}</>}
              {patchSize && <> · patch {patchSize}</>}
              {overlap != null && <> · overlap {overlap}</>}
              {inputDims && <> · in {inputDims.width}×{inputDims.height}</>}
              {inputDims && scale && (
                <> · out {inputDims.width * scale}×{inputDims.height * scale}</>
              )}
            </span>
          </>
        )}
      </div>

      {/* Compact metrics (only when done) */}
      {uiStatus === 'completed' && metricsNode}

      {/* Pulse keyframe — inject once */}
      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.4; }
        }
      `}</style>
    </div>
  )
}