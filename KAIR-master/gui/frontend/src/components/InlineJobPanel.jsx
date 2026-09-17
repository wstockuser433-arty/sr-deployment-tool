import { useEffect, useRef, useState } from 'react'
import { getInferenceProgress } from '../api/client'
// Persists job start timestamps across component remounts.
// Keyed by jobId; entries are pruned lazily when a job reaches terminal state.
const _START_TIMES = new Map()

/**
 * src/components/InlineJobPanel.jsx
 * ---------------------------------
 * Minimal inline panel that lives directly under a "Run …" button in both
 * Inference and Preprocessing. Shows: status pill, stage, progress bar,
 * patch counter, elapsed, output dir, and a Stop button (optionally Pause /
 * Resume via the `secondaryActions` prop). On completion it can render a
 * compact metrics grid via the `metrics` prop.
 */
export default function InlineJobPanel({
  jobId,
  running,
  cancelled = false,
  paused = false,                      // ← NEW
  onStop,
  secondaryActions = [],               // ← NEW: [{ label, onClick, disabled, title }]
  outputDir,
  metrics = null,
  metricsError = '',
  onRetryMetrics = null,
  scale = null,
  patchSize = null,
  overlap = null,
  inputDims = null,
  estimatedPatches = null,
  acceptBar = null,
  stageVocabulary = null,              // ← NEW: override the stage label map
  runningLabel = 'Running',            // ← NEW: default 'Running' for inference
}) {
  const [progress, setProgress] = useState(null)
  
  const [elapsed, setElapsed] = useState(0)
  const pollRef = useRef(null)
  const tickRef = useRef(null)

  // Persist the start time per jobId in a module-level map so remounts
  // (tab switches, HMR, reconciliation) don't reset the elapsed timer.
  const startedAtRef = useRef(null)
  if (!startedAtRef.current && jobId) {
    const cached = _START_TIMES.get(jobId)
    startedAtRef.current = cached ?? Date.now()
    _START_TIMES.set(jobId, startedAtRef.current)
  }

  useEffect(() => {
    setProgress(null)
    setElapsed(0)
    if (jobId) {
      const cached = _START_TIMES.get(jobId)
      startedAtRef.current = cached ?? Date.now()
      _START_TIMES.set(jobId, startedAtRef.current)
      // Immediately compute elapsed so the first render isn't stuck at 00:00
      setElapsed(Math.floor((Date.now() - startedAtRef.current) / 1000))
    } else {
      startedAtRef.current = null
    }
  }, [jobId])


  // Poll progress while running
  useEffect(() => {
    clearInterval(pollRef.current)
        if (!jobId || (!running && !paused)) return

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
  }, [jobId, running, paused])

  // Elapsed ticker
  useEffect(() => {
  if ((!running && !paused) || !startedAtRef.current) { clearInterval(tickRef.current); return }
    tickRef.current = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startedAtRef.current) / 1000))
    }, 1000)
    return () => clearInterval(tickRef.current)
  }, [running, paused, jobId])

  // Prune start-time entry when the job reaches a terminal state
  useEffect(() => {
    if (!jobId) return
    const terminal = cancelled || (progress?.status && ["completed", "failed", "cancelled"].includes(progress.status))
    if (terminal) {
      // Small grace period so a remount right at the end still finds the cached time
      const t = setTimeout(() => _START_TIMES.delete(jobId), 30_000)
      return () => clearTimeout(t)
    }
  }, [jobId, cancelled, progress?.status])

  if (!jobId && !running) return null

  // Derive authoritative status from the polled payload. The `running` prop
  // is only a hint; the backend's `status` field wins once we've seen it.
  const backendStatus = progress?.status || null
  const isTerminal = backendStatus && ["completed", "failed", "cancelled"].includes(backendStatus)
  
  // Priority: explicit cancellation flag from parent > backend status > running prop
   const uiStatus =
    cancelled              ? 'cancelled'
    : paused               ? 'paused'
    : isTerminal           ? backendStatus
    : running              ? 'running'
    : 'completed'

  const STATUS_META = {
    running:   { label: runningLabel, dot: 'var(--cobalt-deep)', pulse: true  },
    paused:    { label: 'Paused',     dot: 'rgb(180,120,20)',    pulse: false },
    completed: { label: 'Complete',   dot: 'var(--ok)',          pulse: false },
    cancelled: { label: 'Cancelled',  dot: 'var(--ink-3)',       pulse: false },
    failed:    { label: 'Failed',     dot: 'var(--bad)',         pulse: false },
  }

  const meta = STATUS_META[uiStatus] || STATUS_META.running


    // A percent is only meaningful when the backend has a real denominator
  // (progress.total > 0). Before the first [PROGRESS] marker, `percent` may
  // be null OR the backend may have fallen through to a default of 100 —
  // both cases must render as "indeterminate", never as a full bar.
  const hasRealProgress = progress?.total > 0 && progress?.percent != null
  const pct = hasRealProgress
    ? Math.min(100, Math.max(0, progress.percent))
    : (uiStatus === 'running' ? 0 : 100)

  const fmtTime = (s) => {
    const m = Math.floor(s / 60)
    const r = s % 60
    return `${String(m).padStart(2, '0')}:${String(r).padStart(2, '0')}`
  }

    const DEFAULT_STAGES = {
    coreg: 'Coregistration',
    'stage a': 'ORB alignment',
    'stage b': 'Phase correlation',
    'stage c': 'ECC refinement',
    radiometric: 'Radiometric normalisation',
    histogram: 'Histogram matching',
    cloud: 'Cloud masking',
    normalize: 'Normalisation',
    degradation: 'Degradation',
    save: 'Saving outputs',
    split: 'Train/test split',
    patch: 'Patch inference',
    infer: 'Patch inference',
    stitch: 'Stitching',
    metric: 'Computing metrics',
    preview: 'Rendering previews',
    done: 'Done',
    complete: 'Done',
  }
  const stages = stageVocabulary || DEFAULT_STAGES

  const stageLabel = uiStatus === 'cancelled'
    ? 'Stopped'
    : uiStatus === 'paused'
      ? 'Paused'
      : stages[progress?.stage] || (uiStatus === 'running' ? runningLabel : 'Complete')

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
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{ fontSize: 11, color: 'var(--ink-3)', fontFamily: 'var(--font-mono)', marginRight: 4 }}>
          {fmtTime(elapsed)}
        </span>

        {(uiStatus === 'running' || uiStatus === 'paused') && secondaryActions.map((a, i) => (
          <button key={i} type="button" onClick={a.onClick} disabled={a.disabled}
            title={a.title}
            style={{
              fontSize: 12, fontWeight: 600, padding: '5px 12px',
              borderRadius: 'var(--radius-sm)',
              border: '1px solid var(--line-2)', background: 'var(--surface)',
              color: a.disabled ? 'var(--ink-3)' : 'var(--ink-2)',
              cursor: a.disabled ? 'not-allowed' : 'pointer',
              opacity: a.disabled ? 0.5 : 1,
            }}>
            {a.label}
          </button>
        ))}

        {(uiStatus === 'running' || uiStatus === 'paused') && onStop && (
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

      {/* Progress bar */}
      <div style={{
        height: 6, borderRadius: 3, background: 'var(--bg-2)',
        overflow: 'hidden', position: 'relative',
      }}>
        {/* Indeterminate shimmer while the backend has no real denominator */}
        {uiStatus === 'running' && !hasRealProgress && (
          <div style={{
            position: 'absolute', inset: 0,
            background: 'linear-gradient(90deg, transparent, var(--cobalt-soft), transparent)',
            animation: 'inlinePanelShimmer 1.4s ease-in-out infinite',
          }} />
        )}
        {/* Determinate fill — rendered whenever we have a real percent OR
        the job has reached a terminal state (so a completed run always
        shows a full bar, even if the pipeline emitted no [PROGRESS]). */}
        {(hasRealProgress || (!running && !paused)) && (
          <div style={{
            height: '100%',
            width: hasRealProgress ? `${pct}%` : '100%',
            background:
              uiStatus === 'running'   ? 'linear-gradient(90deg, var(--cobalt-deep), #7aa9ff)'
            : uiStatus === 'paused'    ? 'rgb(180,120,20)'
            : uiStatus === 'cancelled' ? 'var(--ink-3)'
            : uiStatus === 'failed'    ? 'var(--bad)'
            :                            'var(--ok)',
            transition: 'width 0.4s ease-out',
            borderRadius: 3,
          }} />
        )}
      </div>

      {/* Progress text row */}
      <div style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        marginTop: 6, fontSize: 11, color: 'var(--ink-3)', fontFamily: 'var(--font-mono)',
      }}>
        <span>
          {hasRealProgress
            ? `Patch ${progress.current.toLocaleString()} / ${progress.total.toLocaleString()}`
            : uiStatus === 'running'
              ? (estimatedPatches
                  ? `~${estimatedPatches.toLocaleString()} candidates expected`
                  : progress?.stage
                    ? `${stages[progress.stage] || progress.stage}…`
                    : 'Starting patch extraction…')
              : uiStatus === 'cancelled'
                ? 'Stopped'
                : uiStatus === 'paused'
                  ? 'Paused'
                  : `${progress?.saved?.toLocaleString?.() ?? ''} patches saved`.trim() || 'Complete'}
        </span>
        <span>
          {uiStatus === 'running'
            ? (hasRealProgress ? `${pct.toFixed(1)}%` : 'working…')
            : uiStatus === 'cancelled'
              ? 'stopped'
              : uiStatus === 'paused'
                ? 'paused'
                : uiStatus === 'failed'
                  ? 'failed'
                  : 'done'}
        </span>
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
        @keyframes inlinePanelShimmer {
          0%   { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
      `}</style>
    </div>
  )
}