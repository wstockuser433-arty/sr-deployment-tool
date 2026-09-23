import { useEffect, useState, useRef, useMemo } from 'react'
import ReactCompareImage from 'react-compare-image'
import { useJobContext } from '../context/JobContext'
import {
  listInferenceTasks, listTrainingConfigs, getLatestModel,
  getConfigFromOptions, getConfigFromPath,
  startInference, stopInference,
  startRawPairedInference, startLROnlyInference,
  getRawInferenceMetrics, getRawResultImageUrl, getImageInfo,
  getInferenceSummary, getInferenceProgress,
} from '../api/client'
// import LogConsole from '../components/LogConsole'
import {
  SelectField, NumberField, BoolToggle,
  ArrayEditor, CollapsibleSection, PathField,
} from '../components/FormFields'
// import InlineInferencePanel from '../components/InlineInferencePanel'

import LogConsole from '../components/LogConsole'
import InlineJobPanel from '../components/InlineJobPanel'
import ImageInfoCard from '../components/ImageInfoCard'
import LiveOutputBadge from '../components/LiveOutputBadge'
import useImageMeta from '../hooks/useImageMeta'

/* ─── Constants ─────────────────────────────────────────────── */

/* ─── Live Output status badge ──────────────────────────────── */
// function LiveOutputBadge({ status }) {
//   const styles = {
//     running:   { bg: 'var(--cobalt-soft)', fg: 'var(--cobalt-deep)', label: 'Running'   },
//     cancelled: { bg: 'var(--bg-2)',        fg: 'var(--ink-3)',       label: 'Cancelled' },
//     completed: { bg: 'rgba(60,180,100,0.12)', fg: 'var(--ok)',       label: 'Complete'  },
//     failed:    { bg: 'rgba(220,70,70,0.10)',  fg: 'var(--bad)',      label: 'Failed'    },
//   }
//   const s = styles[status] || styles.running
//   return (
//     <span style={{
//       fontSize: 10, fontWeight: 600, padding: '1px 6px',
//       borderRadius: 4, background: s.bg, color: s.fg,
//       textTransform: 'uppercase', letterSpacing: '0.05em',
//     }}>{s.label}</span>
//   )
// }


const UPSAMPLER_OPTIONS = ['pixelshuffle', 'pixelshuffledirect', 'nearest+conv']
const RESI_OPTIONS = ['1conv', '3conv']
const METRIC_NAMES = ['psnr', 'ssim', 'it_ssim', 'sam', 'uiqi', 'rmse', 'fsim', 'srer']
const METRIC_LABELS = ['PSNR', 'SSIM', 'IT-SSIM', 'SAM', 'UIQI', 'RMSE', 'FSIM', 'SRER']
const LOWER_IS_BETTER = new Set(['sam', 'rmse'])

const DEFAULT_MODEL_CONFIG = {
  upscale: 2, in_chans: 3, img_size: 128, window_size: 8,
  img_range: 1.0, depths: [6, 6, 6, 6, 6, 6], embed_dim: 180,
  num_heads: [6, 6, 6, 6, 6, 6], mlp_ratio: 2,
  upsampler: 'pixelshuffle', resi_connection: '1conv',
}

const DEFAULT_COREG = {
  enable_preprocessing: true,
  coreg_a_enabled: true, coreg_a_max_features: 8000,
  coreg_a_match_ratio: 0.75, coreg_a_ransac_thresh: 4.0,
  coreg_b_enabled: true, coreg_b_upsample_factor: 100,
  radiometric_enabled: true, radiometric_block_size: 256,
  radiometric_rmse_threshold: 35.0, radiometric_n_samples: 150000,
  radiometric_post_hist_match: true, histogram_n_sample_windows: 40,
  nodata_value: 0, saturated_value: 32767,
  clip_percentiles: [2.0, 98.0], percentile_n_sample_windows: 20,
  coreg_preview_decim_dim: 2000,
}

function deepSet(obj, path, value) {
  const keys = path.split('.')
  const next = { ...obj }
  let cur = next
  for (let i = 0; i < keys.length - 1; i++) {
    cur[keys[i]] = { ...cur[keys[i]] }
    cur = cur[keys[i]]
  }
  cur[keys[keys.length - 1]] = value
  return next
}

/* ─── Config auto-load hook ─────────────────────────────────── */
function useConfigAutoLoad(modelSource, customModelPath, setModelConfig) {
  const [selectedOptions, setSelectedOptions] = useState('')
  const [configSource, setConfigSource] = useState(null)
  const pathDebounceRef = useRef(null)

  useEffect(() => {
    if (modelSource !== 'custom' || !customModelPath.trim()) {
      setConfigSource(null)
      return
    }
    clearTimeout(pathDebounceRef.current)
    pathDebounceRef.current = setTimeout(async () => {
      try {
        const r = await getConfigFromPath(customModelPath)
        if (r.data.source !== 'not_found' && Object.keys(r.data.model_config).length > 0) {
          setModelConfig({ ...DEFAULT_MODEL_CONFIG, ...r.data.model_config })
          setConfigSource({ type: 'train_json', label: r.data.task_name })
          setSelectedOptions('')
        }
      } catch { }
    }, 600)
    return () => clearTimeout(pathDebounceRef.current)
  }, [customModelPath, modelSource])

  const handleOptionsSelect = async (name) => {
    setSelectedOptions(name)
    if (!name) { setConfigSource(null); return }
    try {
      const r = await getConfigFromOptions(name)
      setModelConfig({ ...DEFAULT_MODEL_CONFIG, ...r.data.model_config })
      setConfigSource({ type: 'options_file', label: name })
    } catch { }
  }

  return { selectedOptions, configSource, handleOptionsSelect }
}

/* ─── Simplified model selection panel ──────────────────────── */
function ModelSelectionCard({
  modelSource, setModelSource, tasks, selectedTask, setSelectedTask,
  latestInfo, customModelPath, setCustomModelPath,
  optionsFiles = [], selectedOptions = '', onOptionsSelect, configSource = null,
  modelConfig,
}) {
  const detectedArch = useMemo(() => {
    if (!modelConfig) return null
    return `SwinIR ×${modelConfig.upscale}`
  }, [modelConfig])

  return (
    <div className="card" style={{ marginBottom: 16 }}>
      <div className="card-title">Model</div>
      <div className="mode-tabs" style={{ marginBottom: 14 }}>
        <button type="button" className={`mode-tab ${modelSource === 'auto' ? 'active' : ''}`}
          onClick={() => setModelSource('auto')}>Latest from task</button>
        <button type="button" className={`mode-tab ${modelSource === 'custom' ? 'active' : ''}`}
          onClick={() => setModelSource('custom')}>Custom path</button>
      </div>
      {modelSource === 'auto' ? (
        <>
          <SelectField label="Training task" hint="scans superresolution/ for the highest *_E.pth"
            value={selectedTask} onChange={setSelectedTask}
            options={[{ value: '', label: '— select task —' }, ...tasks.map(t => ({ value: t.task_name, label: t.task_name }))]} />
          {latestInfo && (
            <div style={{ background: 'var(--bg-2)', border: '1px solid var(--line-2)', borderRadius: 'var(--radius-sm)', padding: '10px 14px', marginTop: 12 }}>
              <div style={{ fontSize: 12, fontWeight: 500, color: 'var(--ok)' }}>✓ Latest model: iter {latestInfo.latest_iteration}</div>
              <div className="mono" style={{ fontSize: 11, color: 'var(--ink-3)', marginTop: 3, wordBreak: 'break-all' }}>{latestInfo.model_path}</div>
              {Object.keys(latestInfo.model_config || {}).length > 0 ? (
                <div style={{ fontSize: 11, color: 'var(--cobalt-deep)', marginTop: 4, fontWeight: 500 }}>✓ Architecture loaded from {latestInfo.options_file || 'train.json'}</div>
              ) : (
                <div style={{ fontSize: 11, color: 'var(--bad)', marginTop: 4, fontWeight: 500 }}>⚠ Config not found. Please fill in manually.</div>
              )}
            </div>
          )}
        </>
      ) : (
        <>
          <PathField label="Model checkpoint" mode="files" extensions=".pth,.pt"
            value={customModelPath} onChange={setCustomModelPath}
            placeholder="superresolution/task/models/175000_E.pth" mono />
          {optionsFiles.length > 0 && (
            <div className="form-group" style={{ marginTop: 6 }}>
              <label>
                Load config from options file
                <span className="hint" style={{ marginLeft: 6 }}>auto-fills architecture</span>
              </label>
              <select className="text-input" value={selectedOptions}
                onChange={(e) => onOptionsSelect?.(e.target.value)}>
                <option value="">— choose options file —</option>
                {optionsFiles.map((f) => (
                  <option key={f.name} value={f.name}>{f.name}</option>
                ))}
              </select>
            </div>
          )}
          {configSource ? (
            <div style={{
              display: 'flex', alignItems: 'center', gap: 6, marginTop: 6,
              padding: '8px 12px', borderRadius: 'var(--radius-sm)',
              background: 'var(--bg-2)', border: '1px solid var(--line-2)',
            }}>
              <span style={{ color: 'var(--ok)', fontSize: 13 }}>✓</span>
              <span style={{ fontSize: 12, color: 'var(--ink-2)' }}>
                Architecture auto-loaded from{' '}
                <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--ink)' }}>
                  {configSource.type === 'train_json'
                    ? `superresolution/${configSource.label}/options/train.json`
                    : configSource.label}
                </span>
              </span>
            </div>
          ) : (
            <div style={{ fontSize: 11, color: 'var(--ink-3)', marginTop: 6, padding: '8px 12px', background: 'var(--bg-2)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--line-2)' }}>
              ⚠ Architecture not auto-detected. Open <strong>Advanced Configuration</strong> below to set it manually.
            </div>
          )}
        </>
      )}

      {detectedArch && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8, marginTop: 12,
          padding: '8px 12px', borderRadius: 'var(--radius-sm)',
          background: 'var(--cobalt-soft)', border: '1px solid var(--cobalt-deep)',
        }}>
          <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--cobalt-deep)' }}>
            Detected model: {detectedArch}
          </span>
          <span style={{ fontSize: 11, color: 'var(--cobalt-deep)' }}>
            · Scale ×{modelConfig.upscale} · {modelConfig.in_chans} channel{modelConfig.in_chans !== 1 ? 's' : ''}
          </span>
        </div>
      )}
    </div>
  )
}

/* ─── Advanced model architecture (collapsed, locked) ──────── */
function ModelArchCard({ modelConfig, setMC, configSource = null }) {
  const [locked, setLocked] = useState(true)

  const titleNode = (
    <span style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
      ⚙ Advanced Model Configuration
      {configSource && (
        <span style={{ fontSize: 11, color: 'var(--ok)', fontWeight: 400 }}>✓ auto-loaded</span>
      )}
      <button
        type="button"
        onClick={(e) => { e.stopPropagation(); setLocked(l => !l) }}
        title={locked ? 'Unlock config fields for editing' : 'Lock config fields'}
        style={{
          marginLeft: 4, padding: '2px 8px', fontSize: 11,
          borderRadius: 'var(--radius-sm)',
          border: `1px solid ${locked ? 'var(--line-2)' : 'var(--cobalt-deep)'}`,
          background: locked ? 'var(--surface-2)' : 'var(--cobalt-soft)',
          color: locked ? 'var(--ink-3)' : 'var(--cobalt-deep)',
          cursor: 'pointer', fontWeight: 500,
        }}
      >
        {locked ? '🔒 Locked' : '🔓 Unlocked'}
      </button>
    </span>
  )

  return (
    <CollapsibleSection title={titleNode} defaultOpen={false}>
      {locked && (
        <div style={{ fontSize: 12, color: 'var(--ink-3)', marginBottom: 12, padding: '6px 10px', background: 'var(--bg-2)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--line-2)' }}>
          Config is locked to prevent accidental edits. Click <strong>Unlocked</strong> above to edit.
        </div>
      )}
      <div className="grid-2">
        <NumberField label="Upscale" value={modelConfig.upscale} onChange={v => setMC('upscale', v)} min={1} disabled={locked} />
        <NumberField label="In channels" value={modelConfig.in_chans} onChange={v => setMC('in_chans', v)} min={1} disabled={locked} />
      </div>
      <div className="grid-2">
        <NumberField label="Model input patch size" value={modelConfig.img_size} onChange={v => setMC('img_size', v)} min={16} step={8} disabled={locked} />
        <NumberField label="Window size" value={modelConfig.window_size} onChange={v => setMC('window_size', v)} min={4} step={2} disabled={locked} />
      </div>
      <div className="grid-2">
        <NumberField label="Embed dim" value={modelConfig.embed_dim} onChange={v => setMC('embed_dim', v)} min={60} step={12} disabled={locked} />
        <NumberField label="MLP ratio" value={modelConfig.mlp_ratio} onChange={v => setMC('mlp_ratio', v)} min={1} disabled={locked} />
      </div>
      <div className="grid-2">
        <SelectField label="Upsampler" value={modelConfig.upsampler} onChange={v => setMC('upsampler', v)} options={UPSAMPLER_OPTIONS} disabled={locked} />
        <SelectField label="Resi connection" value={modelConfig.resi_connection} onChange={v => setMC('resi_connection', v)} options={RESI_OPTIONS} disabled={locked} />
      </div>
      <ArrayEditor label="Depths" value={modelConfig.depths} onChange={v => setMC('depths', v)} disabled={locked} />
      <ArrayEditor label="Num heads" value={modelConfig.num_heads} onChange={v => setMC('num_heads', v)} disabled={locked} />
    </CollapsibleSection>
  )
}

/* ─── Coregistration options panel ─────────────────────────── */
function CoregSection({ coreg, setCoreg }) {
  const set = (k, v) => setCoreg(prev => ({ ...prev, [k]: v }))
  return (
    <CollapsibleSection title="Satellite Preprocessing (Coregistration + Radiometric)" defaultOpen={false}>
      <div style={{ padding: '10px 14px', background: 'var(--bg-2)', border: '1px solid var(--line-2)', borderRadius: 'var(--radius-sm)', marginBottom: 14, fontSize: 12, color: 'var(--ink-2)', lineHeight: 1.5 }}>
        When enabled, runs the full pipeline3.py alignment stack (CRS reproject → Stage A ORB → Stage B Phase Correlation → Radiometric Regression → Histogram Matching) before patch extraction. Required for raw multi-sensor satellite image pairs with fractional resolution ratios.
      </div>
      <BoolToggle label="Enable preprocessing pipeline" value={coreg.enable_preprocessing}
        onChange={v => set('enable_preprocessing', v)} />
      {coreg.enable_preprocessing && (
        <>
          <div style={{ marginTop: 14, fontWeight: 600, fontSize: 12, color: 'var(--ink-2)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 8 }}>Stage A — ORB Coregistration</div>
          <BoolToggle label="Enable Stage A (ORB)" value={coreg.coreg_a_enabled} onChange={v => set('coreg_a_enabled', v)} />
          {coreg.coreg_a_enabled && (
            <div className="grid-2">
              <NumberField label="Max features" value={coreg.coreg_a_max_features} onChange={v => set('coreg_a_max_features', v)} min={0} step={500} />
              <NumberField label="Match ratio" value={coreg.coreg_a_match_ratio} onChange={v => set('coreg_a_match_ratio', v)} min={0.5} max={1.0} step={0.05} />
            </div>
          )}

          <div style={{ marginTop: 14, fontWeight: 600, fontSize: 12, color: 'var(--ink-2)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 8 }}>Stage B — Phase Correlation</div>
          <BoolToggle label="Enable Stage B (Phase)" value={coreg.coreg_b_enabled} onChange={v => set('coreg_b_enabled', v)} />
          {coreg.coreg_b_enabled && (
            <NumberField label="Upsample factor" value={coreg.coreg_b_upsample_factor} onChange={v => set('coreg_b_upsample_factor', v)} min={0} step={10} />
          )}

          <div style={{ marginTop: 14, fontWeight: 600, fontSize: 12, color: 'var(--ink-2)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 8 }}>Radiometric Normalisation</div>
          <BoolToggle label="Enable radiometric regression" value={coreg.radiometric_enabled} onChange={v => set('radiometric_enabled', v)} />
          {coreg.radiometric_enabled && (
            <>
              <div className="grid-2">
                <NumberField label="Block size (px)" value={coreg.radiometric_block_size} onChange={v => set('radiometric_block_size', v)} min={64} step={64} />
                <NumberField label="RMSE threshold" value={coreg.radiometric_rmse_threshold} onChange={v => set('radiometric_rmse_threshold', v)} min={1} step={1} />
              </div>
              <BoolToggle label="Post histogram matching" value={coreg.radiometric_post_hist_match} onChange={v => set('radiometric_post_hist_match', v)} />
            </>
          )}

          <div style={{ marginTop: 14, fontWeight: 600, fontSize: 12, color: 'var(--ink-2)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 8 }}>Pixel Value Settings</div>
          <div className="grid-2">
            <NumberField label="Nodata value" value={coreg.nodata_value} onChange={v => set('nodata_value', v)} />
            <NumberField label="Saturated value" value={coreg.saturated_value} onChange={v => set('saturated_value', v)} />
          </div>
          <ArrayEditor label="Clip percentiles [lo, hi]" value={coreg.clip_percentiles} onChange={v => set('clip_percentiles', v)} />
        </>
      )}
    </CollapsibleSection>
  )
}

// /* ─── Image Information card ───────────────────────────────── */
// function ImageInfoCard({ meta, title = 'IMAGE INFORMATION' }) {
//   if (!meta) return null
//   const rows = [
//     ['Dimensions', `${meta.width} × ${meta.height}`],
//     ['Bands', meta.bands],
//     ['Data type', meta.dtype || '—'],
//     ['Format', meta.format || '—'],
//     ['Geospatial', meta.geospatial ? '✓' : '—'],
//   ]
//   if (meta.gsd != null) rows.push(['GSD', `${meta.gsd} m`])
//   if (meta.crs) rows.push(['CRS', meta.crs])

//   return (
//     <div style={{
//       background: 'var(--bg-2)', border: '1px solid var(--line-2)',
//       borderRadius: 'var(--radius-sm)', padding: '10px 14px', marginTop: 10,
//     }}>
//       <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--ink-3)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 8 }}>
//         {title}
//       </div>
//       <div style={{ display: 'grid', gridTemplateColumns: '110px 1fr', rowGap: 4, fontSize: 12 }}>
//         {rows.map(([k, v]) => (
//           <div key={k} style={{ display: 'contents' }}>
//             <div style={{ color: 'var(--ink-3)' }}>{k}</div>
//             <div style={{ fontFamily: 'var(--font-mono)', color: 'var(--ink-1)', wordBreak: 'break-all' }}>{v}</div>
//           </div>
//         ))}
//       </div>
//     </div>
//   )
// }

/* ─── Inference summary card ───────────────────────────────── */
function InferenceSummaryCard({ summary }) {
  if (!summary) return null
  return (
    <div style={{
      background: 'var(--bg-2)', border: '1px solid var(--line-2)',
      borderRadius: 'var(--radius-sm)', padding: '10px 14px', marginTop: 10,
    }}>
      <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--ink-3)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 8 }}>
        INFERENCE SUMMARY
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '130px 1fr', rowGap: 4, fontSize: 12 }}>
        <div style={{ color: 'var(--ink-3)' }}>Input</div>
        <div style={{ fontFamily: 'var(--font-mono)' }}>{summary.input_width} × {summary.input_height}</div>
        <div style={{ color: 'var(--ink-3)' }}>Scale</div>
        <div style={{ fontFamily: 'var(--font-mono)' }}>×{summary.scale}</div>
        <div style={{ color: 'var(--ink-3)' }}>Output</div>
        <div style={{ fontFamily: 'var(--font-mono)', color: 'var(--cobalt-deep)', fontWeight: 600 }}>
          {summary.output_width} × {summary.output_height}
        </div>
        <div style={{ color: 'var(--ink-3)' }}>Patch</div>
        <div style={{ fontFamily: 'var(--font-mono)' }}>{summary.patch_size} × {summary.patch_size}</div>
        <div style={{ color: 'var(--ink-3)' }}>Overlap</div>
        <div style={{ fontFamily: 'var(--font-mono)' }}>{summary.overlap} px</div>
        <div style={{ color: 'var(--ink-3)' }}>Estimated patches</div>
        <div style={{ fontFamily: 'var(--font-mono)' }}>
          {summary.total_patches.toLocaleString()}
          <span style={{ color: 'var(--ink-3)', marginLeft: 6 }}>
            ({summary.patches_x} × {summary.patches_y})
          </span>
        </div>
      </div>
    </div>
  )
}

/* ─── Metrics table ─────────────────────────────────────────── */
function MetricsTable({ metrics }) {
  if (!metrics) return null
  const { sr, lr_bicubic, delta } = metrics
  const fmt = (v) => (typeof v === 'number' && isFinite(v)) ? v.toFixed(4) : '—'

  return (
    <div style={{ marginTop: 20 }}>
      <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 10, color: 'var(--ink-1)' }}>Metrics</div>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid var(--line-2)' }}>
              <th style={{ textAlign: 'left', padding: '6px 10px', color: 'var(--ink-2)', fontWeight: 600 }}>Row</th>
              {METRIC_LABELS.map(l => (
                <th key={l} style={{ textAlign: 'right', padding: '6px 10px', color: 'var(--ink-2)', fontWeight: 600 }}>{l}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr style={{ borderBottom: '1px solid var(--line-2)' }}>
              <td style={{ padding: '6px 10px', fontWeight: 600, color: 'var(--cobalt-deep)' }}>SR</td>
              {METRIC_NAMES.map(k => (
                <td key={k} style={{ textAlign: 'right', padding: '6px 10px', fontFamily: 'monospace', color: 'var(--ink-1)' }}>{fmt(sr?.[k])}</td>
              ))}
            </tr>
            <tr style={{ borderBottom: '1px solid var(--line-2)' }}>
              <td style={{ padding: '6px 10px', fontWeight: 500, color: 'var(--ink-2)' }}>Bicubic LR</td>
              {METRIC_NAMES.map(k => (
                <td key={k} style={{ textAlign: 'right', padding: '6px 10px', fontFamily: 'monospace', color: 'var(--ink-2)' }}>{fmt(lr_bicubic?.[k])}</td>
              ))}
            </tr>
            <tr>
              <td style={{ padding: '6px 10px', fontWeight: 600, color: 'var(--ink-1)' }}>Δ (SR − LR)</td>
              {METRIC_NAMES.map(k => {
                const d = delta?.[k]
                const lowerBetter = LOWER_IS_BETTER.has(k)
                const improved = typeof d === 'number' && (lowerBetter ? d < 0 : d > 0)
                const worsened = typeof d === 'number' && (lowerBetter ? d > 0 : d < 0)
                return (
                  <td key={k} style={{
                    textAlign: 'right', padding: '6px 10px', fontFamily: 'monospace', fontWeight: 600,
                    color: improved ? 'var(--ok)' : worsened ? 'var(--bad)' : 'var(--ink-2)',
                  }}>
                    {typeof d === 'number' && isFinite(d) ? (d >= 0 ? '+' : '') + d.toFixed(4) : '—'}
                  </td>
                )
              })}
            </tr>
          </tbody>
        </table>
      </div>
      <div style={{ marginTop: 10, fontSize: 11, color: 'var(--ink-3)', display: 'flex', gap: 12, flexWrap: 'wrap' }}>
        {METRIC_NAMES.map(k => {
          const d = delta?.[k]
          if (typeof d !== 'number' || !isFinite(d)) return null
          const lowerBetter = LOWER_IS_BETTER.has(k)
          const improved = lowerBetter ? d < 0 : d > 0
          return (
            <span key={k} style={{ color: improved ? 'var(--ok)' : 'var(--bad)' }}>
              Δ{k.toUpperCase().replace('_', '-')} {d >= 0 ? '+' : ''}{d.toFixed(3)}
            </span>
          )
        })}
      </div>
    </div>
  )
}

/* ─── Image viewer ──────────────────────────────────────────── */
function ImageViewer({ images }) {
  const [lightbox, setLightbox] = useState(null)
  return (
    <>
      <div style={{ marginTop: 20 }}>
        <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 10, color: 'var(--ink-1)' }}>Result Images</div>
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          {images.map(({ label, url }) => (
            <div key={label} style={{ flex: '1 1 140px', minWidth: 120, cursor: 'pointer' }} onClick={() => setLightbox(url)}>
              <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--ink-2)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.05em' }}>{label}</div>
              <img src={url} alt={label}
                style={{ width: '100%', borderRadius: 'var(--radius-sm)', border: '1px solid var(--line-2)', objectFit: 'cover', aspectRatio: '1/1', background: 'var(--bg-2)' }}
                onError={e => { e.target.style.opacity = 0.3 }}
              />
              <div style={{ fontSize: 10, color: 'var(--ink-3)', marginTop: 4, textAlign: 'center' }}>Click to enlarge</div>
            </div>
          ))}
        </div>
      </div>
      {lightbox && (
        <div onClick={() => setLightbox(null)} style={{
          position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.85)', zIndex: 9999,
          display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'zoom-out',
        }}>
          <img src={lightbox} alt="enlarged"
            style={{ maxWidth: '90vw', maxHeight: '90vh', borderRadius: 8, boxShadow: '0 0 60px rgba(0,0,0,0.8)' }} />
          <div style={{ position: 'absolute', top: 20, right: 24, color: '#fff', fontSize: 28, fontWeight: 300, lineHeight: 1 }}>×</div>
        </div>
      )}
    </>
  )
}

/* ─── LR / SR slider comparison ─────────────────────────────── */
function ImageCompareSlider({ lrUrl, srUrl, title = 'LR vs SR Comparison' }) {
  const [lrLoaded, setLrLoaded] = useState(false)
  const [srLoaded, setSrLoaded] = useState(false)
  const [lrErr, setLrErr] = useState(false)
  const [srErr, setSrErr] = useState(false)
  const [fullscreen, setFullscreen] = useState(false)

  useEffect(() => {
    setLrLoaded(false); setSrLoaded(false); setLrErr(false); setSrErr(false)
    const imgLr = new Image(); imgLr.src = lrUrl
    imgLr.onload = () => setLrLoaded(true)
    imgLr.onerror = () => setLrErr(true)
    const imgSr = new Image(); imgSr.src = srUrl
    imgSr.onload = () => setSrLoaded(true)
    imgSr.onerror = () => setSrErr(true)
  }, [lrUrl, srUrl])

  useEffect(() => {
    if (!fullscreen) return
    const handler = (e) => { if (e.key === 'Escape') setFullscreen(false) }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [fullscreen])

  const ready = lrLoaded && srLoaded
  const hasError = lrErr || srErr

  const sliderNode = (
    <ReactCompareImage
      leftImage={lrUrl} rightImage={srUrl}
      leftImageLabel="LR" rightImageLabel="SR"
      sliderLineColor="var(--cobalt-deep)"
      sliderHandleColor="var(--cobalt-deep)"
    />
  )

  return (
    <>
      <div style={{ marginTop: 24 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
          <div style={{ fontWeight: 600, fontSize: 13, color: 'var(--ink-1)' }}>{title}</div>
          {ready && (
            <button type="button" onClick={() => setFullscreen(true)}
              style={{
                display: 'flex', alignItems: 'center', gap: 5,
                padding: '4px 12px', fontSize: 12, fontWeight: 500, cursor: 'pointer',
                borderRadius: 'var(--radius-sm)',
                background: 'var(--cobalt-soft)', color: 'var(--cobalt-deep)',
                border: '1px solid var(--cobalt-deep)',
              }}>
              <span style={{ fontSize: 13 }}>&#x26F6;</span> Fullscreen
            </button>
          )}
        </div>
        <div style={{
          borderRadius: 'var(--radius-sm)', overflow: 'hidden',
          border: '1px solid var(--line-2)', background: 'var(--bg-2)',
          minHeight: 200, display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          {hasError ? (
            <div style={{ fontSize: 12, color: 'var(--bad)', padding: 20 }}>Could not load one or both images for comparison.</div>
          ) : !ready ? (
            <div style={{ fontSize: 12, color: 'var(--ink-3)', padding: 20 }}>Loading images...</div>
          ) : sliderNode}
        </div>
        <div style={{ fontSize: 11, color: 'var(--ink-3)', marginTop: 6, textAlign: 'center' }}>
          Drag the slider to compare LR (left) and SR (right)
        </div>
      </div>
      {fullscreen && (
        <div onClick={() => setFullscreen(false)} style={{
          position: 'fixed', inset: 0, zIndex: 100000, background: '#0a0a0a',
          display: 'flex', flexDirection: 'column',
        }}>
          <div onClick={e => e.stopPropagation()} style={{
            flexShrink: 0, height: 52, display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            padding: '0 20px', background: 'rgba(255,255,255,0.05)', borderBottom: '1px solid rgba(255,255,255,0.1)',
          }}>
            <span style={{ color: '#fff', fontWeight: 600, fontSize: 14 }}>{title}</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
              <span style={{ color: 'rgba(255,255,255,0.35)', fontSize: 12 }}>Drag handle to compare · ESC to close</span>
              <button type="button" onClick={() => setFullscreen(false)}
                style={{
                  background: 'rgba(255,255,255,0.1)', border: '1px solid rgba(255,255,255,0.2)',
                  color: '#fff', borderRadius: 6, padding: '4px 14px', cursor: 'pointer', fontSize: 13,
                }}>✕ Close</button>
            </div>
          </div>
          <div onClick={e => e.stopPropagation()} style={{ flex: 1, overflow: 'hidden', display: 'flex', alignItems: 'stretch' }}>
            <ReactCompareImage leftImage={lrUrl} rightImage={srUrl} leftImageLabel="LR" rightImageLabel="SR"
              sliderLineColor="#5b9cf6" sliderHandleColor="#5b9cf6" />
          </div>
          <div onClick={e => e.stopPropagation()} style={{
            flexShrink: 0, height: 36, display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            padding: '0 28px', fontSize: 12, color: 'rgba(255,255,255,0.35)', borderTop: '1px solid rgba(255,255,255,0.07)',
          }}>
            <span>◀ LR (Low Resolution)</span>
            <span>SR (Super-Resolved) ▶</span>
          </div>
        </div>
      )}
    </>
  )
}

/* ─── Band checkbox selector with descriptions ────────────── */
const DEFAULT_BAND_LABELS = { 1: 'Red', 2: 'Green', 3: 'Blue', 4: 'NIR' }

function BandCheckboxSelector({ totalBands, selectedBands, onChange, label, descriptions }) {
  const toggle = (b) => {
    if (selectedBands.includes(b)) {
      onChange(selectedBands.filter(x => x !== b))
    } else {
      onChange([...selectedBands, b])
    }
  }
  const all = Array.from({ length: totalBands }, (_, i) => i + 1)
  return (
    <div className="form-group">
      <label>{label} <span className="hint">click to select · order = display channel order</span></label>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 6 }}>
        {all.map(b => {
          const pos = selectedBands.indexOf(b)
          const checked = pos !== -1
          const desc = descriptions?.[b - 1] || (totalBands <= 4 ? DEFAULT_BAND_LABELS[b] : null)
          return (
            <button key={b} type="button" onClick={() => toggle(b)}
              title={desc || `Band ${b}`}
              style={{
                padding: '4px 10px', fontSize: 12, borderRadius: 'var(--radius-sm)',
                border: `1px solid ${checked ? 'var(--cobalt-deep)' : 'var(--line-2)'}`,
                background: checked ? 'var(--cobalt-soft)' : 'var(--surface)',
                color: checked ? 'var(--cobalt-deep)' : 'var(--ink-3)',
                fontWeight: checked ? 600 : 400, cursor: 'pointer', transition: 'all 0.12s',
                position: 'relative',
              }}>
              {checked && (
                <span style={{
                  position: 'absolute', top: -6, right: -4, fontSize: 9, fontWeight: 700,
                  background: 'var(--cobalt-deep)', color: '#fff', borderRadius: 6,
                  padding: '1px 4px', lineHeight: 1,
                }}>{pos + 1}</span>
              )}
              Band {b}{desc ? ` (${desc})` : ''}
            </button>
          )
        })}
      </div>
      <div style={{ fontSize: 11, color: 'var(--ink-3)' }}>
        Selected (display order): [{selectedBands.join(', ')}]
      </div>
    </div>
  )
}

/* ─── Per-band image viewer ─────────────────────────────────── */
function BandImageViewer({ jobId, nBands, lrBands, hrBands, paired }) {
  const [lightbox, setLightbox] = useState(null)
  if (!jobId || nBands < 1) return null

  const makeRow = (prefix, bands, title) => (
    <div style={{ marginBottom: 16 }}>
      <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--ink-2)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 6 }}>{title}</div>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        {Array.from({ length: nBands }, (_, i) => {
          const spectral = bands?.[i]
          const url = getRawResultImageUrl(jobId, `${prefix}_band_${i + 1}.png`)
          const label = spectral ? `Spectral ${spectral}` : `Band ${i + 1}`
          return (
            <div key={i} style={{ flex: '1 1 100px', minWidth: 90, cursor: 'pointer' }}
              onClick={() => setLightbox({ url, label })}>
              <img src={url} alt={label}
                style={{ width: '100%', borderRadius: 'var(--radius-sm)', border: '1px solid var(--line-2)', objectFit: 'cover', aspectRatio: '1/1', background: 'var(--bg-2)', filter: 'grayscale(1)' }}
                onError={e => { e.target.style.opacity = 0.25 }} />
              <div style={{ fontSize: 10, color: 'var(--ink-3)', marginTop: 3, textAlign: 'center' }}>{label}</div>
            </div>
          )
        })}
      </div>
    </div>
  )

  return (
    <div style={{ marginTop: 20 }}>
      <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 12, color: 'var(--ink-1)' }}>Individual Bands</div>
      {makeRow('lr', lrBands, 'LR — input bands')}
      {makeRow('sr', lrBands, 'SR — output bands')}
      {paired && makeRow('hr', hrBands, 'HR — reference bands')}
      {lightbox && (
        <div onClick={() => setLightbox(null)} style={{
          position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.87)', zIndex: 9999,
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', cursor: 'zoom-out',
        }}>
          <div style={{ color: '#fff', fontSize: 14, fontWeight: 600, marginBottom: 10 }}>{lightbox.label}</div>
          <img src={lightbox.url} alt={lightbox.label} style={{ maxWidth: '88vw', maxHeight: '82vh', borderRadius: 8 }} />
          <div style={{ color: 'rgba(255,255,255,0.4)', fontSize: 11, marginTop: 8 }}>click to close</div>
        </div>
      )}
    </div>
  )
}

/* ─── Per-band metrics table ────────────────────────────────── */
function PerBandMetricsTable({ perBand, lrBands }) {
  if (!perBand || Object.keys(perBand).length === 0) return null
  const entries = Object.entries(perBand)
  const fmt = v => (typeof v === 'number' && isFinite(v)) ? v.toFixed(4) : '—'
  return (
    <div style={{ marginTop: 20 }}>
      <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 10, color: 'var(--ink-1)' }}>Per-Band Metrics</div>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ borderBottom: '2px solid var(--line-2)' }}>
              <th style={{ textAlign: 'left', padding: '6px 10px', color: 'var(--ink-2)', fontWeight: 600 }}>Band</th>
              <th style={{ textAlign: 'right', padding: '6px 10px', color: 'var(--ink-2)', fontWeight: 600 }}>PSNR</th>
              <th style={{ textAlign: 'right', padding: '6px 10px', color: 'var(--ink-2)', fontWeight: 600 }}>SSIM</th>
            </tr>
          </thead>
          <tbody>
            {entries.map(([key, vals], i) => {
              const spectral = lrBands?.[i]
              const rowLabel = spectral ? `Spectral ${spectral} (ch ${i + 1})` : key
              return (
                <tr key={key} style={{ borderBottom: '1px solid var(--line-2)' }}>
                  <td style={{ padding: '6px 10px', color: 'var(--ink-2)' }}>{rowLabel}</td>
                  <td style={{ textAlign: 'right', padding: '6px 10px', fontFamily: 'monospace', color: 'var(--ink-1)' }}>{fmt(vals.psnr)}</td>
                  <td style={{ textAlign: 'right', padding: '6px 10px', fontFamily: 'monospace', color: 'var(--ink-1)' }}>{fmt(vals.ssim)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

/* ─── Patched metrics parsing ──────────────────────────────── */
function parsePatchedMetrics(lines) {
  const result = { sr: {}, lr_bicubic: {}, delta: {} }
  let section = null

  for (const line of lines) {
    const l = line.toLowerCase()
    if (l.includes('average sr metrics')) { section = 'sr'; continue }
    if (l.includes('average lr') || l.includes('average bicubic')) { section = 'lr'; continue }
    if (l.includes('average delta')) { section = 'delta'; continue }

    if (section) {
      const kv = line.match(/[Δ]?(\w+):\s*([+-]?[\d.]+)/)
      if (kv) {
        const rawKey = kv[1].toLowerCase()
        const key = section === 'delta' ? rawKey.replace(/^d/, '') : rawKey
        const val = parseFloat(kv[2])
        if (!isNaN(val)) {
          if (section === 'sr') result.sr[key] = val
          else if (section === 'lr') result.lr_bicubic[key] = val
          else if (section === 'delta') result.delta[key] = val
        }
      }
      if (line.trim() === '' || l.includes('===') || l.includes('---')) section = null
    }
  }
  const hasData = Object.keys(result.sr).length > 0
  return hasData ? result : null
}

/* ─── Visual assessment ────────────────────────────────────── */
const VISUAL_REPORT_MARKER_RE = /PREVIEW_READY (\S+) (\S+) (\S+)/
const VISUAL_REPORT_STAGES = ['grid', 'fft', 'errormap', 'residual']
const VISUAL_REPORT_STAGE_LABELS = {
  grid: 'LR / SR / HR', fft: 'FFT spectrum', errormap: 'Error map', residual: 'Residual map',
}
const INFERENCE_PREVIEW_STAGES = VISUAL_REPORT_STAGES.map(key => ({ key, label: VISUAL_REPORT_STAGE_LABELS[key] }))

const CLIENT_PSNR_GAIN_THRESHOLD = 1.5
const CLIENT_SSIM_FLOOR = 0.85

function parseVisualReportSamples(lines, jobId) {
  const bySample = {}
  for (const line of lines) {
    const match = line.match(VISUAL_REPORT_MARKER_RE)
    if (!match) continue
    const [, filename, stage, scene] = match
    if (!VISUAL_REPORT_STAGES.includes(stage)) continue
    if (!bySample[scene]) bySample[scene] = { scene }
    bySample[scene][stage] = `/api/inference/preview/${jobId}/${encodeURIComponent(filename)}`
  }
  return Object.values(bySample)
}

function VisualAssessmentGallery({ samples }) {
  const [lightbox, setLightbox] = useState(null)
  if (!samples || samples.length === 0) return null
  return (
    <div style={{ marginTop: 20 }}>
      <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 4, color: 'var(--ink-1)' }}>
        Visual Assessment — {samples.length} sample patch{samples.length !== 1 ? 'es' : ''}
      </div>
      <div style={{ fontSize: 11, color: 'var(--ink-3)', marginBottom: 12 }}>
        Layers 3–5 of the QA framework: visual grid, FFT/radial frequency spectrum, error &amp; residual maps.
      </div>
      {samples.map(sample => (
        <div key={sample.scene} style={{ marginBottom: 18, paddingBottom: 14, borderBottom: '1px solid var(--line-2)' }}>
          <div style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--ink-2)', marginBottom: 8, wordBreak: 'break-all' }}>
            {sample.scene}
          </div>
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
            {VISUAL_REPORT_STAGES.filter(stage => sample[stage]).map(stage => (
              <div key={stage} style={{ flex: '1 1 140px', minWidth: 120, cursor: 'pointer' }}
                onClick={() => setLightbox(sample[stage])}>
                <div style={{ fontSize: 10, fontWeight: 600, color: 'var(--ink-3)', marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                  {VISUAL_REPORT_STAGE_LABELS[stage]}
                </div>
                <img src={sample[stage]} alt={stage}
                  style={{ width: '100%', borderRadius: 'var(--radius-sm)', border: '1px solid var(--line-2)', background: 'var(--bg-2)' }}
                  onError={e => { e.target.style.opacity = 0.3 }} />
              </div>
            ))}
          </div>
        </div>
      ))}
      {lightbox && (
        <div onClick={() => setLightbox(null)} style={{
          position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.85)', zIndex: 9999,
          display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'zoom-out',
        }}>
          <img src={lightbox} alt="enlarged"
            style={{ maxWidth: '90vw', maxHeight: '90vh', borderRadius: 8, boxShadow: '0 0 60px rgba(0,0,0,0.8)' }} />
        </div>
      )}
    </div>
  )
}

function PassFailBadge({ metrics }) {
  const delta = metrics?.delta
  const sr = metrics?.sr
  const psnrGain = delta?.psnr
  const srSsim = sr?.ssim
  if (typeof psnrGain !== 'number' || typeof srSsim !== 'number') return null

  const psnrPass = psnrGain >= CLIENT_PSNR_GAIN_THRESHOLD
  const ssimPass = srSsim >= CLIENT_SSIM_FLOOR
  const overallPass = psnrPass && ssimPass

  return (
    <div style={{
      marginTop: 16, padding: '10px 14px', borderRadius: 'var(--radius-sm)',
      border: `1px solid ${overallPass ? 'var(--ok)' : 'var(--bad)'}`,
      background: overallPass ? 'rgba(60, 180, 100, 0.08)' : 'rgba(220, 70, 70, 0.08)',
    }}>
      <div style={{ fontWeight: 700, fontSize: 13, color: overallPass ? 'var(--ok)' : 'var(--bad)', marginBottom: 4 }}>
        {overallPass ? '✓ PASS' : '✗ FAIL'} — client acceptance bar
      </div>
      <div style={{ fontSize: 11, color: 'var(--ink-2)', display: 'flex', gap: 16, flexWrap: 'wrap' }}>
        <span>{psnrPass ? '✓' : '✗'} PSNR gain ≥ {CLIENT_PSNR_GAIN_THRESHOLD} dB (actual: {psnrGain >= 0 ? '+' : ''}{psnrGain.toFixed(2)} dB)</span>
        <span>{ssimPass ? '✓' : '✗'} SR SSIM ≥ {CLIENT_SSIM_FLOOR} (actual: {srSsim.toFixed(3)})</span>
      </div>
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════════
   Reusable image-metadata hook
   ═══════════════════════════════════════════════════════════════ */
// function useImageMeta(path, delay = 700) {
//   const [meta, setMeta] = useState(null)
//   useEffect(() => {
//     if (!path || !path.trim()) { setMeta(null); return }
//     const t = setTimeout(() => {
//       getImageInfo(path).then(r => setMeta(r.data)).catch(() => setMeta(null))
//     }, delay)
//     return () => clearTimeout(t)
//   }, [path, delay])
//   return meta
// }

/* ═══════════════════════════════════════════════════════════════
   Reusable model auto-load hook (latest from task)
   ═══════════════════════════════════════════════════════════════ */
function useLatestModel(modelSource, selectedTask, setModelConfig) {
  const [latestInfo, setLatestInfo] = useState(null)
  useEffect(() => {
    if (modelSource === 'auto' && selectedTask) {
      getLatestModel(selectedTask)
        .then(r => {
          setLatestInfo(r.data)
          if (r.data.model_config && Object.keys(r.data.model_config).length > 0)
            setModelConfig({ ...DEFAULT_MODEL_CONFIG, ...r.data.model_config })
        })
        .catch(() => setLatestInfo(null))
    }
  }, [selectedTask, modelSource])
  return latestInfo
}

/* ═══════════════════════════════════════════════════════════════
   TAB 1: Patched Images
   ═══════════════════════════════════════════════════════════════ */
function PatchedTab({ tasks, optionsFiles, jobId, setJobId }) {
  const [modelSource, setModelSource] = useState('auto')
  const [selectedTask, setSelectedTask] = useState('')
  const [customModelPath, setCustomModelPath] = useState('')
  const [modelConfig, setModelConfig] = useState(DEFAULT_MODEL_CONFIG)
  const latestInfo = useLatestModel(modelSource, selectedTask, setModelConfig)
  const { selectedOptions, configSource, handleOptionsSelect } = useConfigAutoLoad(modelSource, customModelPath, setModelConfig)

  const [inferConfig, setInferConfig] = useState({
    lr_dir: '', hr_dir: '', sr_dir: 'testsets/output/sr',
    tile: '', tile_overlap: 32, overwrite_sr: true, log_dir: 'testsets/output',
    visual_report_enabled: true, visual_report_samples: 10, visual_report_seed: 23,
  })
  const [jobDone, setJobDone] = useState(false)
  const [patchedMetrics, setPatchedMetrics] = useState(null)
  const [visualSamples, setVisualSamples] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const allLinesRef = useRef([])
  const [cancelled, setCancelled] = useState(false)

  const setMC = (path, value) => setModelConfig(prev => deepSet(prev, path, value))
  const setIC = (key, value) => setInferConfig(prev => ({ ...prev, [key]: value }))

    // ── inside PatchedTab, add these state/props ──
  // const [patchedMetrics, setPatchedMetrics] = useState(null)
  const [patchedMetricsError, setPatchedMetricsError] = useState('')
  const [progressSummary, setProgressSummary] = useState(null)   // { output_width, output_height, total_patches }

  const jobIdRef = useRef(null)

  const handleSubmit = async (e) => {

    e.preventDefault(); setError(''); setLoading(true)
    setJobId(null); setJobDone(false); setPatchedMetrics(null); setVisualSamples([])
    allLinesRef.current = []; setPatchedMetricsError('')
    try {
      const modelPath = modelSource === 'auto' ? (latestInfo?.model_path || '') : customModelPath
      const payload = {
        model_path: modelPath, lr_dir: inferConfig.lr_dir, hr_dir: inferConfig.hr_dir,
        sr_dir: inferConfig.sr_dir, tile: inferConfig.tile ? parseInt(inferConfig.tile) : null,
        tile_overlap: inferConfig.tile_overlap, overwrite_sr: inferConfig.overwrite_sr,
        log_dir: inferConfig.log_dir, model_config: modelConfig,
        visual_report_enabled: inferConfig.visual_report_enabled,
        visual_report_samples: inferConfig.visual_report_samples,
        visual_report_seed: inferConfig.visual_report_seed,
      }
      const r = await startInference(payload)
      jobIdRef.current = r.data.job_id
      setJobId(r.data.job_id)
      // after the existing useEffect / hooks, fetch summary once lr_dir is set
      useEffect(() => {
        // Only when tile size is provided can we estimate patch count for a directory job;
        // otherwise we show the raw tile params only.
        setProgressSummary(null)
      }, [inferConfig.lr_dir, inferConfig.tile, inferConfig.tile_overlap, modelConfig.upscale])
    } catch (err) {
      setError(err.response?.data?.detail || String(err))
    } finally { setLoading(false) }
  }

  const handleLogLine = (line) => { allLinesRef.current.push(line) }
  const handleComplete = () => {
    // Ignore terminal events for a job we no longer own.
    if (jobIdRef.current !== jobId) return
    setJobDone(true)
    const parsed = parsePatchedMetrics(allLinesRef.current)
    if (parsed) setPatchedMetrics(parsed)
    if (jobId) setVisualSamples(parseVisualReportSamples(allLinesRef.current, jobId))
  }

  return (
    <div className="module-grid rise" style={{ animationDelay: '100ms' }}>
      <div className="col">
        <form onSubmit={handleSubmit}>
          <ModelSelectionCard
            modelSource={modelSource} setModelSource={setModelSource}
            tasks={tasks} selectedTask={selectedTask} setSelectedTask={setSelectedTask}
            latestInfo={latestInfo} customModelPath={customModelPath} setCustomModelPath={setCustomModelPath}
            optionsFiles={optionsFiles} selectedOptions={selectedOptions}
            onOptionsSelect={handleOptionsSelect} configSource={configSource}
            modelConfig={modelConfig}
          />
          <CollapsibleSection title="Input / Output Paths" defaultOpen>
            <PathField label="LR image dir" mode="dirs" value={inferConfig.lr_dir}
              onChange={v => setIC('lr_dir', v)} placeholder="testsets/my_test/lr" />
            <PathField label="HR image dir" mode="dirs" hint="ground truth for metrics"
              value={inferConfig.hr_dir} onChange={v => setIC('hr_dir', v)} placeholder="testsets/my_test/hr" />
            <PathField label="SR output dir" mode="dirs" value={inferConfig.sr_dir}
              onChange={v => setIC('sr_dir', v)} placeholder="testsets/my_test/sr" />
            <PathField label="Log dir" mode="dirs" value={inferConfig.log_dir}
              onChange={v => setIC('log_dir', v)} placeholder="testsets/my_test" />
            <div className="grid-2">
              <div className="form-group">
                <label>Tile size <span className="hint">blank = full image</span></label>
                <input type="number" className="num-input" value={inferConfig.tile || ''}
                  onChange={e => setIC('tile', e.target.value)} placeholder="e.g. 256" min={64} step={8} />
              </div>
              <NumberField label="Tile overlap" value={inferConfig.tile_overlap}
                onChange={v => setIC('tile_overlap', v)} min={0} step={8} />
            </div>
            <BoolToggle label="Overwrite existing SR images" value={inferConfig.overwrite_sr}
              onChange={v => setIC('overwrite_sr', v)} />
          </CollapsibleSection>
          <CollapsibleSection title="Visual Assessment (5-layer QA)" defaultOpen>
            <BoolToggle label="Generate visual grids, FFT spectrum, error &amp; residual maps"
              value={inferConfig.visual_report_enabled}
              onChange={v => setIC('visual_report_enabled', v)}
              tooltip="Adds the client's remaining QA layers (visual grid, FFT/radial spectrum, error map, residual map) for a random sample of patches." />
            {inferConfig.visual_report_enabled && (
              <div className="grid-2">
                <NumberField label="Sample patches" value={inferConfig.visual_report_samples}
                  onChange={v => setIC('visual_report_samples', v)} min={1} max={50} step={1} />
                <NumberField label="Random seed" value={inferConfig.visual_report_seed}
                  onChange={v => setIC('visual_report_seed', v)} min={0} step={1} />
              </div>
            )}
          </CollapsibleSection>
          {error && <div style={{ color: 'var(--bad)', fontSize: 13, marginBottom: 12 }}>{error}</div>}
          <button type="submit" className="btn btn-primary full-width"
            disabled={loading || (modelSource === 'auto' && !selectedTask)}>
            {loading ? 'Starting…' : '▶ Run Super-Resolution'}
          </button>

          <InlineJobPanel
            jobId={jobId}
            running={!!jobId && !jobDone && !cancelled}
            cancelled={cancelled}
            onStop={async () => {
              setCancelled(true)
              try { await stopInference(jobId) } catch { /* ignore */ }
            }}
            // onStop={() => stopInference(jobId).catch(() => { })}
            progressFetcher={getInferenceProgress}    // ← ADD
            outputDir={inferConfig.sr_dir}
            metrics={jobDone ? patchedMetrics : null}
            metricsError={patchedMetricsError}
            onRetryMetrics={() => {
              const parsed = parsePatchedMetrics(allLinesRef.current)
              if (parsed) setPatchedMetrics(parsed)
              else setPatchedMetricsError('Could not parse metrics from log.')
            }}
            scale={modelConfig.upscale}
            patchSize={inferConfig.tile ? parseInt(inferConfig.tile) : null}
            overlap={inferConfig.tile_overlap}
            acceptBar={{ psnrGain: 1.5, ssimFloor: 0.85 }}
          />
        </form>
      </div>
      <div className="col">
        <ModelArchCard modelConfig={modelConfig} setMC={setMC} configSource={configSource} />

        {jobId && (
          <CollapsibleSection
            title={
              <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                Live Output
                <LiveOutputBadge
                  status={
                    cancelled   ? 'cancelled'
                    : jobDone   ? 'completed'
                    : jobId     ? 'running'
                    :           'completed'
                  }
                />
              </span>
            }
            defaultOpen={false}
          >
            <LogConsole
              domain="inference"
              jobId={jobId}
              onStop={async () => {
                setCancelled(true)                        // ← flips badge immediately
                try { await stopInference(jobId) } catch { }
              }}
              onComplete={() => {
                if (jobIdRef.current !== jobId) return
                setJobDone(true)
              }}
              showControls={{ copy: true, pause: false, 
                      cancel: false, elapsed: false }}
            />
          </CollapsibleSection>
        )}

        {jobDone && visualSamples.length > 0 && (
          <div className="card" style={{ marginTop: 16 }}>
            <div className="card-title">Visual Assessment</div>
            <VisualAssessmentGallery samples={visualSamples} />
          </div>
        )}

        {!jobId && (
          <div className="card">
            <div className="card-title">About</div>
            <p className="text-muted text-sm" style={{ lineHeight: 1.5 }}>
              Runs SwinIR on a directory of pre-aligned LR image patches, producing SR outputs and computing
              8 quality metrics — PSNR, SSIM, IT-SSIM, SAM, UIQI, RMSE, FSIM, SRER — per image against the
              HR ground truth directory. A bicubic baseline is also computed so the model improvement can be measured.
            </p>
            <div className="mono" style={{ fontSize: 11, color: 'var(--ink-3)', marginTop: 12, padding: 8, background: 'var(--bg)', borderRadius: 'var(--radius-sm)' }}>
              Per-image rows + average summary — results saved to: log_dir/&lt;sr_dir_name&gt;.log
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════════
   TAB 2: Raw HR+LR Inference
   ═══════════════════════════════════════════════════════════════ */
function RawPairedTab({ tasks, optionsFiles, jobId, setJobId }) {
  const [modelSource, setModelSource] = useState('auto')
  const [selectedTask, setSelectedTask] = useState('')
  const [customModelPath, setCustomModelPath] = useState('')
  const [modelConfig, setModelConfig] = useState(DEFAULT_MODEL_CONFIG)
  const latestInfo = useLatestModel(modelSource, selectedTask, setModelConfig)
  const { selectedOptions, configSource, handleOptionsSelect } = useConfigAutoLoad(modelSource, customModelPath, setModelConfig)

  const [coreg, setCoreg] = useState(DEFAULT_COREG)
  const [config, setConfig] = useState({
    lr_path: '', hr_path: '', lr_bands: [3, 2, 1], hr_bands: [1, 2, 3],
    output_dir: 'testsets/raw_inference_output', patch_size: 256,
    overlap: 0, scale_factor: 2,
    visual_report_enabled: true, visual_report_samples: 10, visual_report_seed: 23,
  })
  const [jobDone, setJobDone] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [metrics, setMetrics] = useState(null)
  const [metricsError, setMetricsError] = useState('')
  const [visualSamples, setVisualSamples] = useState([])
  const [summary, setSummary] = useState(null)
  const jobIdRef = useRef(null)
  const allLinesRef = useRef([])

  const setMC = (path, value) => setModelConfig(prev => deepSet(prev, path, value))
  const setCfg = (key, value) => setConfig(prev => ({ ...prev, [key]: value }))

  const lrMeta = useImageMeta(config.lr_path)
  const hrMeta = useImageMeta(config.hr_path)
  const [cancelled, setCancelled] = useState(false)

  // Fetch inference summary whenever LR meta / patch params change
  useEffect(() => {
    if (!lrMeta?.width || !lrMeta?.height) { setSummary(null); return }
    const t = setTimeout(() => {
      getInferenceSummary({
        width: lrMeta.width, height: lrMeta.height,
        patch_size: config.patch_size, overlap: config.overlap,
        scale: config.scale_factor,
      }).then(r => setSummary(r.data)).catch(() => setSummary(null))
    }, 300)
    return () => clearTimeout(t)
  }, [lrMeta?.width, lrMeta?.height, config.patch_size, config.overlap, config.scale_factor])

  // Sync scale_factor from model config (authoritative)
  useEffect(() => {
    if (modelConfig?.upscale && modelConfig.upscale !== config.scale_factor) {
      setCfg('scale_factor', modelConfig.upscale)
    }
  }, [modelConfig?.upscale])

  const fetchMetrics = async (jid) => {
    setMetricsError('')
    try {
      await new Promise(r => setTimeout(r, 800))
      const m = await getRawInferenceMetrics(jid)
      setMetrics(m.data)
    } catch (err) {
      const detail = err.response?.data?.detail || String(err)
      setMetricsError(`Could not load metrics: ${detail}`)
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault(); setError(''); setLoading(true)
    setJobId(null); setJobDone(false); setMetrics(null); setMetricsError(''); setVisualSamples([])
    allLinesRef.current = []
    try {
      const modelPath = modelSource === 'auto' ? (latestInfo?.model_path || '') : customModelPath
      const payload = { ...config, model_path: modelPath, model_network_config: modelConfig, coreg }
      const r = await startRawPairedInference(payload)
      jobIdRef.current = r.data.job_id
      setJobId(r.data.job_id)
    } catch (err) {
      setError(err.response?.data?.detail || String(err))
    } finally { setLoading(false) }
  }

  const handleLogLine = (line) => { 
    if (jobIdRef.current !== jobId) return
    allLinesRef.current.push(line) 
  }

  const resultImages = jobDone && jobId ? [
    { label: 'LR', url: getRawResultImageUrl(jobId, 'lr_display.png') },
    { label: 'SR', url: getRawResultImageUrl(jobId, 'sr_display.png') },
    { label: 'HR', url: getRawResultImageUrl(jobId, 'hr_display.png') },
  ] : []

  return (
    <div className="module-grid rise" style={{ animationDelay: '100ms' }}>
      <div className="col">
        <form onSubmit={handleSubmit}>
          <ModelSelectionCard
            modelSource={modelSource} setModelSource={setModelSource}
            tasks={tasks} selectedTask={selectedTask} setSelectedTask={setSelectedTask}
            latestInfo={latestInfo} customModelPath={customModelPath} setCustomModelPath={setCustomModelPath}
            optionsFiles={optionsFiles} selectedOptions={selectedOptions}
            onOptionsSelect={handleOptionsSelect} configSource={configSource}
            modelConfig={modelConfig}
          />

          <div className="card" style={{ marginBottom: 16 }}>
            <div className="card-title">Input</div>
            <PathField label="LR image" mode="files" extensions=".tif,.tiff,.jp2,.img,.png,.jpg,.jpeg,.bmp"
              value={config.lr_path} onChange={v => setCfg('lr_path', v)} placeholder="/path/to/lr.tif" mono />
            {lrMeta && <ImageInfoCard meta={lrMeta} />}
            <PathField label="HR image" mode="files" extensions=".tif,.tiff,.jp2,.img,.png,.jpg,.jpeg,.bmp"
              hint="ground truth" value={config.hr_path} onChange={v => setCfg('hr_path', v)}
              placeholder="/path/to/hr.tif" mono />
            {hrMeta && <ImageInfoCard meta={hrMeta} title="HR REFERENCE INFORMATION" />}
            {lrMeta ? (
              <BandCheckboxSelector totalBands={lrMeta.bands} selectedBands={config.lr_bands}
                onChange={v => setCfg('lr_bands', v)} label="LR display bands"
                descriptions={lrMeta.band_descriptions} />
            ) : (
              <ArrayEditor label="LR RGB bands" value={config.lr_bands} onChange={v => setCfg('lr_bands', v)} />
            )}
            {hrMeta ? (
              <BandCheckboxSelector totalBands={hrMeta.bands} selectedBands={config.hr_bands}
                onChange={v => setCfg('hr_bands', v)} label="HR display bands"
                descriptions={hrMeta.band_descriptions} />
            ) : (
              <ArrayEditor label="HR RGB bands" value={config.hr_bands} onChange={v => setCfg('hr_bands', v)} />
            )}
          </div>

          <div className="card" style={{ marginBottom: 16 }}>
            <div className="card-title">Inference</div>
            <div className="grid-2">
              <div className="form-group">
                <label>Scale <span className="hint">from model config</span></label>
                <input className="num-input" value={`×${config.scale_factor}`} readOnly
                  style={{ background: 'var(--bg-2)', color: 'var(--ink-2)' }} />
              </div>
              <NumberField label="Overlap (LR px)" value={config.overlap}
                onChange={v => setCfg('overlap', v)} min={0} step={8} />
            </div>
            <NumberField label="Patch size (LR px)" value={config.patch_size}
              onChange={v => setCfg('patch_size', v)} min={32} step={8} />
            <PathField label="Output directory" mode="dirs"
              value={config.output_dir} onChange={v => setCfg('output_dir', v)}
              placeholder="testsets/raw_inference_output" mono />
            <div style={{
              display: 'flex', alignItems: 'center', gap: 8, marginTop: 8,
              padding: '8px 12px', borderRadius: 'var(--radius-sm)',
              background: 'var(--bg-2)', border: '1px solid var(--line-2)', fontSize: 11,
              color: 'var(--ink-3)',
            }}>
              <span>✓ Preserve geospatial metadata · CRS · transform (GeoTIFF output)</span>
            </div>
            <InferenceSummaryCard summary={summary} />
          </div>

          <CollapsibleSection title="Visual Assessment (5-layer QA)" defaultOpen={false}>
            <BoolToggle label="Generate visual grids, FFT spectrum, error &amp; residual maps"
              value={config.visual_report_enabled}
              onChange={v => setCfg('visual_report_enabled', v)}
              tooltip="Adds the client's remaining QA layers (visual grid, FFT/radial spectrum, error map, residual map) for a random sample of patch-sized crops within the stitched scene." />
            {config.visual_report_enabled && (
              <div className="grid-2">
                <NumberField label="Sample patches" value={config.visual_report_samples}
                  onChange={v => setCfg('visual_report_samples', v)} min={1} max={50} step={1} />
                <NumberField label="Random seed" value={config.visual_report_seed}
                  onChange={v => setCfg('visual_report_seed', v)} min={0} step={1} />
              </div>
            )}
          </CollapsibleSection>

          <CoregSection coreg={coreg} setCoreg={setCoreg} />

          {error && <div style={{ color: 'var(--bad)', fontSize: 13, marginBottom: 12 }}>{error}</div>}
          <button type="submit" className="btn btn-primary full-width"
            disabled={loading || !config.lr_path || !config.hr_path || (modelSource === 'auto' && !selectedTask)}>
            {loading ? 'Starting…' : '▶ Run Super-Resolution'}
          </button>

          <InlineJobPanel
            jobId={jobId}
            running={!!jobId && !jobDone && !cancelled}
            cancelled={cancelled}
            onStop={async () => {
              setCancelled(true)
              try { await stopInference(jobId) } catch { /* ignore */ }
            }}
            // onStop={() => stopInference(jobId).catch(() => { })}
            progressFetcher={getInferenceProgress}    // ← ADD
            outputDir={config.output_dir}
            metrics={jobDone ? metrics : null}
            metricsError={metricsError}
            onRetryMetrics={() => fetchMetrics(jobIdRef.current)}
            scale={config.scale_factor}
            patchSize={config.patch_size}
            overlap={config.overlap}
            inputDims={lrMeta ? { width: lrMeta.width, height: lrMeta.height } : null}
            estimatedPatches={summary?.total_patches ?? null}
            acceptBar={{ psnrGain: 1.5, ssimFloor: 0.85 }}
          />
        </form>
      </div>

      <div className="col">
        <ModelArchCard modelConfig={modelConfig} setMC={setMC} configSource={configSource} />
        {jobId && (
          <CollapsibleSection
            title={
              <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                Live Output
                <LiveOutputBadge
                  status={
                    cancelled   ? 'cancelled'
                    : jobDone   ? 'completed'
                    : jobId     ? 'running'
                    :           'completed'
                  }
                />
              </span>
            }
            defaultOpen={false}
          >
            <LogConsole
              domain="inference"
              jobId={jobId}
              onStop={async () => {
                setCancelled(true)                        // ← flips badge immediately
                try { await stopInference(jobId) } catch { }
              }}
              onComplete={() => {
                if (jobIdRef.current !== jobId) return
                setJobDone(true)
              }}
              showControls={{ copy: true, pause: false, 
                      cancel: false, elapsed: false }}
            />
          </CollapsibleSection>
        )}
        {jobDone && (
          <div className="card" style={{ marginTop: 16 }}>
            <div className="card-title">Results</div>
            {resultImages.length > 0 && <ImageViewer images={resultImages} />}
            {jobId && (
              <ImageCompareSlider
                lrUrl={getRawResultImageUrl(jobId, 'lr_display.png')}
                srUrl={getRawResultImageUrl(jobId, 'sr_display.png')}
                title="LR ↔ SR Comparison"
              />
            )}
            <BandImageViewer
              jobId={jobId} nBands={config.lr_bands.length}
              lrBands={config.lr_bands} hrBands={config.hr_bands} paired />
            <VisualAssessmentGallery samples={visualSamples} />
            <PerBandMetricsTable perBand={metrics?.per_band} lrBands={config.lr_bands} />
          </div>
        )}
        {!jobId && (
          <div className="card">
            <div className="card-title">About</div>
            <p className="text-muted text-sm" style={{ lineHeight: 1.6 }}>
              Provide a raw LR satellite image and a co-located HR ground truth image (GeoTIFF, JP2, or standard formats).
              The mode optionally runs the full coregistration stack — CRS reprojection, ORB keypoint alignment,
              phase cross-correlation sub-pixel correction, radiometric regression and histogram matching —
              to precisely align the LR image onto the HR pixel grid before inference.
            </p>
            <p className="text-muted text-sm" style={{ lineHeight: 1.6, marginTop: 10 }}>
              The LR image is then rescaled to an <strong>exact integer ratio</strong> relative to the
              HR dimensions, regardless of the real sensor resolution difference (e.g. 1.7×, 2.3×).
              SwinIR runs on overlapping patches which are stitched together using a Hann-window blend.
            </p>
          </div>
        )}
      </div>
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════════
   TAB 3: LR-Only Inference
   ═══════════════════════════════════════════════════════════════ */
function LROnlyTab({ tasks, optionsFiles, jobId, setJobId }) {
  const [modelSource, setModelSource] = useState('auto')
  const [selectedTask, setSelectedTask] = useState('')
  const [customModelPath, setCustomModelPath] = useState('')
  const [modelConfig, setModelConfig] = useState(DEFAULT_MODEL_CONFIG)
  const latestInfo = useLatestModel(modelSource, selectedTask, setModelConfig)
  const { selectedOptions, configSource, handleOptionsSelect } = useConfigAutoLoad(modelSource, customModelPath, setModelConfig)

  const [config, setConfig] = useState({
    lr_path: '', lr_bands: [1, 2, 3],
    output_dir: 'testsets/raw_inference_output/lr_only',
    patch_size: 256, overlap: 0, scale_factor: 2,
  })
  const [jobDone, setJobDone] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [summary, setSummary] = useState(null)

  const setMC = (path, value) => setModelConfig(prev => deepSet(prev, path, value))
  const setCfg = (key, value) => setConfig(prev => ({ ...prev, [key]: value }))
  const lrMeta = useImageMeta(config.lr_path)
  const jobIdRef = useRef(null)


  useEffect(() => {
    if (!lrMeta?.width || !lrMeta?.height) { setSummary(null); return }
    const t = setTimeout(() => {
      getInferenceSummary({
        width: lrMeta.width, height: lrMeta.height,
        patch_size: config.patch_size, overlap: config.overlap,
        scale: config.scale_factor,
      }).then(r => setSummary(r.data)).catch(() => setSummary(null))
    }, 300)
    return () => clearTimeout(t)
  }, [lrMeta?.width, lrMeta?.height, config.patch_size, config.overlap, config.scale_factor])

  useEffect(() => {
    if (modelConfig?.upscale && modelConfig.upscale !== config.scale_factor) {
      setCfg('scale_factor', modelConfig.upscale)
    }
  }, [modelConfig?.upscale])

  const [cancelled, setCancelled] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault(); setError(''); setLoading(true)
    setJobId(null); setJobDone(false)
    try {
      const modelPath = modelSource === 'auto' ? (latestInfo?.model_path || '') : customModelPath
      const payload = { ...config, model_path: modelPath, model_network_config: modelConfig }
      const r = await startLROnlyInference(payload)
      // openLogStream('inference', r.data.job_id, () => { }, () => setJobDone(true))
      jobIdRef.current = r.data.job_id
      setJobId(r.data.job_id)
      setCancelled(false)
    } catch (err) {
      setError(err.response?.data?.detail || String(err))
    } finally { setLoading(false) }
  }

  const resultImages = jobDone && jobId ? [
    { label: 'LR Input', url: getRawResultImageUrl(jobId, 'lr_display.png') },
    { label: 'SR Output', url: getRawResultImageUrl(jobId, 'sr_display.png') },
  ] : []

  return (
    <div className="module-grid rise" style={{ animationDelay: '100ms' }}>
      <div className="col">
        <form onSubmit={handleSubmit}>
          <ModelSelectionCard
            modelSource={modelSource} setModelSource={setModelSource}
            tasks={tasks} selectedTask={selectedTask} setSelectedTask={setSelectedTask}
            latestInfo={latestInfo} customModelPath={customModelPath} setCustomModelPath={setCustomModelPath}
            optionsFiles={optionsFiles} selectedOptions={selectedOptions}
            onOptionsSelect={handleOptionsSelect} configSource={configSource}
            modelConfig={modelConfig}
          />
          <div className="card" style={{ marginBottom: 16 }}>
            <div className="card-title">Input</div>
            <PathField label="LR image" mode="files" extensions=".tif,.tiff,.jp2,.img,.png,.jpg,.jpeg,.bmp"
              value={config.lr_path} onChange={v => setCfg('lr_path', v)} placeholder="/path/to/image.tif" mono />
            {lrMeta && <ImageInfoCard meta={lrMeta} />}
            {lrMeta ? (
              <BandCheckboxSelector totalBands={lrMeta.bands} selectedBands={config.lr_bands}
                onChange={v => setCfg('lr_bands', v)} label="LR display bands"
                descriptions={lrMeta.band_descriptions} />
            ) : (
              <ArrayEditor label="LR RGB bands" value={config.lr_bands} onChange={v => setCfg('lr_bands', v)} />
            )}
          </div>

          <div className="card" style={{ marginBottom: 16 }}>
            <div className="card-title">Inference</div>
            <div className="grid-2">
              <div className="form-group">
                <label>Scale <span className="hint">from model config</span></label>
                <input className="num-input" value={`×${config.scale_factor}`} readOnly
                  style={{ background: 'var(--bg-2)', color: 'var(--ink-2)' }} />
              </div>
              <NumberField label="Overlap (LR px)" value={config.overlap}
                onChange={v => setCfg('overlap', v)} min={0} step={8} />
            </div>
            <NumberField label="Patch size (LR px)" value={config.patch_size}
              onChange={v => setCfg('patch_size', v)} min={32} step={8} />
            <PathField label="Output directory" mode="dirs"
              value={config.output_dir} onChange={v => setCfg('output_dir', v)}
              placeholder="testsets/raw_inference_output/lr_only" mono />
            <InferenceSummaryCard summary={summary} />
          </div>

          {error && <div style={{ color: 'var(--bad)', fontSize: 13, marginBottom: 12 }}>{error}</div>}
          <button type="submit" className="btn btn-primary full-width"
            disabled={loading || !config.lr_path || (modelSource === 'auto' && !selectedTask)}>
            {loading ? 'Starting…' : '▶ Run Super-Resolution'}
          </button>

          <InlineJobPanel
            jobId={jobId}
            running={!!jobId && !jobDone && !cancelled}
            cancelled={cancelled}
            onStop={async () => {
              setCancelled(true)
              try { await stopInference(jobId) } catch { /* ignore */ }
            }}
            progressFetcher={getInferenceProgress}    // ← ADD
            outputDir={config.output_dir}
            scale={config.scale_factor}
            patchSize={config.patch_size}
            overlap={config.overlap}
            inputDims={lrMeta ? { width: lrMeta.width, height: lrMeta.height } : null}
            estimatedPatches={summary?.total_patches ?? null}
          />
        </form>
      </div>

      <div className="col">
        <ModelArchCard modelConfig={modelConfig} setMC={setMC} configSource={configSource} />
        {jobId && (
          <CollapsibleSection
            title={
              <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                Live Output
                <LiveOutputBadge
                  status={
                    cancelled   ? 'cancelled'
                    : jobDone   ? 'completed'
                    : jobId     ? 'running'
                    :           'completed'
                  }
                />
              </span>
            }
            defaultOpen={false}
          >
            <LogConsole
              domain="inference"
              jobId={jobId}
              onStop={async () => {
                setCancelled(true)                        // ← flips badge immediately
                try { await stopInference(jobId) } catch { }
              }}
              onComplete={() => {
                if (jobIdRef.current !== jobId) return
                setJobDone(true)
              }}
              showControls={{ copy: true, pause: false, 
                      cancel: false, elapsed: false }}
            />
          </CollapsibleSection>
        )}
        {jobDone && resultImages.length > 0 && (
          <div className="card" style={{ marginTop: 16 }}>
            <div className="card-title">Results</div>
            <ImageViewer images={resultImages} />
            {jobId && (
              <ImageCompareSlider
                lrUrl={getRawResultImageUrl(jobId, 'lr_display.png')}
                srUrl={getRawResultImageUrl(jobId, 'sr_display.png')}
                title="LR ↔ SR Comparison" />
            )}
            <BandImageViewer jobId={jobId} nBands={config.lr_bands.length}
              lrBands={config.lr_bands} paired={false} />
          </div>
        )}
        {!jobId && (
          <div className="card">
            <div className="card-title">About</div>
            <p className="text-muted text-sm" style={{ lineHeight: 1.6 }}>
              Runs SwinIR on a single LR image with no HR ground truth. No metrics are
              computed — only the LR input and SR output are saved and displayed.
              Useful for real-world inference on unlabelled satellite imagery.
            </p>
            <p className="text-muted text-sm" style={{ lineHeight: 1.6, marginTop: 10 }}>
              Supports GeoTIFF, JP2, PNG, and other standard image formats.
              Band selection applies only to multi-band geospatial files.
            </p>
          </div>
        )}
      </div>
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════════
   Main page
   ═══════════════════════════════════════════════════════════════ */
export default function Inference() {
  const [activeTab, setActiveTab] = useState(0)
  const [tasks, setTasks] = useState([])
  const [optionsFiles, setOptionsFiles] = useState([])
  const { jobs, setJobId: setCtxJobId } = useJobContext()

  useEffect(() => {
    listInferenceTasks().then(r => setTasks(r.data)).catch(() => { })
    listTrainingConfigs().then(r => setOptionsFiles(r.data)).catch(() => { })
  }, [])

  const tabs = [
    { label: 'Patched Images', icon: '▦' },
    { label: 'Raw HR+LR Inference', icon: '⟳' },
    { label: 'LR-Only Inference', icon: '↑' },
  ]

  const makeJobProps = (key) => ({
    jobId: jobs[key],
    setJobId: (id) => setCtxJobId(key, id),
  })

  return (
    <div>
      <div className="topbar">
        <div className="topbar-title"><h2>Inference</h2></div>
      </div>

      <div className="content">
        <h1 className="editorial rise" style={{ fontSize: 32, marginBottom: 10 }}>Run Inference and Metrics</h1>
        <p className="rise" style={{ color: 'var(--ink-2)', marginBottom: 24, maxWidth: 680 }}>
          Run a trained SwinIR model on LR images and compute super-resolution quality metrics.
          Choose a mode below — patched directory inference, raw paired satellite image inference,
          or LR-only inference without ground truth.
        </p>

        <div className="mode-tabs rise" style={{ marginBottom: 28, animationDelay: '80ms' }}>
          {tabs.map((t, i) => (
            <button key={i} type="button"
              className={`mode-tab ${activeTab === i ? 'active' : ''}`}
              onClick={() => setActiveTab(i)}>
              <span style={{ marginRight: 6 }}>{t.icon}</span>{t.label}
            </button>
          ))}
        </div>

        <div style={{ display: activeTab === 0 ? 'block' : 'none' }}>
          <PatchedTab tasks={tasks} optionsFiles={optionsFiles} {...makeJobProps('inference-patched')} />
        </div>
        <div style={{ display: activeTab === 1 ? 'block' : 'none' }}>
          <RawPairedTab tasks={tasks} optionsFiles={optionsFiles} {...makeJobProps('inference-raw')} />
        </div>
        <div style={{ display: activeTab === 2 ? 'block' : 'none' }}>
          <LROnlyTab tasks={tasks} optionsFiles={optionsFiles} {...makeJobProps('inference-lr')} />
        </div>
      </div>
    </div>
  )
}