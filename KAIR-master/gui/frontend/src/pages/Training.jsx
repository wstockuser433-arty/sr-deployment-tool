import { useEffect, useState, useRef } from 'react'
import {
  listTrainingConfigs, getTrainingConfig, listTrainingRuns,
  startTraining, stopTraining, pauseTraining, resumeTraining
} from '../api/client'
import { useJobContext } from '../context/JobContext'
import LogConsole from '../components/LogConsole'
import {
  SelectField, TextField, NumberField, BoolToggle,
  ArrayEditor, CollapsibleSection, PathField
} from '../components/FormFields'

const SCALE_OPTIONS = [
  { value: 2, label: '×2' }, { value: 3, label: '×3' },
  { value: 4, label: '×4' }, { value: 8, label: '×8' },
]
const LOSS_TYPES = ['l1', 'l2', 'l2sum', 'ssim', 'charbonnier']
const UPSAMPLER_OPTIONS = ['pixelshuffle', 'pixelshuffledirect', 'nearest+conv']
const RESI_OPTIONS = ['1conv', '3conv']
const GAN_TYPES = ['gan', 'ragan', 'lsgan', 'wgan', 'softplusgan']
const DEG_TYPES = ['bsrgan', 'bsrgan_plus', 'real_esrgan', 'satellite']
const BAND_PRESETS = [
  { key: 'pan',    label: 'Panchromatic — 1 band',                  n: 1,  bands: ['Pan'] },
  { key: 'rgb',    label: 'RGB — 3 bands',                           n: 3,  bands: ['R', 'G', 'B'] },
  { key: 'ms4',    label: 'Multispectral 4-band (Pleiades / Planet)', n: 4,  bands: ['B', 'G', 'R', 'NIR'] },
  { key: 'ms8',    label: 'Multispectral 8-band (WorldView)',         n: 8,  bands: ['C', 'B', 'G', 'Y', 'R', 'RE', 'NIR1', 'NIR2'] },
  { key: 's2',     label: 'Sentinel-2 — 13 bands',                   n: 13, bands: ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12'] },
  { key: 'custom', label: 'Custom…',                                 n: null, bands: [] },
]

const DEFAULT_PSNR = {
  training_type: 'psnr',
  task: 'swinir_sr_x2_psnr',
  scale: 2, n_channels: 3, gpu_ids: [0],
  pretrained_netG: null, pretrained_netE: null,
  train_dataset: {
    dataroot_H: 'trainsets/trainH', dataroot_L: null,
    H_size: 256, dataloader_shuffle: true,
    dataloader_num_workers: 4, dataloader_batch_size: 2,
    dataset_type: 'sr', use_photometric_aug: false,
    degradation_type: null, shuffle_prob: null, lq_patchsize: null, use_sharp: null,
  },
  test_dataset: {
    dataroot_H: 'testsets/Set5/HR', dataroot_L: 'testsets/Set5/LR_bicubic/X2',
    dataset_type: 'sr',
  },
  net_g: {
    net_type: 'swinir', upscale: 2, in_chans: 3, img_size: 128,
    window_size: 8, img_range: 1.0, depths: [6, 6, 6, 6, 6, 6],
    embed_dim: 180, num_heads: [6, 6, 6, 6, 6, 6],
    mlp_ratio: 2, upsampler: 'pixelshuffle', resi_connection: '1conv', init_type: 'default',
  },
  train_params: {
    G_lossfn_type: 'l1', G_lossfn_weight: 1.0, E_decay: 0.999,
    G_optimizer_type: 'adam', G_optimizer_lr: 1e-4, G_optimizer_wd: 0,
    G_optimizer_reuse: true, G_scheduler_type: 'MultiStepLR',
    G_scheduler_milestones: [300000, 400000, 500000, 600000, 700000],
    G_scheduler_gamma: 0.5, G_param_strict: true, E_param_strict: true,
    checkpoint_test: 10000, checkpoint_save: 10000, checkpoint_print: 1000,
  },
  gan_extras: null,
}

const DEFAULT_GAN = {
  ...DEFAULT_PSNR,
  training_type: 'gan',
  task: 'swinir_sr_x2_gan',
  train_dataset: {
    ...DEFAULT_PSNR.train_dataset,
    dataset_type: 'sr',
    degradation_type: 'bsrgan',
    shuffle_prob: 0.1, lq_patchsize: 128, use_sharp: true,
  },
  train_params: {
    ...DEFAULT_PSNR.train_params,
    G_optimizer_lr: 5e-5,
    G_scheduler_milestones: [100000, 200000, 300000, 400000, 500000],
    checkpoint_test: 5000, checkpoint_save: 5000,
  },
  gan_extras: {
    F_lossfn_type: 'l1', F_lossfn_weight: 1.0,
    F_feature_layer: [2, 7, 16, 25, 34], F_weights: [0.1, 0.1, 1.0, 1.0, 1.0],
    F_use_input_norm: true, F_use_range_norm: false,
    gan_type: 'gan', D_lossfn_weight: 0.1,
    D_optimizer_lr: 5e-5, D_optimizer_wd: 0,
    D_scheduler_milestones: [100000, 200000, 300000, 400000, 500000],
    D_scheduler_gamma: 0.5, D_init_iters: 0, G_optimizer_lr: 5e-5,
  },
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

function formatTrainingElapsed(sec) {
  const h = Math.floor(sec / 3600)
  const m = Math.floor((sec % 3600) / 60)
  const s = sec % 60
  if (h > 0) return `${h}h ${m}m ${s}s`
  if (m > 0) return `${m}m ${s}s`
  return `${s}s`
}

function parseTrainingMetrics(lines) {
  let psnr = null, ssim = null, loss = null, finalIter = null
  for (const line of lines) {
    // <epoch:  1, iter:    1000, lr:1.000e-04>
    const iterM = line.match(/<epoch:\s*\d+,\s*iter:\s*([\d,]+)/)
    if (iterM) finalIter = parseInt(iterM[1].replace(/,/g, ''))
    // Average PSNR: 33.60dB  (keep last occurrence = most recent checkpoint)
    const psnrM = line.match(/Average PSNR:\s*([0-9.]+)/i)
    if (psnrM) psnr = parseFloat(psnrM[1])
    // SSIM: 0.9265
    const ssimM = line.match(/\bSSIM:\s*([0-9.]+)/i)
    if (ssimM) ssim = parseFloat(ssimM[1])
    // G_loss: 1.234e-03  (scientific notation)
    const lossM = line.match(/G_loss:\s*([0-9.e+\-]+)/i)
    if (lossM) loss = parseFloat(lossM[1])
  }
  return { psnr, ssim, loss, finalIter }
}

export default function Training() {
  const [mode, setMode] = useState('psnr')
  const [form, setForm] = useState(DEFAULT_PSNR)
  const [configs, setConfigs] = useState([])
  const [runs, setRuns] = useState([])
  const { jobs, setJobId: setCtxJobId } = useJobContext()
  const jobId = jobs['training']
  const setJobId = (id) => setCtxJobId('training', id)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [bandPreset, setBandPreset] = useState('rgb')
  const [customBandCount, setCustomBandCount] = useState(3)
  const [trainingMetrics, setTrainingMetrics] = useState(null)
  const [trainingInfo, setTrainingInfo] = useState(null)
  const [jobDone, setJobDone] = useState(false)
  const allLinesRef = useRef([])
  const startTimeRef = useRef(null)

  const set = (path, value) => setForm((prev) => deepSet(prev, path, value))

  const applyBands = (presetKey, customN = customBandCount) => {
    const preset = BAND_PRESETS.find((p) => p.key === presetKey)
    const n = preset.n ?? customN
    setBandPreset(presetKey)
    if (preset.n) setCustomBandCount(preset.n)
    setForm((prev) => deepSet(deepSet(prev, 'n_channels', n), 'net_g.in_chans', n))
  }

  useEffect(() => {
    listTrainingConfigs().then((r) => setConfigs(r.data)).catch(() => { })
    listTrainingRuns().then((r) => setRuns(r.data)).catch(() => { })
  }, [jobId])

  useEffect(() => {
    setForm(mode === 'gan' ? DEFAULT_GAN : DEFAULT_PSNR)
    setBandPreset('rgb')
    setCustomBandCount(3)
  }, [mode])

  const loadConfig = async (name) => {
    try {
      const r = await getTrainingConfig(name)
      const cfg = r.data
      const isGan = cfg.model === 'gan'
      setMode(isGan ? 'gan' : 'psnr')
    } catch { }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    setJobDone(false)
    setTrainingMetrics(null)
    setTrainingInfo({ task: form.task, mode, scale: form.scale, n_channels: form.n_channels })
    allLinesRef.current = []
    startTimeRef.current = Date.now()
    try {
      const payload = { ...form, training_type: mode }
      const r = await startTraining(payload)
      setJobId(r.data.job_id)
    } catch (err) {
      setError(err.response?.data?.detail || String(err))
    } finally {
      setLoading(false)
    }
  }

  const handleStop = async () => {
    if (jobId) await stopTraining(jobId).catch(() => { })
  }

  const handlePause = async () => {
    if (jobId) await pauseTraining(jobId).catch(() => { })
  }

  const handleResume = async () => {
    if (jobId) await resumeTraining(jobId).catch(() => { })
  }

  const handleLogLine = (line) => { allLinesRef.current.push(line) }

  const handleTrainingComplete = () => {
    const elapsed = startTimeRef.current ? Math.floor((Date.now() - startTimeRef.current) / 1000) : null
    const parsed = parseTrainingMetrics(allLinesRef.current)
    setTrainingMetrics({ ...parsed, elapsed })
    setJobDone(true)
    // Refresh runs list so the completed run shows up immediately
    listTrainingRuns().then((r) => setRuns(r.data)).catch(() => { })
  }

  return (
    <div>
      <div className="topbar">
        <div className="topbar-title">
          <h2>Training</h2>
        </div>
      </div>

      <div className="content">
        <h1 className="editorial rise" style={{ fontSize: 32, marginBottom: 10 }}>Configure and launch SwinIR model training</h1>
        <p className="rise" style={{ color: 'var(--ink-2)', marginBottom: 30, maxWidth: 600 }}>
          Set up hyperparameters, dataset paths, and architecture settings for PSNR or GAN-based super-resolution training.
        </p>

        <div className="module-grid rise" style={{ animationDelay: '100ms' }}>
          {/* ─── Left: Config form ─────────────────────────── */}
          <div className="col">
            {/* Mode selector */}
            <div className="mode-tabs">
              <button className={`mode-tab ${mode === 'psnr' ? 'active' : ''}`} onClick={() => setMode('psnr')}>
                PSNR Training
              </button>
              <button className={`mode-tab ${mode === 'gan' ? 'active' : ''}`} onClick={() => setMode('gan')}>
                GAN Training
              </button>
            </div>

            {/* Load preset config */}
            {configs.length > 0 && (
              <div className="form-group">
                <label>Load preset config <span className="hint">(optional — autofills scale, dataset type)</span></label>
                <select className="text-input" onChange={(e) => e.target.value && loadConfig(e.target.value)} defaultValue="">
                  <option value="">— choose a config file —</option>
                  {configs.map((c) => (
                    <option key={c.name} value={c.name}>{c.name}</option>
                  ))}
                </select>
              </div>
            )}

            <form onSubmit={handleSubmit}>
              {/* ── Basic settings ── */}
              <CollapsibleSection title="Basic Settings" defaultOpen>
                <TextField label="Task name" hint="used as output directory name"
                  value={form.task} onChange={(v) => set('task', v)} />
                <div className="grid-2">
                  <SelectField label="Scale" value={form.scale}
                    onChange={(v) => set('scale', parseInt(v))} options={SCALE_OPTIONS} />
                  <div className="form-group">
                    <label>Input bands</label>
                    <select className="text-input" value={bandPreset}
                      onChange={(e) => applyBands(e.target.value)}>
                      {BAND_PRESETS.map((p) => (
                        <option key={p.key} value={p.key}>{p.label}</option>
                      ))}
                    </select>
                    {bandPreset === 'custom' ? (
                      <div style={{ marginTop: 6, display: 'flex', alignItems: 'center', gap: 8 }}>
                        <input type="number" className="num-input" style={{ width: 80 }}
                          value={customBandCount} min={1} max={200}
                          onChange={(e) => {
                            const n = Math.max(1, parseInt(e.target.value) || 1)
                            setCustomBandCount(n)
                            applyBands('custom', n)
                          }} />
                        <span style={{ fontSize: 12, color: 'var(--ink-2)' }}>bands</span>
                      </div>
                    ) : (
                      <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap', marginTop: 6 }}>
                        {BAND_PRESETS.find((p) => p.key === bandPreset)?.bands.map((b) => (
                          <span key={b} style={{
                            padding: '2px 7px', borderRadius: 10, fontSize: 11, fontWeight: 500,
                            background: 'var(--bg-2)', border: '1px solid var(--line-2)', color: 'var(--ink-2)',
                          }}>{b}</span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </CollapsibleSection>

              {/* ── Dataset ── */}
              <CollapsibleSection title="Dataset" defaultOpen>
                <PathField label="HR train dir" mode="dirs" value={form.train_dataset.dataroot_H}
                  onChange={(v) => set('train_dataset.dataroot_H', v)} placeholder="trainsets/trainH" />
                <PathField label="LR train dir" mode="dirs" hint="leave blank to use bicubic downsampling"
                  value={form.train_dataset.dataroot_L || ''}
                  onChange={(v) => set('train_dataset.dataroot_L', v || null)} placeholder="trainsets/trainL (optional)" />
                <PathField label="HR test dir" mode="dirs" value={form.test_dataset.dataroot_H}
                  onChange={(v) => set('test_dataset.dataroot_H', v)} />
                <PathField label="LR test dir" mode="dirs" value={form.test_dataset.dataroot_L || ''}
                  onChange={(v) => set('test_dataset.dataroot_L', v || null)} placeholder="(optional)" />
                <div className="grid-3">
                  <NumberField label="HR patch size" value={form.train_dataset.H_size}
                    onChange={(v) => set('train_dataset.H_size', v)} min={64} step={32} />
                  <NumberField label="Batch size" value={form.train_dataset.dataloader_batch_size}
                    onChange={(v) => set('train_dataset.dataloader_batch_size', v)} min={1} />
                  <NumberField label="Workers" value={form.train_dataset.dataloader_num_workers}
                    onChange={(v) => set('train_dataset.dataloader_num_workers', v)} min={0} />
                </div>
                {mode === 'gan' && (
                  <>
                    <SelectField label="Degradation type" value={form.train_dataset.degradation_type || 'bsrgan'}
                      onChange={(v) => set('train_dataset.degradation_type', v)} options={DEG_TYPES} />
                    <div className="grid-2">
                      <NumberField label="LQ patch size" value={form.train_dataset.lq_patchsize || 128}
                        onChange={(v) => set('train_dataset.lq_patchsize', v)} min={32} step={16} />
                      <NumberField label="Shuffle prob" value={form.train_dataset.shuffle_prob || 0.1}
                        onChange={(v) => set('train_dataset.shuffle_prob', v)} min={0} max={1} step={0.05} />
                    </div>
                    <BoolToggle label="Use sharp (USM sharpening on HR)"
                      value={!!form.train_dataset.use_sharp}
                      onChange={(v) => set('train_dataset.use_sharp', v)} />
                  </>
                )}
              </CollapsibleSection>

              {/* ── Pre-trained model ── */}
              <CollapsibleSection title="Pre-trained Model" defaultOpen={false}>
                <PathField label="Pretrained netG path" mode="files" extensions=".pth,.pt"
                  hint="null = train from scratch"
                  value={form.pretrained_netG || ''}
                  onChange={(v) => set('pretrained_netG', v || null)} placeholder="model_zoo/..." />
                {mode === 'gan' && (
                  <>
                    <PathField label="Pretrained netD path" mode="files" extensions=".pth,.pt"
                      value={form.pretrained_netD || ''}
                      onChange={(v) => set('pretrained_netD', v || null)} placeholder="(optional)" />
                    <PathField label="Pretrained netE path" mode="files" extensions=".pth,.pt"
                      value={form.pretrained_netE_gan || ''}
                      onChange={(v) => set('pretrained_netE_gan', v || null)} placeholder="(optional)" />
                  </>
                )}
              </CollapsibleSection>

              {/* ── Network architecture ── */}
              <CollapsibleSection title="Network Architecture" defaultOpen={false}>
                <div className="grid-2">
                  <NumberField label="Embed dim" hint="divisible by all num_heads"
                    value={form.net_g.embed_dim} onChange={(v) => set('net_g.embed_dim', v)} min={60} step={12} />
                  <NumberField label="MLP ratio" value={form.net_g.mlp_ratio}
                    onChange={(v) => set('net_g.mlp_ratio', v)} min={1} />
                </div>
                <div className="grid-2">
                  <SelectField label="Upsampler" value={form.net_g.upsampler}
                    onChange={(v) => set('net_g.upsampler', v)} options={UPSAMPLER_OPTIONS} />
                  <SelectField label="Resi connection" value={form.net_g.resi_connection}
                    onChange={(v) => set('net_g.resi_connection', v)} options={RESI_OPTIONS} />
                </div>
                <ArrayEditor label="Depths (transformer blocks per stage)"
                  value={form.net_g.depths} onChange={(v) => set('net_g.depths', v)} />
                <ArrayEditor label="Num heads (must match depths length)"
                  value={form.net_g.num_heads} onChange={(v) => set('net_g.num_heads', v)} />
              </CollapsibleSection>

              {/* ── Training hyperparams ── */}
              <CollapsibleSection title="Training Hyperparameters" defaultOpen={false}>
                <div className="grid-2">
                  <SelectField label="Loss function" value={form.train_params.G_lossfn_type}
                    onChange={(v) => set('train_params.G_lossfn_type', v)} options={LOSS_TYPES} />
                  <NumberField label="Learning rate (G)" value={form.train_params.G_optimizer_lr}
                    onChange={(v) => set('train_params.G_optimizer_lr', v)} step={1e-5} />
                </div>
                <div className="grid-3">
                  <NumberField label="EMA decay" value={form.train_params.E_decay}
                    onChange={(v) => set('train_params.E_decay', v)} min={0} max={1} step={0.001} />
                  <NumberField label="LR gamma" value={form.train_params.G_scheduler_gamma}
                    onChange={(v) => set('train_params.G_scheduler_gamma', v)} min={0} max={1} step={0.1} />
                  <NumberField label="Ckpt print" value={form.train_params.checkpoint_print}
                    onChange={(v) => set('train_params.checkpoint_print', v)} min={100} step={100} />
                </div>
                <div className="grid-2">
                  <NumberField label="Ckpt save" value={form.train_params.checkpoint_save}
                    onChange={(v) => set('train_params.checkpoint_save', v)} min={1000} step={1000} />
                  <NumberField label="Ckpt test" value={form.train_params.checkpoint_test}
                    onChange={(v) => set('train_params.checkpoint_test', v)} min={1000} step={1000} />
                </div>
                <ArrayEditor label="LR milestones (iterations)"
                  value={form.train_params.G_scheduler_milestones}
                  onChange={(v) => set('train_params.G_scheduler_milestones', v)} />
              </CollapsibleSection>

              {/* ── GAN-only extras ── */}
              {mode === 'gan' && form.gan_extras && (
                <CollapsibleSection title="GAN / Discriminator Settings" defaultOpen={false}>
                  <div className="grid-2">
                    <SelectField label="GAN type" value={form.gan_extras.gan_type}
                      onChange={(v) => set('gan_extras.gan_type', v)} options={GAN_TYPES} />
                    <SelectField label="Perceptual loss" value={form.gan_extras.F_lossfn_type}
                      onChange={(v) => set('gan_extras.F_lossfn_type', v)} options={['l1', 'l2']} />
                  </div>
                  <div className="grid-2">
                    <NumberField label="D loss weight" value={form.gan_extras.D_lossfn_weight}
                      onChange={(v) => set('gan_extras.D_lossfn_weight', v)} step={0.01} />
                    <NumberField label="D LR" value={form.gan_extras.D_optimizer_lr}
                      onChange={(v) => set('gan_extras.D_optimizer_lr', v)} step={1e-6} />
                  </div>
                  <ArrayEditor label="D LR milestones"
                    value={form.gan_extras.D_scheduler_milestones}
                    onChange={(v) => set('gan_extras.D_scheduler_milestones', v)} />
                </CollapsibleSection>
              )}

              {error && (
                <div style={{ color: 'var(--bad)', fontSize: 13, marginBottom: 12 }}>{error}</div>
              )}

              <button type="submit" className="btn btn-primary full-width" disabled={loading} style={{ marginTop: 8 }}>
                {loading ? 'Starting…' : `▶ Start ${mode.toUpperCase()} Training`}
              </button>
            </form>
          </div>

          {/* ─── Right: Recent runs + log ─────────────────── */}
          <div className="col">
            <div className="card">
              <div className="card-title">Training Run History</div>
              {runs.length === 0 ? (
                <p className="text-muted text-sm">No training runs found in superresolution/</p>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {runs.map((r) => (
                    <div key={r.task_name} className="ds-row" style={{ padding: '10px 14px' }}>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 7, flexWrap: 'wrap', marginBottom: 4 }}>
                          <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--ink)' }}>{r.task_name}</span>
                          <span style={{
                            fontSize: 10, fontWeight: 600, padding: '1px 6px', borderRadius: 4,
                            background: r.config_type === 'gan' ? 'rgba(99,102,241,0.12)' : 'var(--surface-2)',
                            color: r.config_type === 'gan' ? 'var(--cobalt-deep)' : 'var(--ink-2)',
                            border: '1px solid', borderColor: r.config_type === 'gan' ? 'rgba(99,102,241,0.3)' : 'var(--line-2)',
                            textTransform: 'uppercase',
                          }}>
                            {r.config_type === 'plain' ? 'PSNR' : (r.config_type ?? 'unknown').toUpperCase()}
                          </span>
                          {r.scale != null && (
                            <span style={{
                              fontSize: 10, fontWeight: 600, padding: '1px 6px', borderRadius: 4,
                              background: 'var(--surface-2)', color: 'var(--ink-2)',
                              border: '1px solid var(--line-2)',
                            }}>×{r.scale}</span>
                          )}
                          {r.n_channels != null && (
                            <span style={{
                              fontSize: 10, padding: '1px 6px', borderRadius: 4,
                              background: 'var(--surface-2)', color: 'var(--ink-3)',
                              border: '1px solid var(--line-2)',
                            }}>{r.n_channels}ch</span>
                          )}
                        </div>
                        <div style={{ fontSize: 11, color: 'var(--ink-3)', fontFamily: 'var(--font-mono)', display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                          <span>{r.latest_iteration != null ? `iter ${r.latest_iteration.toLocaleString()}` : 'no checkpoint'}</span>
                          {r.best_psnr != null && (
                            <span style={{ color: 'var(--ok)', fontWeight: 600 }}>PSNR {r.best_psnr.toFixed(2)} dB</span>
                          )}
                          {r.best_ssim != null && (
                            <span style={{ color: 'var(--ok)' }}>SSIM {r.best_ssim.toFixed(4)}</span>
                          )}
                        </div>
                        {r.latest_model_path && (
                          <div style={{
                            fontSize: 10, color: 'var(--ink-3)', fontFamily: 'var(--font-mono)',
                            marginTop: 3, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                          }} title={r.latest_model_path}>
                            {r.latest_model_path}
                          </div>
                        )}
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 4, marginLeft: 10 }}>
                        {r.has_log && (
                          <span style={{ fontSize: 10, color: 'var(--ok)', fontWeight: 600 }}>LOG</span>
                        )}
                        {!r.latest_model_path && (
                          <span style={{ fontSize: 10, color: 'var(--ink-3)' }}>no model</span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {jobId && (
              <LogConsole
                domain="training"
                jobId={jobId}
                onStop={handleStop}
                onPause={handlePause}
                onResume={handleResume}
                onLine={handleLogLine}
                onComplete={handleTrainingComplete}
              />
            )}
            {jobDone && (
              <div className="card" style={{ marginTop: 16 }}>
                <div className="card-title">Training Summary</div>

                {/* Task / type / scale header row */}
                {trainingInfo && (
                  <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 12, marginTop: 4 }}>
                    <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--ink)' }}>{trainingInfo.task}</span>
                    <span style={{
                      fontSize: 10, fontWeight: 600, padding: '2px 7px', borderRadius: 4,
                      background: trainingInfo.mode === 'gan' ? 'rgba(99,102,241,0.12)' : 'var(--surface-2)',
                      color: trainingInfo.mode === 'gan' ? 'var(--cobalt-deep)' : 'var(--ink-2)',
                      border: '1px solid', borderColor: trainingInfo.mode === 'gan' ? 'rgba(99,102,241,0.3)' : 'var(--line-2)',
                      textTransform: 'uppercase', alignSelf: 'center',
                    }}>{trainingInfo.mode}</span>
                    {trainingInfo.scale && (
                      <span style={{
                        fontSize: 10, fontWeight: 600, padding: '2px 7px', borderRadius: 4,
                        background: 'var(--surface-2)', color: 'var(--ink-2)',
                        border: '1px solid var(--line-2)', alignSelf: 'center',
                      }}>×{trainingInfo.scale}</span>
                    )}
                    {trainingInfo.n_channels && (
                      <span style={{
                        fontSize: 10, padding: '2px 7px', borderRadius: 4,
                        background: 'var(--surface-2)', color: 'var(--ink-3)',
                        border: '1px solid var(--line-2)', alignSelf: 'center',
                      }}>{trainingInfo.n_channels}ch</span>
                    )}
                  </div>
                )}

                <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
                  {trainingMetrics?.finalIter != null && (
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13, padding: '8px 0', borderBottom: '1px solid var(--line-2)' }}>
                      <span style={{ color: 'var(--ink-2)' }}>Final iteration</span>
                      <span style={{ fontFamily: 'monospace', fontWeight: 600 }}>{trainingMetrics.finalIter.toLocaleString()}</span>
                    </div>
                  )}
                  {trainingMetrics?.psnr != null && (
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13, padding: '8px 0', borderBottom: '1px solid var(--line-2)' }}>
                      <span style={{ color: 'var(--ink-2)' }}>PSNR (last checkpoint)</span>
                      <span style={{ fontFamily: 'monospace', fontWeight: 600, color: 'var(--ok)' }}>{trainingMetrics.psnr.toFixed(2)} dB</span>
                    </div>
                  )}
                  {trainingMetrics?.ssim != null && (
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13, padding: '8px 0', borderBottom: '1px solid var(--line-2)' }}>
                      <span style={{ color: 'var(--ink-2)' }}>SSIM (last checkpoint)</span>
                      <span style={{ fontFamily: 'monospace', fontWeight: 600, color: 'var(--ok)' }}>{trainingMetrics.ssim.toFixed(4)}</span>
                    </div>
                  )}
                  {trainingMetrics?.loss != null && (
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13, padding: '8px 0', borderBottom: '1px solid var(--line-2)' }}>
                      <span style={{ color: 'var(--ink-2)' }}>Final G_loss</span>
                      <span style={{ fontFamily: 'monospace', color: 'var(--ink-2)' }}>{trainingMetrics.loss.toExponential(3)}</span>
                    </div>
                  )}
                  {trainingMetrics?.elapsed != null && (
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13, padding: '8px 0', borderBottom: '1px solid var(--line-2)' }}>
                      <span style={{ color: 'var(--ink-2)' }}>Total time</span>
                      <span style={{ fontFamily: 'monospace', color: 'var(--ink-2)' }}>{formatTrainingElapsed(trainingMetrics.elapsed)}</span>
                    </div>
                  )}
                  {trainingInfo?.task && (
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, padding: '8px 0', gap: 12 }}>
                      <span style={{ color: 'var(--ink-3)', flexShrink: 0 }}>Model saved to</span>
                      <span style={{ fontFamily: 'monospace', color: 'var(--ink-3)', textAlign: 'right', wordBreak: 'break-all' }}>
                        superresolution/{trainingInfo.task}/models/
                      </span>
                    </div>
                  )}
                  {trainingMetrics?.psnr == null && trainingMetrics?.ssim == null && (
                    <div style={{ fontSize: 12, color: 'var(--ink-3)', paddingTop: 6 }}>
                      No PSNR / SSIM found in log — check test interval (checkpoint_test) or log output above.
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
