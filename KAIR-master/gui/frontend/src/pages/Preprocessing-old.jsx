import { useState, useEffect, useRef } from 'react'
import {
  startPipeline3, startRunPipeline, stopPreprocessing,
  pausePreprocessing, resumePreprocessing,
  detectPreprocessingStructure, listDirectory, fsImageUrl,
  getImageInfo, compareImages,
} from '../api/client'
import { useJobContext } from '../context/JobContext'
import LogConsole from '../components/LogConsole'
import {
  SelectField, TextField, NumberField, BoolToggle,
  ArrayEditor, CollapsibleSection, PathField
} from '../components/FormFields'

const PREVIEW_STAGES = [
  { key: 'load_hr',     label: 'HR Loaded' },
  { key: 'bands_hr',    label: 'HR Bands' },
  { key: 'load_lr',     label: 'LR Loaded' },
  { key: 'bands_lr',    label: 'LR Bands' },
  { key: 'coreg_a',     label: 'Coreg — ORB' },
  { key: 'coreg_b',     label: 'Coreg — Phase' },
  { key: 'radiometric', label: 'Radiometric' },
  { key: 'patches',     label: 'Sample Patches' },
]

function StepPreviewPanel({ previews }) {
  const [lightbox, setLightbox] = useState(null)
  const stages = PREVIEW_STAGES.filter(s => previews[s.key])

  if (stages.length === 0) {
    return (
      <div className="card" style={{ textAlign: 'center', padding: '60px 40px' }}>
        <div style={{ fontSize: 32, marginBottom: 16, opacity: 0.3 }}>⬡</div>
        <div style={{ fontSize: 13, color: 'var(--ink-3)', lineHeight: 1.7 }}>
          No step previews yet.<br />
          Run <strong>Pipeline A</strong> on a satellite image pair to see the intermediate
          output at each processing stage.
        </div>
      </div>
    )
  }

  return (
    <div>
      {lightbox && (
        <div onClick={() => setLightbox(null)} style={{
          position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.87)', zIndex: 9999,
          display: 'flex', flexDirection: 'column', alignItems: 'center',
          justifyContent: 'center', cursor: 'zoom-out',
        }}>
          <div style={{ color: '#fff', fontSize: 15, fontWeight: 600, marginBottom: 12 }}>
            {lightbox.label}
          </div>
          <img src={lightbox.url} alt={lightbox.label}
            style={{ maxWidth: '88vw', maxHeight: '80vh', borderRadius: 8, boxShadow: '0 0 60px rgba(0,0,0,0.8)' }} />
          <div style={{ color: 'rgba(255,255,255,0.45)', fontSize: 12, marginTop: 10 }}>
            Scene: {lightbox.scene} &nbsp;·&nbsp; click anywhere to close
          </div>
        </div>
      )}

      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))',
        gap: 20,
      }}>
        {stages.map(s => (
          <div key={s.key} className="card"
            style={{ cursor: 'pointer', padding: 0, overflow: 'hidden', transition: 'box-shadow 0.15s' }}
            onClick={() => setLightbox({ url: previews[s.key].url, scene: previews[s.key].scene, label: s.label })}>
            <img
              src={previews[s.key].url} alt={s.label}
              style={{ width: '100%', display: 'block', objectFit: 'cover', aspectRatio: '4/3', background: 'var(--bg-2)' }}
            />
            <div style={{ padding: '10px 14px' }}>
              <div style={{ fontWeight: 600, fontSize: 13, color: 'var(--ink)' }}>{s.label}</div>
              <div style={{ fontSize: 11, color: 'var(--ink-3)', marginTop: 3 }}>
                Scene: {previews[s.key].scene}
              </div>
              <div style={{ fontSize: 10, color: 'var(--cobalt-deep)', marginTop: 4 }}>Click to enlarge</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Per-class results panel (Pipeline B classed-directory output) ─────────────

function ImageSampleRow({ label, images, onLightbox }) {
  if (!images.length) return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--ink-2)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 6 }}>{label}</div>
      <div style={{ fontSize: 12, color: 'var(--ink-3)' }}>No images found in output directory.</div>
    </div>
  )
  return (
    <div style={{ marginBottom: 20 }}>
      <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--ink-2)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 8 }}>{label}</div>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        {images.map((img) => {
          const url = fsImageUrl(img.path)
          return (
            <div key={img.path}
              style={{ flex: '1 1 90px', minWidth: 80, maxWidth: 130, cursor: 'pointer' }}
              onClick={() => onLightbox({ url, name: img.name })}>
              <img src={url} alt={img.name}
                style={{ width: '100%', borderRadius: 'var(--radius-sm)', border: '1px solid var(--line-2)', objectFit: 'cover', aspectRatio: '1/1', background: 'var(--bg-2)' }}
                onError={e => { e.target.style.opacity = 0.3 }}
              />
              <div style={{ fontSize: 10, color: 'var(--ink-3)', marginTop: 3, textAlign: 'center', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{img.name}</div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function ClassResultsPanel({ results }) {
  const [selectedIdx, setSelectedIdx] = useState(0)
  const [hrImages, setHrImages] = useState([])
  const [lrImages, setLrImages] = useState([])
  const [loading, setLoading] = useState(false)
  const [lightbox, setLightbox] = useState(null)

  const cls = results[selectedIdx]
  const IMG_EXT = '.png,.jpg,.jpeg,.bmp'

  useEffect(() => {
    if (!cls) return
    setLoading(true)
    setHrImages([])
    setLrImages([])
    Promise.all([
      listDirectory(cls.hrDir, 'files', IMG_EXT).catch(() => ({ data: { entries: [] } })),
      listDirectory(cls.lrDir, 'files', IMG_EXT).catch(() => ({ data: { entries: [] } })),
    ]).then(([hr, lr]) => {
      setHrImages(hr.data.entries.slice(0, 8))
      setLrImages(lr.data.entries.slice(0, 8))
    }).finally(() => setLoading(false))
  }, [cls?.hrDir, cls?.lrDir])

  if (!results.length) return null

  const totalImages = results.reduce((s, r) => s + (r.count || 0), 0)

  return (
    <div style={{ marginBottom: 36 }}>
      {lightbox && (
        <div onClick={() => setLightbox(null)} style={{
          position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.87)', zIndex: 9999,
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', cursor: 'zoom-out',
        }}>
          <div style={{ color: '#fff', fontSize: 14, fontWeight: 600, marginBottom: 12 }}>{lightbox.name}</div>
          <img src={lightbox.url} alt={lightbox.name} style={{ maxWidth: '88vw', maxHeight: '82vh', borderRadius: 8 }} />
          <div style={{ color: 'rgba(255,255,255,0.4)', fontSize: 11, marginTop: 8 }}>click to close</div>
        </div>
      )}

      <div style={{ fontWeight: 600, fontSize: 16, marginBottom: 4 }}>Per-Class Results</div>
      <div style={{ fontSize: 12, color: 'var(--ink-3)', marginBottom: 18 }}>
        {results.length} classes processed · {totalImages.toLocaleString()} images total
      </div>

      {/* Class selector */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 20 }}>
        {results.map((r, i) => (
          <button key={r.name} type="button"
            onClick={() => setSelectedIdx(i)}
            style={{
              padding: '4px 11px', fontSize: 12, borderRadius: 'var(--radius-sm)', cursor: 'pointer',
              border: `1px solid ${i === selectedIdx ? 'var(--cobalt-deep)' : 'var(--line-2)'}`,
              background: i === selectedIdx ? 'var(--cobalt-soft)' : 'var(--surface)',
              color: i === selectedIdx ? 'var(--cobalt-deep)' : 'var(--ink-2)',
              fontWeight: i === selectedIdx ? 600 : 400,
            }}>
            {r.name}
            <span style={{ marginLeft: 5, fontSize: 10, opacity: 0.65 }}>{(r.count || 0).toLocaleString()}</span>
          </button>
        ))}
      </div>

      {/* Selected class sample images */}
      {cls && (
        <div className="card">
          <div className="card-title" style={{ marginBottom: 16 }}>
            {cls.name}
            <span style={{ marginLeft: 8, fontSize: 12, fontWeight: 400, color: 'var(--ink-3)' }}>
              {(cls.count || 0).toLocaleString()} images processed
            </span>
          </div>
          {loading ? (
            <div style={{ color: 'var(--ink-3)', fontSize: 12, padding: '12px 0' }}>Loading samples…</div>
          ) : (
            <>
              <ImageSampleRow label="HR Output" images={hrImages} onLightbox={setLightbox} />
              <ImageSampleRow label="LR Output" images={lrImages} onLightbox={setLightbox} />
            </>
          )}
          <div style={{ marginTop: 8, fontSize: 11, color: 'var(--ink-3)' }}>
            Showing up to 8 sample images per output. Click to enlarge.
          </div>
        </div>
      )}
    </div>
  )
}

const BAND_PRESETS = [
  { key: 'pan',    label: 'Panchromatic — 1 band',                   n: 1,  bands: ['Pan'],                                                                           rgbHR: [1,1,1],   rgbLR: [1,1,1] },
  { key: 'rgb',    label: 'RGB — 3 bands',                            n: 3,  bands: ['R','G','B'],                                                                     rgbHR: [1,2,3],   rgbLR: [1,2,3] },
  { key: 'ms4',    label: 'Multispectral 4-band (Pleiades / Planet)', n: 4,  bands: ['B','G','R','NIR'],                                                               rgbHR: [3,2,1],   rgbLR: [3,2,1] },
  { key: 'ms8',    label: 'Multispectral 8-band (WorldView)',         n: 8,  bands: ['C','B','G','Y','R','RE','NIR1','NIR2'],                                          rgbHR: [5,3,2],   rgbLR: [5,3,2] },
  { key: 's2',     label: 'Sentinel-2 — 13 bands',                   n: 13, bands: ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12'],            rgbHR: [4,3,2],   rgbLR: [4,3,2] },
  { key: 'custom', label: 'Custom…',                                  n: null, bands: [],                                                                              rgbHR: null,      rgbLR: null },
]

function MetaPill({ meta }) {
  if (!meta) return null
  return (
    <div style={{
      display: 'inline-flex', gap: 10, fontSize: 11, color: 'var(--cobalt-deep)',
      background: 'var(--cobalt-soft)', borderRadius: 'var(--radius-sm)',
      padding: '4px 10px', marginTop: 4, flexWrap: 'wrap',
    }}>
      <span>{meta.bands} band{meta.bands !== 1 ? 's' : ''}</span>
      <span>·</span>
      <span>{meta.width} × {meta.height} px</span>
      {meta.format && <><span>·</span><span>{meta.format}</span></>}
    </div>
  )
}

// ── Pipeline A defaults ────────────────────────────────────────────────────────
const DEFAULT_P3 = {
  hr_image_path: '', lr_image_path: '', output_dir: 'output_patches',
  supported_extensions: ['.tif', '.tiff', '.jp2', '.png', '.jpg', '.jpeg', '.bmp'],
  hr_rgb_bands: [1, 2, 3], lr_rgb_bands: [3, 2, 1],
  scale_factor: 2, hr_patch_size: 256, stride: 256,
  nodata_value: 0, saturated_value: 32767, clip_percentiles: [2.0, 98.0],
  max_nodata_fraction: 0.05, min_variance: 120.0, min_ecc_score: 0.78, min_ssim: 0.60,
  radiometric_block_size: 256, radiometric_rmse_threshold: 35.0,
  radiometric_n_samples: 150000, radiometric_post_hist_match: true,
  coreg_a: { enabled: true, max_features: 8000, match_ratio: 0.75, ransac_thresh: 4.0, downsample: 0.25 },
  coreg_b: { enabled: true, downsample: 0.25, upsample_factor: 100 },
  coreg_c: { enabled: true, max_iter: 100, eps: 1e-5, warp_mode: 'translation', discard_on_fail: true },
  class_filter: [],
  train_test_split: false, train_ratio: 0.8, test_output_dir: '',
  // Unpaired-HR degradation (used when lr_image_path is left empty)
  degradation_enabled: true, degradation_type: 'satellite',
  bsrgan: { jpeg_prob: 0.9, scale2_prob: 0.25, isp_prob: 0.25, noise_level1: 2, noise_level2: 25 },
  real_esrgan: {
    blur_prob_1: 1.0, resize_prob_1: 1.0, gaussian_noise_prob_1: 0.5,
    poisson_noise_prob_1: 0.1, speckle_noise_prob_1: 0.1, jpeg_prob_1: 0.9,
    noise_level1_s1: 2, noise_level2_s1: 25, blur_prob_2: 0.8, resize_prob_2: 1.0,
    gaussian_noise_prob_2: 0.5, poisson_noise_prob_2: 0.1, speckle_noise_prob_2: 0.1,
    jpeg_prob_2: 0.8, noise_level1_s2: 2, noise_level2_s2: 15,
    final_jpeg_prob: 0.5, resize_back_prob: 0.5, isp_prob: 0.1,
  },
  bsrgan_plus: {
    shuffle_prob: 0.5, use_sharp: false, sharpening_weight: 0.5,
    sharpening_radius: 50, sharpening_threshold: 10, poisson_prob: 0.1,
    speckle_prob: 0.1, isp_prob: 0.1, noise_level1: 2, noise_level2: 25,
  },
  satellite: {
    blur_prob_1: 1.0, blur_type_1: 'mtf', resize_prob_1: 0.75,
    poisson_prob_1: 0.75, read_noise_prob_1: 0.55, haze_prob_1: 0.45, jpeg_prob_1: 0.12,
    blur_prob_2: 0.92, blur_type_2: 'mtf', resize_prob_2: 0.70,
    poisson_prob_2: 0.60, read_noise_prob_2: 0.45, haze_prob_2: 0.35, jpeg_prob_2: 0.08,
    final_jpeg_prob: 0.10, resize_back_prob: 0.35, isp_prob: 0.08,
    noise_level1: 0.8, noise_level2: 5.0,
    mtf_sigma_optics_range: [0.8, 2.8], mtf_detector_width_range: [0.7, 1.8],
    mtf_atm_sigma_range: [0.4, 1.8],
  },
}

// ── Pipeline B defaults ────────────────────────────────────────────────────────
const DEFAULT_RP = {
  task: 'preprocess_sr_x2', pipeline_mode: 'hr_only',
  degradation_type: 'satellite', scale: 2, n_channels: 3, seed: 42, num_workers: 1,
  input_hr_dir: '', input_lr_dir: '', output_hr_dir: 'trainsets/hr', output_lr_dir: 'trainsets/lr',
  supported_extensions: ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'],
  save_format: 'png', save_hr_copy: true,
  normalize_enabled: false, normalize_low_percentile: 2.0, normalize_high_percentile: 98.0,
  cloud_mask_enabled: false, cloud_mask_threshold: 0.4, cloud_mask_average_over: 4,
  cloud_mask_dilation_size: 2, cloud_mask_nodata: 0.0, cloud_mask_auto_scale: true,
  bsrgan: { jpeg_prob: 0.9, scale2_prob: 0.25, isp_prob: 0.25, noise_level1: 2, noise_level2: 25 },
  real_esrgan: {
    blur_prob_1: 1.0, resize_prob_1: 1.0, gaussian_noise_prob_1: 0.5,
    poisson_noise_prob_1: 0.1, speckle_noise_prob_1: 0.1, jpeg_prob_1: 0.9,
    noise_level1_s1: 2, noise_level2_s1: 25, blur_prob_2: 0.8, resize_prob_2: 1.0,
    gaussian_noise_prob_2: 0.5, poisson_noise_prob_2: 0.1, speckle_noise_prob_2: 0.1,
    jpeg_prob_2: 0.8, noise_level1_s2: 2, noise_level2_s2: 15,
    final_jpeg_prob: 0.5, resize_back_prob: 0.5, isp_prob: 0.1,
  },
  bsrgan_plus: {
    shuffle_prob: 0.5, use_sharp: false, sharpening_weight: 0.5,
    sharpening_radius: 50, sharpening_threshold: 10, poisson_prob: 0.1,
    speckle_prob: 0.1, isp_prob: 0.1, noise_level1: 2, noise_level2: 25,
  },
  satellite: {
    blur_prob_1: 1.0, blur_type_1: 'mtf', resize_prob_1: 0.75,
    poisson_prob_1: 0.75, read_noise_prob_1: 0.55, haze_prob_1: 0.45, jpeg_prob_1: 0.12,
    blur_prob_2: 0.92, blur_type_2: 'mtf', resize_prob_2: 0.70,
    poisson_prob_2: 0.60, read_noise_prob_2: 0.45, haze_prob_2: 0.35, jpeg_prob_2: 0.08,
    final_jpeg_prob: 0.10, resize_back_prob: 0.35, isp_prob: 0.08,
    noise_level1: 0.8, noise_level2: 5.0,
    mtf_sigma_optics_range: [0.8, 2.8], mtf_detector_width_range: [0.7, 1.8],
    mtf_atm_sigma_range: [0.4, 1.8],
  },
  train_test_split: false, train_ratio: 0.8,
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

// ── Pipeline A Component ─────────────────────────────────────────────────────

function Pipeline3Form({ onJobStart }) {
  const [form, setForm] = useState(DEFAULT_P3)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [bandPreset, setBandPreset] = useState('rgb')
  const [customBandCount, setCustomBandCount] = useState(3)
  const [hrMeta, setHrMeta] = useState(null)
  const [lrMeta, setLrMeta] = useState(null)
  const [pairPsnr, setPairPsnr] = useState(null)
  const [hrSubdirs, setHrSubdirs] = useState([])
  const [classSearch, setClassSearch] = useState('')
  const hrTimerRef = useRef(null)
  const lrTimerRef = useRef(null)
  const psnrTimerRef = useRef(null)
  const subdirTimerRef = useRef(null)
  const fileInputRef = useRef(null)

  const set = (path, value) => setForm((prev) => deepSet(prev, path, value))

  const handleLoadDegradationFile = (e) => {
    const file = e.target.files[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = (ev) => {
      try {
        const loaded = JSON.parse(ev.target.result)
        setForm(prev => ({ ...prev, ...loaded }))
      } catch {
        alert('Invalid JSON file.')
      }
    }
    reader.readAsText(file)
    e.target.value = ''
  }

  const handleSaveDegradationFile = () => {
    const blob = new Blob([JSON.stringify(form, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'pipeline_a_config.json'
    a.click()
    URL.revokeObjectURL(url)
  }

  const applyBandPreset = (key, customN = customBandCount) => {
    const preset = BAND_PRESETS.find(p => p.key === key)
    setBandPreset(key)
    if (preset.n) setCustomBandCount(preset.n)
    if (preset.rgbHR) setForm(prev => ({ ...prev, hr_rgb_bands: preset.rgbHR, lr_rgb_bands: preset.rgbLR }))
  }

  useEffect(() => {
    clearTimeout(hrTimerRef.current)
    if (!form.hr_image_path.trim()) { setHrMeta(null); return }
    hrTimerRef.current = setTimeout(() => {
      getImageInfo(form.hr_image_path).then(r => setHrMeta(r.data)).catch(() => setHrMeta(null))
    }, 700)
    return () => clearTimeout(hrTimerRef.current)
  }, [form.hr_image_path])

  useEffect(() => {
    clearTimeout(lrTimerRef.current)
    if (!form.lr_image_path.trim()) { setLrMeta(null); return }
    lrTimerRef.current = setTimeout(() => {
      getImageInfo(form.lr_image_path).then(r => setLrMeta(r.data)).catch(() => setLrMeta(null))
    }, 700)
    return () => clearTimeout(lrTimerRef.current)
  }, [form.lr_image_path])

  useEffect(() => {
    clearTimeout(psnrTimerRef.current)
    if (!form.hr_image_path.trim() || !form.lr_image_path.trim()) { setPairPsnr(null); return }
    psnrTimerRef.current = setTimeout(() => {
      compareImages(form.hr_image_path, form.lr_image_path)
        .then(r => setPairPsnr(r.data))
        .catch(() => setPairPsnr(null))
    }, 1200)
    return () => clearTimeout(psnrTimerRef.current)
  }, [form.hr_image_path, form.lr_image_path])

  // Detect class subfolders when the HR path changes
  useEffect(() => {
    clearTimeout(subdirTimerRef.current)
    const p = form.hr_image_path.trim()
    if (!p) { setHrSubdirs([]); set('class_filter', []); return }
    subdirTimerRef.current = setTimeout(() => {
      listDirectory(p, 'dirs').then(r => {
        const dirs = (r.data.entries || []).filter(e => e.type === 'dir').map(e => e.name)
        setHrSubdirs(dirs)
        // If subdirs are newly detected, default to all selected (empty = all)
        if (dirs.length === 0) set('class_filter', [])
      }).catch(() => setHrSubdirs([]))
    }, 700)
    return () => clearTimeout(subdirTimerRef.current)
  }, [form.hr_image_path])

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      const r = await startPipeline3(form)
      onJobStart(r.data.job_id)
    } catch (err) {
      setError(err.response?.data?.detail || String(err))
    } finally {
      setLoading(false)
    }
  }

  const deg = form.degradation_type

  return (
    <form onSubmit={handleSubmit}>
      {/* ── Paths ── */}
      <CollapsibleSection title="Input / Output Paths" defaultOpen>
        <PathField label="HR image path" hint=".JP2, GeoTIFF, PNG, JPG, BMP…" mode="files"
          extensions=".jp2,.tif,.tiff,.png,.jpg,.jpeg,.bmp,.img,.nitf,.nc,.hdr"
          value={form.hr_image_path} onChange={(v) => set('hr_image_path', v)}
          placeholder="path/to/HR.JP2" />
        <MetaPill meta={hrMeta} />
        <PathField label="LR image path" hint=".JP2, GeoTIFF, PNG, JPG, BMP…" mode="files"
          extensions=".jp2,.tif,.tiff,.png,.jpg,.jpeg,.bmp,.img,.nitf,.nc,.hdr"
          value={form.lr_image_path} onChange={(v) => set('lr_image_path', v)}
          placeholder="path/to/LR.JP2" />
        <MetaPill meta={lrMeta} />

        {/* Class subfolder selector — shown when the HR directory has subdirectories */}
        {hrSubdirs.length > 0 && (() => {
          // form.class_filter stores INCLUDED class names. Empty array = all classes.
          const isSelected = (name) => form.class_filter.length === 0 || form.class_filter.includes(name)
          const toggle = (name) => {
            if (form.class_filter.length === 0) {
              // All selected → uncheck one → include all except this
              set('class_filter', hrSubdirs.filter(n => n !== name))
            } else if (form.class_filter.includes(name)) {
              // Was included → remove
              const next = form.class_filter.filter(n => n !== name)
              set('class_filter', next)
            } else {
              // Was excluded → add back
              const next = [...form.class_filter, name]
              // If now all classes are included, normalize to [] (= all)
              set('class_filter', next.length === hrSubdirs.length ? [] : next)
            }
          }
          return (
            <div className="form-group" style={{ marginTop: 8 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                <label style={{ marginBottom: 0 }}>
                  Class filter
                  <span className="hint" style={{ marginLeft: 6 }}>
                    {form.class_filter.length === 0
                      ? `all ${hrSubdirs.length} classes`
                      : `${form.class_filter.length} / ${hrSubdirs.length} selected`}
                  </span>
                </label>
                <button type="button" className="btn" style={{ padding: '3px 9px', fontSize: 11 }}
                  onClick={() => set('class_filter', [])}>Select all</button>
              </div>
              <input
                className="text-input"
                placeholder="Search classes…"
                value={classSearch}
                onChange={e => setClassSearch(e.target.value)}
                style={{ marginBottom: 6, fontSize: 12 }}
              />
              <div style={{
                maxHeight: 180, overflowY: 'auto', border: '1px solid var(--line-2)',
                borderRadius: 'var(--radius-sm)', padding: '6px 8px',
                display: 'flex', flexDirection: 'column', gap: 3,
              }}>
                {hrSubdirs
                  .filter(name => !classSearch || name.toLowerCase().includes(classSearch.toLowerCase()))
                  .map(name => (
                    <label key={name} style={{
                      display: 'flex', alignItems: 'center', gap: 8,
                      fontSize: 12, cursor: 'pointer', userSelect: 'none',
                      color: isSelected(name) ? 'var(--ink)' : 'var(--ink-3)',
                    }}>
                      <input type="checkbox" checked={isSelected(name)} onChange={() => toggle(name)} />
                      {name}
                    </label>
                  ))}
              </div>
            </div>
          )
        })()}

        <PathField label="Output directory" mode="dirs"
          value={form.output_dir} onChange={(v) => set('output_dir', v)}
          placeholder="output_patches" />
        {pairPsnr && (
          <div style={{
            background: 'var(--surface)', border: '1px solid var(--line-2)',
            borderRadius: 'var(--radius-sm)', padding: '10px 14px', marginTop: 4,
          }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--ink-2)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 8 }}>
              HR / LR Comparison
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
              <span style={{ color: 'var(--ink-2)' }}>PSNR (LR upsampled → HR)</span>
              <span style={{
                fontFamily: 'monospace', fontWeight: 700,
                color: pairPsnr.psnr >= 32 ? 'var(--ok)' : pairPsnr.psnr >= 25 ? 'var(--ink)' : 'var(--bad)',
              }}>
                {pairPsnr.psnr} dB
              </span>
            </div>
            <div style={{ display: 'flex', gap: 16, fontSize: 11, color: 'var(--ink-3)' }}>
              <span>HR: {pairPsnr.hr_bands} bands · {pairPsnr.hr_width}×{pairPsnr.hr_height}</span>
              <span>LR: {pairPsnr.lr_bands} bands · {pairPsnr.lr_width}×{pairPsnr.lr_height}</span>
              <span>{pairPsnr.bands_compared} band{pairPsnr.bands_compared !== 1 ? 's' : ''} compared</span>
            </div>
          </div>
        )}
        <div className="form-group">
          <label>Sensor / input bands</label>
          <select className="text-input" value={bandPreset} onChange={e => applyBandPreset(e.target.value)}>
            {BAND_PRESETS.map(p => <option key={p.key} value={p.key}>{p.label}</option>)}
          </select>
          {bandPreset === 'custom' ? (
            <div style={{ marginTop: 6, display: 'flex', alignItems: 'center', gap: 8 }}>
              <input type="number" className="num-input" style={{ width: 80 }}
                value={customBandCount} min={1} max={200}
                onChange={e => {
                  const n = Math.max(1, parseInt(e.target.value) || 1)
                  setCustomBandCount(n)
                  applyBandPreset('custom', n)
                }} />
              <span style={{ fontSize: 12, color: 'var(--ink-2)' }}>bands</span>
            </div>
          ) : (
            <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap', marginTop: 6 }}>
              {BAND_PRESETS.find(p => p.key === bandPreset)?.bands.map(b => (
                <span key={b} style={{
                  padding: '2px 7px', borderRadius: 10, fontSize: 11, fontWeight: 500,
                  background: 'var(--bg-2)', border: '1px solid var(--line-2)', color: 'var(--ink-2)',
                }}>{b}</span>
              ))}
            </div>
          )}
        </div>
        <div className="grid-2">
          <ArrayEditor label="HR RGB bands (1-indexed)" value={form.hr_rgb_bands}
            onChange={(v) => set('hr_rgb_bands', v)} />
          <ArrayEditor label="LR RGB bands (1-indexed)" value={form.lr_rgb_bands}
            onChange={(v) => set('lr_rgb_bands', v)} />
        </div>
      </CollapsibleSection>

      {/* ── Patch geometry ── */}
      <CollapsibleSection title="Patch Geometry" defaultOpen>
        <div className="grid-3">
          <SelectField label="Scale factor" value={form.scale_factor}
            onChange={(v) => set('scale_factor', parseInt(v))}
            options={[{ value: 2, label: '×2' }, { value: 4, label: '×4' }]} />
          <NumberField label="HR patch size" value={form.hr_patch_size}
            onChange={(v) => set('hr_patch_size', v)} min={64} step={32}
            hint={`LR = ${form.hr_patch_size / form.scale_factor}`}
            tooltip="Size of extracted HR patches in pixels. Corresponding LR patch = HR size ÷ scale factor." />
          <NumberField label="Stride" value={form.stride}
            onChange={(v) => set('stride', v)} min={16} step={16}
            tooltip="Step size (pixels) between consecutive patch windows. Setting lower than patch size creates overlapping patches, yielding more training data at the cost of speed." />
        </div>
      </CollapsibleSection>

      {/* ── Degradation (HR-only mode: no LR path given → synthesize LR from HR patches) ── */}
      {!form.lr_image_path.trim() && (
        <CollapsibleSection title="Degradation" defaultOpen>
          <p style={{ color: 'var(--ink-3)', fontSize: 12, marginBottom: 14 }}>
            No LR image path set — each HR patch will be degraded to synthesize its matching LR patch.
          </p>
          <input
            ref={fileInputRef}
            type="file"
            accept=".json"
            style={{ display: 'none' }}
            onChange={handleLoadDegradationFile}
          />
          <div style={{ display: 'flex', gap: 8, marginBottom: 14 }}>
            <button type="button" className="btn" style={{ fontSize: 12, padding: '5px 12px' }}
              onClick={() => fileInputRef.current?.click()}>
              ↑ Load config
            </button>
            <button type="button" className="btn" style={{ fontSize: 12, padding: '5px 12px' }}
              onClick={handleSaveDegradationFile}>
              ↓ Save full config
            </button>
          </div>
          <SelectField label="Degradation type" value={form.degradation_type}
            onChange={(v) => set('degradation_type', v)}
            options={['bsrgan', 'real_esrgan', 'bsrgan_plus', 'satellite']} />

          {deg === 'bsrgan' && (
            <div className="grid-2">
              <NumberField label="JPEG prob" value={form.bsrgan.jpeg_prob}
                onChange={(v) => set('bsrgan.jpeg_prob', v)} min={0} max={1} step={0.05}
                tooltip="Probability of applying JPEG compression artifact simulation per degradation pass." />
              <NumberField label="ISP prob" value={form.bsrgan.isp_prob}
                onChange={(v) => set('bsrgan.isp_prob', v)} min={0} max={1} step={0.05}
                tooltip="Probability of applying camera ISP pipeline simulation (tone mapping, colour space conversion)." />
              <NumberField label="Noise level 1" value={form.bsrgan.noise_level1}
                onChange={(v) => set('bsrgan.noise_level1', v)} min={0} step={1} />
              <NumberField label="Noise level 2" value={form.bsrgan.noise_level2}
                onChange={(v) => set('bsrgan.noise_level2', v)} min={0} step={5} />
            </div>
          )}

          {deg === 'satellite' && (
            <>
              <p style={{ color: 'var(--ink-3)', fontSize: 12, marginBottom: 16 }}>
                Satellite-optimized: MTF/PSF blur, shot noise, read noise, atmospheric haze.
              </p>
              <div className="grid-2">
                <SelectField label="Stage 1 blur type" value={form.satellite.blur_type_1}
                  onChange={(v) => set('satellite.blur_type_1', v)}
                  options={['mtf', 'anisotropic']} />
                <NumberField label="Stage 1 blur prob" value={form.satellite.blur_prob_1}
                  onChange={(v) => set('satellite.blur_prob_1', v)} min={0} max={1} step={0.05}
                  tooltip="Probability of applying MTF/PSF convolution blur in stage 1 of the two-stage satellite degradation pipeline." />
              </div>
              <div className="grid-2">
                <NumberField label="Poisson prob (stage 1)" value={form.satellite.poisson_prob_1}
                  onChange={(v) => set('satellite.poisson_prob_1', v)} min={0} max={1} step={0.05} />
                <NumberField label="Haze prob (stage 1)" value={form.satellite.haze_prob_1}
                  onChange={(v) => set('satellite.haze_prob_1', v)} min={0} max={1} step={0.05} />
              </div>
              <div className="grid-2">
                <NumberField label="Noise level min" value={form.satellite.noise_level1}
                  onChange={(v) => set('satellite.noise_level1', v)} min={0} step={0.1} />
                <NumberField label="Noise level max" value={form.satellite.noise_level2}
                  onChange={(v) => set('satellite.noise_level2', v)} min={0} step={0.5} />
              </div>
              <ArrayEditor label="MTF optics sigma range [min, max]"
                value={form.satellite.mtf_sigma_optics_range}
                onChange={(v) => set('satellite.mtf_sigma_optics_range', v)}
                integer={false}
                tooltip="Gaussian sigma range [min, max] (px) for optical lens MTF blur. Higher values simulate blurrier optics / larger PSF." />
            </>
          )}

          {deg === 'bsrgan_plus' && (
            <div className="grid-2">
              <NumberField label="Shuffle prob" value={form.bsrgan_plus.shuffle_prob}
                onChange={(v) => set('bsrgan_plus.shuffle_prob', v)} min={0} max={1} step={0.05} />
              <BoolToggle label="Use sharp" value={form.bsrgan_plus.use_sharp}
                onChange={(v) => set('bsrgan_plus.use_sharp', v)} />
            </div>
          )}

          {deg === 'real_esrgan' && (
            <div className="grid-2">
              <NumberField label="Blur prob (stage 1)" value={form.real_esrgan.blur_prob_1}
                onChange={(v) => set('real_esrgan.blur_prob_1', v)} min={0} max={1} step={0.1} />
              <NumberField label="JPEG prob (stage 1)" value={form.real_esrgan.jpeg_prob_1}
                onChange={(v) => set('real_esrgan.jpeg_prob_1', v)} min={0} max={1} step={0.05} />
            </div>
          )}
        </CollapsibleSection>
      )}

      {/* ── Quality filters ── */}
      <CollapsibleSection title="Quality Filters" defaultOpen={false}>
        <div className="grid-2">
          <NumberField label="Max nodata fraction" value={form.max_nodata_fraction}
            onChange={(v) => set('max_nodata_fraction', v)} min={0} max={1} step={0.01}
            tooltip="Maximum fraction of nodata/zero-value pixels allowed per patch. Patches exceeding this are discarded." />
          <NumberField label="Min variance" value={form.min_variance}
            onChange={(v) => set('min_variance', v)} min={0} step={10}
            tooltip="Minimum pixel variance threshold. Featureless patches (uniform cloud, water, bare ground) below this are rejected." />
        </div>
        <div className="grid-2">
          <NumberField label="Min ECC score" value={form.min_ecc_score}
            onChange={(v) => set('min_ecc_score', v)} min={0} max={1} step={0.01}
            tooltip="Per-patch ECC coregistration quality score [0–1]. Patches below this threshold are discarded as misaligned between HR and LR." />
          <NumberField label="Min SSIM" value={form.min_ssim}
            onChange={(v) => set('min_ssim', v)} min={0} max={1} step={0.01}
            tooltip="Structural similarity [0–1] between HR and LR patch. Low values indicate poor co-registration; such patches are rejected." />
        </div>
        <div className="grid-2">
          <NumberField label="Nodata value" value={form.nodata_value}
            onChange={(v) => set('nodata_value', v)} min={0} />
          <NumberField label="Saturated value" value={form.saturated_value}
            onChange={(v) => set('saturated_value', v)} min={1} />
        </div>
      </CollapsibleSection>

      {/* ── Coregistration ── */}
      <CollapsibleSection title="Coregistration" defaultOpen={false}>
        <div className="section-title" style={{ marginTop: 10 }}><h3>Stage A — ORB Keypoint Matching</h3></div>
        <BoolToggle label="Enable Stage A" value={form.coreg_a.enabled}
          onChange={(v) => set('coreg_a.enabled', v)} />
        {form.coreg_a.enabled && (
          <div className="grid-2">
            <NumberField label="Max features" value={form.coreg_a.max_features}
              onChange={(v) => set('coreg_a.max_features', v)} min={0} step={500}
              tooltip="Max ORB keypoints to detect. Higher = better coverage on complex imagery, at the cost of speed." />
            <NumberField label="RANSAC thresh" value={form.coreg_a.ransac_thresh}
              onChange={(v) => set('coreg_a.ransac_thresh', v)} min={0.5} step={0.5}
              tooltip="Max reprojection error (px) for RANSAC inlier selection. Lower = stricter alignment filtering; fewer but higher-quality matches." />
          </div>
        )}
        <div className="section-title" style={{ marginTop: 16 }}><h3>Stage B — Phase Correlation</h3></div>
        <BoolToggle label="Enable Stage B" value={form.coreg_b.enabled}
          onChange={(v) => set('coreg_b.enabled', v)} />
        <div className="section-title" style={{ marginTop: 16 }}><h3>Stage C — Per-patch ECC Refinement</h3></div>
        <BoolToggle label="Enable Stage C" value={form.coreg_c.enabled}
          onChange={(v) => set('coreg_c.enabled', v)} />
        {form.coreg_c.enabled && (
          <div className="grid-2">
            <NumberField label="Max iterations" value={form.coreg_c.max_iter}
              onChange={(v) => set('coreg_c.max_iter', v)} min={10} step={10} />
            <SelectField label="Warp mode" value={form.coreg_c.warp_mode}
              onChange={(v) => set('coreg_c.warp_mode', v)}
              options={['translation', 'euclidean']} />
          </div>
        )}
      </CollapsibleSection>

      {/* ── Radiometric ── */}
      <CollapsibleSection title="Radiometric Normalisation" defaultOpen={false}>
        <div className="grid-2">
          <NumberField label="RMSE threshold" value={form.radiometric_rmse_threshold}
            onChange={(v) => set('radiometric_rmse_threshold', v)} step={1} />
          <NumberField label="Block size" value={form.radiometric_block_size}
            onChange={(v) => set('radiometric_block_size', v)} min={64} step={64} />
        </div>
        <BoolToggle label="Post histogram matching (correct NIR leakage)"
          value={form.radiometric_post_hist_match}
          onChange={(v) => set('radiometric_post_hist_match', v)}
          tooltip="Applies histogram matching after linear regression to correct residual colour / NIR band leakage between spectral channels." />
      </CollapsibleSection>

      {/* ── Train/test split ── */}
      <CollapsibleSection title="Train / Test Split" defaultOpen={false}>
        <BoolToggle label="Apply train/test split after pipeline completes"
          value={form.train_test_split} onChange={(v) => set('train_test_split', v)}
          tooltip="Randomly split extracted patches into separate train and test directories after the pipeline finishes." />
        {form.train_test_split && (
          <>
            <NumberField label="Train ratio" value={form.train_ratio}
              onChange={(v) => set('train_ratio', v)} min={0.1} max={0.99} step={0.05} />
            <PathField label="Test output directory (optional)"
              hint="defaults to OUTPUT_DIR_test" mode="dirs"
              value={form.test_output_dir} onChange={(v) => set('test_output_dir', v)} />
          </>
        )}
      </CollapsibleSection>

      {error && <div style={{ color: 'var(--bad)', fontSize: 13, marginBottom: 12 }}>{error}</div>}

      <button type="submit" className="btn btn-primary full-width" disabled={loading} style={{ marginTop: 8 }}>
        {loading ? 'Starting…' : '▶ Run Pipeline'}
      </button>
    </form>
  )
}

// ── Pipeline B Component ─────────────────────────────────────────────────────

function RunPipelineForm({ onJobStart }) {
  const [form, setForm] = useState(DEFAULT_RP)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const fileInputRef = useRef(null)
  const [classStructure, setClassStructure] = useState(null)
  const detectDebounceRef = useRef(null)
  const [bandPreset, setBandPreset] = useState('rgb')
  const [customBandCount, setCustomBandCount] = useState(3)

  useEffect(() => {
    clearTimeout(detectDebounceRef.current)
    if (!form.input_hr_dir.trim()) { setClassStructure(null); return }
    detectDebounceRef.current = setTimeout(async () => {
      try {
        const r = await detectPreprocessingStructure(form.input_hr_dir)
        setClassStructure(r.data)
      } catch { setClassStructure(null) }
    }, 800)
    return () => clearTimeout(detectDebounceRef.current)
  }, [form.input_hr_dir])

  const set = (path, value) => setForm((prev) => deepSet(prev, path, value))

  const applyBands = (key, customN = customBandCount) => {
    const preset = BAND_PRESETS.find(p => p.key === key)
    const n = preset.n ?? customN
    setBandPreset(key)
    if (preset.n) setCustomBandCount(preset.n)
    set('n_channels', n)
  }

  const handleLoadDegradationFile = (e) => {
    const file = e.target.files[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = (ev) => {
      try {
        const loaded = JSON.parse(ev.target.result)
        setForm(prev => ({ ...prev, ...loaded }))
      } catch {
        alert('Invalid JSON file.')
      }
    }
    reader.readAsText(file)
    e.target.value = ''
  }

  const handleSaveDegradationFile = () => {
    const blob = new Blob([JSON.stringify(form, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'pipeline_b_config.json'
    a.click()
    URL.revokeObjectURL(url)
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      const r = await startRunPipeline(form)
      onJobStart(r.data.job_id)
    } catch (err) {
      setError(err.response?.data?.detail || String(err))
    } finally {
      setLoading(false)
    }
  }

  const deg = form.degradation_type

  return (
    <form onSubmit={handleSubmit}>
      <CollapsibleSection title="Pipeline Settings" defaultOpen>
        <TextField label="Task name" value={form.task} onChange={(v) => set('task', v)} />
        <div className="grid-2">
          <SelectField label="Pipeline mode" value={form.pipeline_mode}
            onChange={(v) => set('pipeline_mode', v)}
            options={[
              { value: 'hr_only', label: 'HR only → degrade to LR' },
              { value: 'hr_lr_pair', label: 'HR+LR pair (no degradation)' },
            ]} />
          <SelectField label="Scale" value={form.scale}
            onChange={(v) => set('scale', parseInt(v))}
            options={[{ value: 2, label: '×2' }, { value: 3, label: '×3' }, { value: 4, label: '×4' }]} />
        </div>
        <div className="grid-2">
          <div className="form-group">
            <label>Input bands</label>
            <select className="text-input" value={bandPreset} onChange={e => applyBands(e.target.value)}>
              {BAND_PRESETS.map(p => <option key={p.key} value={p.key}>{p.label}</option>)}
            </select>
            {bandPreset === 'custom' ? (
              <div style={{ marginTop: 6, display: 'flex', alignItems: 'center', gap: 8 }}>
                <input type="number" className="num-input" style={{ width: 80 }}
                  value={customBandCount} min={1} max={200}
                  onChange={e => {
                    const n = Math.max(1, parseInt(e.target.value) || 1)
                    setCustomBandCount(n)
                    applyBands('custom', n)
                  }} />
                <span style={{ fontSize: 12, color: 'var(--ink-2)' }}>bands</span>
              </div>
            ) : (
              <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap', marginTop: 6 }}>
                {BAND_PRESETS.find(p => p.key === bandPreset)?.bands.map(b => (
                  <span key={b} style={{
                    padding: '2px 7px', borderRadius: 10, fontSize: 11, fontWeight: 500,
                    background: 'var(--bg-2)', border: '1px solid var(--line-2)', color: 'var(--ink-2)',
                  }}>{b}</span>
                ))}
              </div>
            )}
          </div>
          <NumberField label="Workers" value={form.num_workers}
            onChange={(v) => set('num_workers', v)} min={1} />
        </div>
      </CollapsibleSection>

      <CollapsibleSection title="Paths" defaultOpen>
        <PathField label="Input HR path" hint="file, flat dir, or dir with class subfolders" mode="files"
          extensions=".png,.jpg,.jpeg,.tif,.tiff,.bmp,.jp2"
          value={form.input_hr_dir}
          onChange={(v) => set('input_hr_dir', v)} />
        {classStructure && classStructure.is_classed && (
          <div style={{
            background: 'var(--cobalt-soft)', border: '1px solid var(--cobalt-deep)',
            borderRadius: 'var(--radius-sm)', padding: '8px 12px', fontSize: 12,
            color: 'var(--cobalt-deep)', marginTop: 4, marginBottom: 4,
          }}>
            <strong>{classStructure.classes.length} classes detected</strong>
            {' · '}{classStructure.total_images.toLocaleString()} total images
            {classStructure.classes.length <= 10 && (
              <span style={{ color: 'var(--ink-2)', marginLeft: 8 }}>
                ({classStructure.classes.slice(0, 5).join(', ')}{classStructure.classes.length > 5 ? `, +${classStructure.classes.length - 5} more` : ''})
              </span>
            )}
            <div style={{ marginTop: 4, fontSize: 11, color: 'var(--ink-2)' }}>
              Pipeline will run once per class. Outputs → {form.output_hr_dir || 'output_hr'}/<em>class</em>/ and {form.output_lr_dir || 'output_lr'}/<em>class</em>/
            </div>
          </div>
        )}
        {classStructure && !classStructure.is_classed && form.input_hr_dir.trim() && (() => {
          const p = form.input_hr_dir.trim()
          const isFile = /\.[a-zA-Z0-9]+$/.test(p.split(/[/\\]/).pop() || '')
          return (
            <div style={{ fontSize: 11, color: 'var(--ink-3)', marginTop: 2, marginBottom: 4 }}>
              {isFile ? 'Single file — processed as one image' : 'Flat directory — all images processed together'}
            </div>
          )
        })()}
        {form.pipeline_mode === 'hr_lr_pair' && (
          <PathField label="Input LR path" hint="file, flat dir, or dir with class subfolders" mode="files"
            extensions=".png,.jpg,.jpeg,.tif,.tiff,.bmp,.jp2"
            value={form.input_lr_dir}
            onChange={(v) => set('input_lr_dir', v)} />
        )}
        <div className="grid-2">
          <PathField label="Output HR dir" mode="dirs" value={form.output_hr_dir}
            onChange={(v) => set('output_hr_dir', v)} />
          <PathField label="Output LR dir" mode="dirs" value={form.output_lr_dir}
            onChange={(v) => set('output_lr_dir', v)} />
        </div>
        <div className="grid-2">
          <SelectField label="Save format" value={form.save_format}
            onChange={(v) => set('save_format', v)}
            options={['png', 'tif', 'jpg']} />
          <BoolToggle label="Save HR copy" value={form.save_hr_copy}
            onChange={(v) => set('save_hr_copy', v)}
            tooltip="Save a copy of the (optionally normalised) HR image to the output HR directory alongside the synthetic LR." />
        </div>
      </CollapsibleSection>

      {form.pipeline_mode === 'hr_only' && (
        <CollapsibleSection title="Degradation" defaultOpen>
          <input
            ref={fileInputRef}
            type="file"
            accept=".json"
            style={{ display: 'none' }}
            onChange={handleLoadDegradationFile}
          />
          <div style={{ display: 'flex', gap: 8, marginBottom: 14 }}>
            <button type="button" className="btn" style={{ fontSize: 12, padding: '5px 12px' }}
              onClick={() => fileInputRef.current?.click()}>
              ↑ Load config
            </button>
            <button type="button" className="btn" style={{ fontSize: 12, padding: '5px 12px' }}
              onClick={handleSaveDegradationFile}>
              ↓ Save full config
            </button>
          </div>
          <SelectField label="Degradation type" value={form.degradation_type}
            onChange={(v) => set('degradation_type', v)}
            options={['bsrgan', 'real_esrgan', 'bsrgan_plus', 'satellite']} />

          {deg === 'bsrgan' && (
            <div className="grid-2">
              <NumberField label="JPEG prob" value={form.bsrgan.jpeg_prob}
                onChange={(v) => set('bsrgan.jpeg_prob', v)} min={0} max={1} step={0.05}
                tooltip="Probability of applying JPEG compression artifact simulation per degradation pass." />
              <NumberField label="ISP prob" value={form.bsrgan.isp_prob}
                onChange={(v) => set('bsrgan.isp_prob', v)} min={0} max={1} step={0.05}
                tooltip="Probability of applying camera ISP pipeline simulation (tone mapping, colour space conversion)." />
              <NumberField label="Noise level 1" value={form.bsrgan.noise_level1}
                onChange={(v) => set('bsrgan.noise_level1', v)} min={0} step={1} />
              <NumberField label="Noise level 2" value={form.bsrgan.noise_level2}
                onChange={(v) => set('bsrgan.noise_level2', v)} min={0} step={5} />
            </div>
          )}

          {deg === 'satellite' && (
            <>
              <p style={{ color: 'var(--ink-3)', fontSize: 12, marginBottom: 16 }}>
                Satellite-optimized: MTF/PSF blur, shot noise, read noise, atmospheric haze.
              </p>
              <div className="grid-2">
                <SelectField label="Stage 1 blur type" value={form.satellite.blur_type_1}
                  onChange={(v) => set('satellite.blur_type_1', v)}
                  options={['mtf', 'anisotropic']} />
                <NumberField label="Stage 1 blur prob" value={form.satellite.blur_prob_1}
                  onChange={(v) => set('satellite.blur_prob_1', v)} min={0} max={1} step={0.05}
                  tooltip="Probability of applying MTF/PSF convolution blur in stage 1 of the two-stage satellite degradation pipeline." />
              </div>
              <div className="grid-2">
                <NumberField label="Poisson prob (stage 1)" value={form.satellite.poisson_prob_1}
                  onChange={(v) => set('satellite.poisson_prob_1', v)} min={0} max={1} step={0.05} />
                <NumberField label="Haze prob (stage 1)" value={form.satellite.haze_prob_1}
                  onChange={(v) => set('satellite.haze_prob_1', v)} min={0} max={1} step={0.05} />
              </div>
              <div className="grid-2">
                <NumberField label="Noise level min" value={form.satellite.noise_level1}
                  onChange={(v) => set('satellite.noise_level1', v)} min={0} step={0.1} />
                <NumberField label="Noise level max" value={form.satellite.noise_level2}
                  onChange={(v) => set('satellite.noise_level2', v)} min={0} step={0.5} />
              </div>
              <ArrayEditor label="MTF optics sigma range [min, max]"
                value={form.satellite.mtf_sigma_optics_range}
                onChange={(v) => set('satellite.mtf_sigma_optics_range', v)}
                integer={false}
                tooltip="Gaussian sigma range [min, max] (px) for optical lens MTF blur. Higher values simulate blurrier optics / larger PSF." />
            </>
          )}

          {deg === 'bsrgan_plus' && (
            <div className="grid-2">
              <NumberField label="Shuffle prob" value={form.bsrgan_plus.shuffle_prob}
                onChange={(v) => set('bsrgan_plus.shuffle_prob', v)} min={0} max={1} step={0.05} />
              <BoolToggle label="Use sharp" value={form.bsrgan_plus.use_sharp}
                onChange={(v) => set('bsrgan_plus.use_sharp', v)} />
            </div>
          )}

          {deg === 'real_esrgan' && (
            <div className="grid-2">
              <NumberField label="Blur prob (stage 1)" value={form.real_esrgan.blur_prob_1}
                onChange={(v) => set('real_esrgan.blur_prob_1', v)} min={0} max={1} step={0.1} />
              <NumberField label="JPEG prob (stage 1)" value={form.real_esrgan.jpeg_prob_1}
                onChange={(v) => set('real_esrgan.jpeg_prob_1', v)} min={0} max={1} step={0.05} />
            </div>
          )}
        </CollapsibleSection>
      )}

      <CollapsibleSection title="Normalisation" defaultOpen={false}>
        <BoolToggle label="Enable percentile normalisation"
          value={form.normalize_enabled} onChange={(v) => set('normalize_enabled', v)}
          tooltip="Scale pixel values using percentile clipping before saving, ensuring outputs land in a consistent 0–255 display range." />
        {form.normalize_enabled && (
          <div className="grid-2">
            <NumberField label="Low percentile" value={form.normalize_low_percentile}
              onChange={(v) => set('normalize_low_percentile', v)} min={0} max={49} step={0.5}
              tooltip="Pixel value at this percentile maps to 0 in the normalised output (dark clip point)." />
            <NumberField label="High percentile" value={form.normalize_high_percentile}
              onChange={(v) => set('normalize_high_percentile', v)} min={51} max={100} step={0.5}
              tooltip="Pixel value at this percentile maps to 255 in the normalised output (bright clip point)." />
          </div>
        )}
      </CollapsibleSection>

      <CollapsibleSection title="Train / Test Split" defaultOpen={false}>
        <BoolToggle label="Split outputs into train/test sets after completion"
          value={form.train_test_split} onChange={(v) => set('train_test_split', v)}
          tooltip="Shuffle and split all processed output images into train/test directories after the pipeline completes." />
        {form.train_test_split && (
          <NumberField label="Train ratio" value={form.train_ratio}
            onChange={(v) => set('train_ratio', v)} min={0.1} max={0.99} step={0.05} />
        )}
      </CollapsibleSection>

      {error && <div style={{ color: 'var(--bad)', fontSize: 13, marginBottom: 12 }}>{error}</div>}

      <button type="submit" className="btn btn-primary full-width" disabled={loading} style={{ marginTop: 8 }}>
        {loading ? 'Starting…' : '▶ Run Pipeline'}
      </button>
    </form>
  )
}

// ── Main Preprocessing page ─────────────────────────────────────────────────

export default function Preprocessing() {
  const [activeTab, setActiveTab] = useState(0)
  const { jobs, setJobId: setCtxJobId } = useJobContext()
  const jobId = jobs['preprocessing']
  const setJobId = (id) => setCtxJobId('preprocessing', id)
  const [previewMap, setPreviewMap] = useState({})
  const [classResults, setClassResults] = useState([])
  const logPanelRef = useRef(null)

  const handleStop = async () => {
    if (jobId) await stopPreprocessing(jobId).catch(() => { })
  }

  const handlePause = async () => {
    if (jobId) await pausePreprocessing(jobId).catch(() => { })
  }

  const handleResume = async () => {
    if (jobId) await resumePreprocessing(jobId).catch(() => { })
  }

  const handleJobStart = (jid) => {
    setJobId(jid)
    setPreviewMap({})
    setClassResults([])
    setTimeout(() => {
      logPanelRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }, 50)
  }

  const handleLogLine = (line) => {
    if (line.startsWith('[CLASS_DONE] ')) {
      try {
        const data = JSON.parse(line.slice('[CLASS_DONE] '.length))
        setClassResults(prev => [...prev, data])
      } catch { }
    }
  }

  const previewCount = Object.keys(previewMap).length
  const hasClassResults = classResults.length > 0
  const tabs = [
    { label: 'Pipeline A — Pleiades' },
    { label: 'Pipeline B — HR Degradation' },
    {
      label: hasClassResults ? 'Class Results' : 'Step Preview',
      badge: hasClassResults ? classResults.length : previewCount > 0 ? previewCount : null,
    },
  ]

  return (
    <div>
      <div className="topbar">
        <div className="topbar-title">
          <h2>Preprocessing</h2>
        </div>
      </div>

      <div className="content">
        <h1 className="editorial rise" style={{ fontSize: 32, marginBottom: 10 }}>Dataset Preparation</h1>
        <p className="rise" style={{ color: 'var(--ink-2)', marginBottom: 24, maxWidth: 600 }}>
          Prepare HR/LR training pairs from raw satellite imagery or degrade existing HR patches.
        </p>

        <div className="mode-tabs rise" style={{ marginBottom: 28, animationDelay: '80ms' }}>
          {tabs.map((t, i) => (
            <button key={i} type="button"
              className={`mode-tab ${activeTab === i ? 'active' : ''}`}
              onClick={() => setActiveTab(i)}>
              {t.label}
              {t.badge && (
                <span style={{
                  marginLeft: 6, fontSize: 10, fontWeight: 700,
                  background: 'var(--cobalt-deep)', color: '#fff',
                  borderRadius: 10, padding: '1px 6px',
                }}>
                  {t.badge}
                </span>
              )}
            </button>
          ))}
        </div>

        {activeTab < 2 && (
          <div className="module-grid rise" style={{ animationDelay: '100ms' }}>
            <div className="col">
              {activeTab === 0 && (
                <div className="animate-in">
                  <div style={{ marginBottom: 12 }}>
                    <span style={{ fontSize: 15, fontWeight: 600 }}>Pleiades / Multi-sensor Patch Extraction</span>
                  </div>
                  <ol style={{ color: 'var(--ink-3)', fontSize: 12.5, marginBottom: 20, lineHeight: 1.7, paddingLeft: 18 }}>
                    <li>Load paired HR + LR GeoTIFF or JP2 satellite images</li>
                    <li>Stage A — ORB keypoint matching &amp; RANSAC homography (coarse global alignment)</li>
                    <li>Stage B — Phase correlation FFT sub-pixel shift</li>
                    <li>Stage C — Per-patch ECC refinement (local alignment)</li>
                    <li>Radiometric regression (linear LR→HR normalisation) + optional histogram matching</li>
                    <li>Sliding-window patch extraction with quality filters (variance, nodata, SSIM, ECC)</li>
                    <li>Optional train/test split of extracted patches</li>
                  </ol>
                  <Pipeline3Form onJobStart={handleJobStart} />
                </div>
              )}
              {activeTab === 1 && (
                <div className="animate-in">
                  <div style={{ marginBottom: 12 }}>
                    <span style={{ fontSize: 15, fontWeight: 600 }}>HR-only / HR+LR Pair Preprocessing</span>
                  </div>
                  <ol style={{ color: 'var(--ink-3)', fontSize: 12.5, marginBottom: 20, lineHeight: 1.7, paddingLeft: 18 }}>
                    <li>Load HR images (and optionally matching LR images for HR+LR pair mode)</li>
                    <li>Optional cloud masking (Sentinel-2 s2cloudless, 10-band)</li>
                    <li>Optional percentile normalisation (scales to 8-bit output range)</li>
                    <li><strong>HR-only mode:</strong> degrade HR → synthetic LR via BSRGAN / Real-ESRGAN / BSRGAN+ / Satellite MTF</li>
                    <li><strong>HR+LR pair mode:</strong> preprocess both HR and LR as-is without degradation</li>
                    <li>Save HR and LR images to output directories in chosen format (PNG / TIF / JPG)</li>
                    <li>Optional train/test split of saved images</li>
                  </ol>
                  <RunPipelineForm onJobStart={handleJobStart} />
                </div>
              )}
            </div>
            <div className="col" ref={logPanelRef}>
              {jobId && (
                <LogConsole
                  domain="preprocessing"
                  jobId={jobId}
                  onStop={handleStop}
                  onPause={handlePause}
                  onResume={handleResume}
                  onPreviewsChange={setPreviewMap}
                  onLine={handleLogLine}
                />
              )}
            </div>
          </div>
        )}

        {activeTab === 2 && (
          <div className="rise" style={{ animationDelay: '80ms' }}>
            {hasClassResults && <ClassResultsPanel results={classResults} />}
            {previewCount > 0 && <StepPreviewPanel previews={previewMap} />}
            {!hasClassResults && previewCount === 0 && <StepPreviewPanel previews={{}} />}
          </div>
        )}
      </div>
    </div>
  )
}
