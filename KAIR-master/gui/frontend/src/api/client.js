import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  headers: { 'Content-Type': 'application/json' },
})

export const checkHealth = () => api.get('/health')

// ── Training ──────────────────────────────────────────────────────────────────
export const listTrainingConfigs = () => api.get('/training/configs')
export const getTrainingConfig = (name) => api.get(`/training/config/${encodeURIComponent(name)}`)
export const listTrainingRuns = () => api.get('/training/runs')
export const startTraining = (data) => api.post('/training/start', data)
export const stopTraining = (jobId) => api.post(`/training/stop/${jobId}`)
export const pauseTraining = (jobId) => api.post(`/training/pause/${jobId}`)
export const resumeTraining = (jobId) => api.post(`/training/resume/${jobId}`)
export const getTrainingStatus = (jobId) => api.get(`/training/status/${jobId}`)

// ── Inference ─────────────────────────────────────────────────────────────────
export const listInferenceTasks = () => api.get('/inference/tasks')
export const getLatestModel = (taskName) => api.get(`/inference/latest-model/${encodeURIComponent(taskName)}`)
export const getConfigFromOptions = (name) => api.get(`/inference/config-from-options/${encodeURIComponent(name)}`)
export const getConfigFromPath = (path) => api.get('/inference/config-from-path', { params: { path } })
export const startInference = (data) => api.post('/inference/start', data)
export const stopInference = (jobId) => api.post(`/inference/stop/${jobId}`)
export const getInferenceStatus = (jobId) => api.get(`/inference/status/${jobId}`)
// Raw satellite image inference
export const startRawPairedInference = (data) => api.post('/inference/raw-paired/start', data)
export const startLROnlyInference = (data) => api.post('/inference/lr-only/start', data)
export const getRawInferenceMetrics = (jobId) => api.get(`/inference/raw/metrics/${jobId}`)
export const getRawResultImageUrl = (jobId, filename) => `/api/inference/raw/result/${jobId}/${filename}`
export const getImageInfo = (path) => api.get('/inference/image-info', { params: { path } })
export const compareImages = (hr, lr) => api.get('/inference/image-compare', { params: { hr, lr } })

export const getInferenceSummary = (params) => api.get('/inference/inference-summary', { params })
export const getInferenceProgress = (jobId) => api.get(`/inference/progress/${jobId}`)

// ── Preprocessing ─────────────────────────────────────────────────────────────
export const startPipeline3 = (data) => api.post('/preprocessing/pipeline3/start', data)
export const startRunPipeline = (data) => api.post('/preprocessing/run-pipeline/start', data)
export const stopPreprocessing = (jobId) => api.post(`/preprocessing/stop/${jobId}`)
export const pausePreprocessing = (jobId) => api.post(`/preprocessing/pause/${jobId}`)
export const resumePreprocessing = (jobId) => api.post(`/preprocessing/resume/${jobId}`)
export const getPreprocessingStatus = (jobId) => api.get(`/preprocessing/status/${jobId}`)
export const detectPreprocessingStructure = (path) =>
  api.get('/preprocessing/detect-structure', { params: { path } })
export const getPreprocessingProgress = (jobId) =>
  api.get(`/preprocessing/progress/${jobId}`)
export const getPreprocessingSummary = (params) =>
  api.get('/preprocessing/preprocessing-summary', { params })


// ── Filesystem browser ────────────────────────────────────────────────────────
export const listDirectory = (path = '', mode = 'dirs', extensions = '') =>
  api.get('/fs/list', { params: { path, mode, extensions } })

/** Returns a URL string (not a promise) for use directly in <img src=...> */
export const fsImageUrl = (path) => `/api/fs/image?path=${encodeURIComponent(path)}`

// ── SSE helper (returns EventSource) ─────────────────────────────────────────
export function openLogStream(domain, jobId, onLine, onStatus) {
  const es = new EventSource(`/api/${domain}/stream/${jobId}`)
  es.onmessage = (e) => onLine(e.data)
  es.addEventListener('status', (e) => {
    // Terminal status — close the stream
    onStatus(e.data)
    es.close()
  })
  es.addEventListener('status_update', (e) => {
    // Non-terminal status change (e.g. running <-> paused) — update without closing
    onStatus(e.data)
  })
  es.onerror = () => {
    onStatus('failed')
    es.close()
  }
  return es
}
