import { createContext, useContext, useState, useEffect, useRef } from 'react'
import { api } from '../api/client'

const JobContext = createContext(null)

const STORAGE_KEY = 'kair_jobs'
const DOMAINS = ['training', 'preprocessing', 'inference-patched', 'inference-raw', 'inference-lr']

// Domain → status endpoint used to validate a persisted job ID.
// Use the actual route you expose; if a domain has no status endpoint, map it to null
// and the reconciler will treat all persisted IDs for that domain as stale.
const STATUS_ENDPOINTS = {
  'training':           (id) => `/training/status/${id}`,
  'preprocessing':      (id) => `/preprocessing/status/${id}`,
  'inference-patched':  (id) => `/inference/status/${id}`,
  'inference-raw':      (id) => `/inference/status/${id}`,
  'inference-lr':       (id) => `/inference/status/${id}`,
}

const TERMINAL = new Set(['completed', 'failed', 'cancelled'])

function emptyJobs() {
  return Object.fromEntries(DOMAINS.map((d) => [d, null]))
}

function loadJobs() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}')
    return Object.fromEntries(DOMAINS.map((d) => [d, saved[d] ?? null]))
  } catch {
    return emptyJobs()
  }
}

export function JobProvider({ children }) {
  const [jobs, setJobs] = useState(loadJobs)
  const [hydrated, setHydrated] = useState(false)
  const reconciledRef = useRef(false)

  // ── Reconcile persisted IDs against the backend, once, on mount ──
  useEffect(() => {
  let cancelled = false

  const reconcile = async () => {
    const initial = loadJobs()
    const checks = DOMAINS.map(async (domain) => {
      const id = initial[domain]
      if (!id) return [domain, null]
      const build = STATUS_ENDPOINTS[domain]
      if (!build) return [domain, null]
      try {
        const r = await api.get(build(id))
        const status = r?.data?.status
        if (status && TERMINAL.has(status)) return [domain, null]
        return [domain, id]
      } catch (err) {
        if (err?.response?.status === 404) return [domain, null]
        return [domain, id]
      }
    })
    const pairs = await Promise.all(checks)
    if (cancelled) return
    const next = Object.fromEntries(pairs)
    setJobs(next)
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(next)) } catch {}
    setHydrated(true)
  }

  reconcile()
  const onFocus = () => reconcile()
  window.addEventListener('focus', onFocus)
  return () => {
    cancelled = true
    window.removeEventListener('focus', onFocus)
  }
}, [])

  // Persist after the first reconcile, not before (avoids clobbering with stale data)
  useEffect(() => {
    if (!hydrated) return
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(jobs)) } catch {}
  }, [jobs, hydrated])

  const setJobId = (domain, jobId) =>
    setJobs((prev) => ({ ...prev, [domain]: jobId }))

  const clearJobId = (domain) =>
    setJobs((prev) => ({ ...prev, [domain]: null }))

  return (
    <JobContext.Provider value={{ jobs, setJobId, clearJobId, hydrated }}>
      {children}
    </JobContext.Provider>
  )
}

export function useJobContext() {
  const ctx = useContext(JobContext)
  if (!ctx) throw new Error('useJobContext must be used inside JobProvider')
  return ctx
}