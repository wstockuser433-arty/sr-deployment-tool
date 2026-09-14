import { createContext, useContext, useState, useEffect, useRef } from 'react'
// import api from '../api/client'   // ← adjust to your axios instance / fetch helper

const JobContext = createContext(null)

const STORAGE_KEY = 'kair_jobs'
const DOMAINS = ['training', 'preprocessing', 'inference-patched', 'inference-raw', 'inference-lr']

// Domain → status endpoint used to validate a persisted job ID.
// Use the actual route you expose; if a domain has no status endpoint, map it to null
// and the reconciler will treat all persisted IDs for that domain as stale.
const STATUS_ENDPOINTS = {
  'training':           (id) => `/api/training/status/${id}`,
  'preprocessing':      (id) => `/api/preprocessing/status/${id}`,
  'inference-patched':  (id) => `/api/inference/status/${id}`,
  'inference-raw':      (id) => `/api/inference/status/${id}`,
  'inference-lr':       (id) => `/api/inference/status/${id}`,
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
    if (reconciledRef.current) return
    reconciledRef.current = true

    const initial = loadJobs()
    const checks = DOMAINS.map(async (domain) => {
      const id = initial[domain]
      if (!id) return [domain, null]

      const build = STATUS_ENDPOINTS[domain]
      if (!build) return [domain, null]  // no way to validate → drop

      try {
        const r = await api.get(build(id))
        const status = r?.data?.status
        if (!status || TERMINAL.has(status)) {
          // Job is done, dead, or unreachable → clear it. Don't leave a stale
          // "running" pill on the page after a restart.
          return [domain, null]
        }
        // Still alive on the backend → keep it
        return [domain, id]
      } catch {
        return [domain, null]
      }
    })

    Promise.all(checks).then((pairs) => {
      const next = Object.fromEntries(pairs)
      setJobs(next)
      try { localStorage.setItem(STORAGE_KEY, JSON.stringify(next)) } catch {}
      setHydrated(true)
    })
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