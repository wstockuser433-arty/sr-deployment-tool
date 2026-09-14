import { useEffect, useState } from 'react'
import { getImageInfo } from '../api/client'

/**
 * Debounced image-metadata fetch.
 * Returns null while loading, while the path is empty, or on error.
 */
export default function useImageMeta(path, delay = 700) {
  const [meta, setMeta] = useState(null)
  useEffect(() => {
    if (!path || !path.trim()) { setMeta(null); return }
    const t = setTimeout(() => {
      getImageInfo(path).then(r => setMeta(r.data)).catch(() => setMeta(null))
    }, delay)
    return () => clearTimeout(t)
  }, [path, delay])
  return meta
}