import { useState, useEffect, useRef } from 'react'
import { listDirectory } from '../api/client'

/**
 * DirBrowser — modal file/folder picker backed by /api/fs/list
 *
 * Props:
 *   initial     — starting path (empty = project root)
 *   mode        — 'dirs' | 'files' | 'both'
 *   extensions  — comma-separated extensions when mode includes files, e.g. '.pth,.pt'
 *   onSelect    — called with the chosen path string
 *   onClose     — called when dismissed without selection
 */
export default function DirBrowser({ initial = '', mode = 'dirs', extensions = '', onSelect, onClose }) {
  const [current, setCurrent] = useState('')
  const [parent, setParent] = useState('')
  const [entries, setEntries] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [manualPath, setManualPath] = useState(initial)
  const inputRef = useRef(null)

  const load = async (path, { autoSelect = false } = {}) => {
    setLoading(true)
    setError('')
    try {
      const r = await listDirectory(path, mode, extensions)
      // Auto-select only when the user explicitly typed a path and hit Go.
      // On mount or folder-click navigation, autoSelect is false so we just browse.
      if (autoSelect) {
        if (r.data.typed_file && mode === 'files') { onSelect(r.data.typed_file); return }
        if (r.data.typed_dir  && mode === 'dirs')  { onSelect(r.data.typed_dir);  return }
      }
      setCurrent(r.data.current)
      setParent(r.data.parent)
      setEntries(r.data.entries)
      setManualPath(r.data.current)
    } catch (e) {
      setError(e.response?.data?.detail || 'Failed to list directory')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load(initial)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    const handler = (e) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose])

  const handleManualNav = (e) => {
    e.preventDefault()
    load(manualPath, { autoSelect: true })
  }

  const dirs = entries.filter(e => e.type === 'dir')
  const files = entries.filter(e => e.type === 'file')

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.5)',
        zIndex: 9000, display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}
    >
      <div
        onClick={e => e.stopPropagation()}
        style={{
          background: 'var(--surface)', border: '1px solid var(--line)',
          borderRadius: 'var(--radius)', boxShadow: 'var(--shadow-lg)',
          width: 560, maxWidth: '95vw', maxHeight: '80vh',
          display: 'flex', flexDirection: 'column',
        }}
      >
        {/* Header */}
        <div style={{ padding: '14px 18px', borderBottom: '1px solid var(--line)', display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{ flex: 1, fontSize: 14, fontWeight: 600, color: 'var(--ink)' }}>
            {mode === 'files' ? 'Select File' : mode === 'dirs' ? 'Select Folder' : 'Browse'}
          </div>
          <button type="button" onClick={onClose} style={{ background: 'none', border: 'none', cursor: 'pointer', fontSize: 18, color: 'var(--ink-3)', lineHeight: 1 }}>✕</button>
        </div>

        {/* Path bar */}
        <form onSubmit={handleManualNav} style={{ padding: '10px 18px', borderBottom: '1px solid var(--line-2)', display: 'flex', gap: 6 }}>
          <input
            ref={inputRef}
            type="text"
            value={manualPath}
            onChange={e => setManualPath(e.target.value)}
            className="text-input"
            style={{ flex: 1, fontFamily: 'var(--font-mono)', fontSize: 12 }}
            placeholder="Type or paste a path…"
          />
          <button type="submit" className="btn" style={{ fontSize: 12, padding: '4px 12px' }}>Go</button>
          {manualPath.trim() && (
            <button
              type="button"
              className="btn btn-primary"
              style={{ fontSize: 12, padding: '4px 12px', whiteSpace: 'nowrap' }}
              onClick={() => onSelect(manualPath.trim())}
            >
              ✓ Use this path
            </button>
          )}
        </form>

        {/* Entry list */}
        <div style={{ flex: 1, overflow: 'auto', padding: '6px 0' }}>
          {loading && (
            <div style={{ textAlign: 'center', padding: 24, color: 'var(--ink-3)', fontSize: 13 }}>Loading…</div>
          )}
          {error && (
            <div style={{ textAlign: 'center', padding: 24, color: 'var(--bad)', fontSize: 13 }}>{error}</div>
          )}

          {!loading && !error && (
            <>
              {/* Up one level */}
              {parent && (
                <button
                  type="button"
                  onClick={() => load(parent)}
                  style={rowStyle}
                  className="fs-row"
                >
                  <span style={iconStyle}>↑</span>
                  <span style={{ color: 'var(--ink-3)', fontStyle: 'italic' }}>.. (up)</span>
                </button>
              )}

              {/* Directories */}
              {dirs.map(d => (
                <button
                  key={d.path}
                  type="button"
                  onClick={() => load(d.path)}
                  style={rowStyle}
                  className="fs-row"
                >
                  <span style={iconStyle}>📁</span>
                  <span style={{ flex: 1, textAlign: 'left', color: 'var(--ink)' }}>{d.name}</span>
                  {mode === 'dirs' && (
                    <span
                      onClick={e => { e.stopPropagation(); onSelect(d.path) }}
                      style={{ fontSize: 11, color: 'var(--cobalt-deep)', padding: '2px 8px', borderRadius: 4, border: '1px solid var(--cobalt-deep)', cursor: 'pointer' }}
                    >
                      Select
                    </span>
                  )}
                </button>
              ))}

              {/* Files */}
              {files.map(f => (
                <button
                  key={f.path}
                  type="button"
                  onClick={() => onSelect(f.path)}
                  style={rowStyle}
                  className="fs-row"
                >
                  <span style={iconStyle}>📄</span>
                  <span style={{ flex: 1, textAlign: 'left', color: 'var(--ink)' }}>{f.name}</span>
                </button>
              ))}

              {dirs.length === 0 && files.length === 0 && !loading && (
                <div style={{ textAlign: 'center', padding: 24, color: 'var(--ink-3)', fontSize: 13 }}>
                  No entries found
                </div>
              )}
            </>
          )}
        </div>

        {/* Footer */}
        <div style={{ padding: '12px 18px', borderTop: '1px solid var(--line)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 10 }}>
          <div style={{ fontSize: 11, color: 'var(--ink-3)', fontFamily: 'var(--font-mono)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }}>
            {current}
          </div>
          {mode !== 'files' && (
            <button
              type="button"
              className="btn btn-primary"
              style={{ fontSize: 12, padding: '5px 16px', whiteSpace: 'nowrap' }}
              onClick={() => onSelect(current)}
            >
              Select this folder
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

const rowStyle = {
  display: 'flex', alignItems: 'center', gap: 10,
  width: '100%', padding: '7px 18px',
  background: 'none', border: 'none', cursor: 'pointer',
  textAlign: 'left', fontSize: 13,
}
const iconStyle = { fontSize: 15, width: 20, flexShrink: 0 }
