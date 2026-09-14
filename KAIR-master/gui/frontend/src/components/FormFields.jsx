import { useState, lazy, Suspense } from 'react'
const DirBrowser = lazy(() => import('./DirBrowser'))

function FieldTooltip({ text }) {
  const [show, setShow] = useState(false)
  return (
    <span style={{ position: 'relative', display: 'inline-block', marginLeft: 5, verticalAlign: 'middle', zIndex: 1000 }}>
      <span
        onMouseEnter={() => setShow(true)}
        onMouseLeave={() => setShow(false)}
        style={{
          cursor: 'help', fontSize: 10, fontWeight: 700,
          color: 'var(--ink-3)', border: '1px solid var(--line-2)',
          borderRadius: '50%', width: 14, height: 14,
          display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
          userSelect: 'none', lineHeight: 1,
        }}
      >?</span>
      {show && (
        <div style={{
          position: 'absolute', bottom: 'calc(100% + 6px)', left: '50%',
          transform: 'translateX(-50%)',
          background: 'var(--ink)', color: '#fff',
          fontSize: 11, lineHeight: 1.55, padding: '7px 10px',
          borderRadius: 'var(--radius-sm)', width: 240,
          zIndex: 1000, boxShadow: '0 4px 16px rgba(0,0,0,0.35)',
          pointerEvents: 'none', wordBreak: 'break-word',
        }}>
          {text}
        </div>
      )}
    </span>
  )
}

/**
 * ArrayEditor — editable list of numbers
 */
export function ArrayEditor({ value, onChange, integer = true, label, disabled, tooltip }) {
  const update = (i, v) => {
    const next = [...value]
    next[i] = integer ? parseInt(v) || 0 : parseFloat(v) || 0
    onChange(next)
  }
  const add = () => onChange([...value, integer ? 6 : 1.0])
  const remove = (i) => onChange(value.filter((_, idx) => idx !== i))

  return (
    <div className="form-group">
      {label && (
        <label>
          {label}
          {tooltip && <FieldTooltip text={tooltip} />}
        </label>
      )}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, alignItems: 'center' }}>
        {value.map((v, i) => (
          <input
            key={i}
            type="number"
            className="num-input"
            style={{ width: '60px' }}
            value={v}
            onChange={(e) => update(i, e.target.value)}
            step={integer ? 1 : 0.1}
            disabled={disabled}
          />
        ))}
        {!disabled && (
          <>
            <button type="button" className="btn btn-ghost icon-btn" onClick={add}>+</button>
            {value.length > 1 && (
              <button type="button" className="btn btn-ghost icon-btn" onClick={() => remove(value.length - 1)}>−</button>
            )}
          </>
        )}
      </div>
    </div>
  )
}

/**
 * BoolToggle — labelled toggle switch
 */
export function BoolToggle({ label, value, onChange, hint, disabled, tooltip }) {
  return (
    <div className="form-group">
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <label className="switch">
          <input type="checkbox" checked={value} onChange={(e) => onChange(e.target.checked)} disabled={disabled} />
          <span className="track" />
        </label>
        <span style={{ fontSize: 13, color: 'var(--ink-2)' }}>
          {label}
          {hint && <span className="hint">{hint}</span>}
          {tooltip && <FieldTooltip text={tooltip} />}
        </span>
      </div>
    </div>
  )
}

/**
 * SelectField — labelled <select>
 */
export function SelectField({ label, value, onChange, options, hint, disabled, tooltip }) {
  return (
    <div className="form-group">
      <label>
        {label}
        {hint && <span className="hint">{hint}</span>}
        {tooltip && <FieldTooltip text={tooltip} />}
      </label>
      <select className="text-input" value={value} onChange={(e) => onChange(e.target.value)} disabled={disabled}>
        {options.map((o) => (
          <option key={o.value ?? o} value={o.value ?? o}>{o.label ?? o}</option>
        ))}
      </select>
    </div>
  )
}

/**
 * TextField — labelled text input
 */
export function TextField({ label, value, onChange, placeholder, hint, mono, disabled, tooltip }) {
  return (
    <div className="form-group">
      <label>
        {label}
        {hint && <span className="hint">{hint}</span>}
        {tooltip && <FieldTooltip text={tooltip} />}
      </label>
      <input
        type="text"
        className="text-input"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        style={mono ? { fontFamily: 'var(--font-mono)', fontSize: 12 } : {}}
        disabled={disabled}
      />
    </div>
  )
}

/**
 * NumberField — labelled number input
 */
export function NumberField({ label, value, onChange, min, max, step = 1, hint, disabled, tooltip }) {
  return (
    <div className="form-group">
      <label>
        {label}
        {hint && <span className="hint">{hint}</span>}
        {tooltip && <FieldTooltip text={tooltip} />}
      </label>
      <input
        type="number"
        className="num-input"
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
      />
    </div>
  )
}

/**
 * PathField — text input + folder/file browse button
 *
 * mode       — 'dirs' | 'files' | 'both'  (passed to DirBrowser)
 * extensions — comma-separated file extensions filter, e.g. '.pth,.pt'
 */
export function PathField({ label, value, onChange, placeholder, hint, mono, disabled, mode = 'dirs', extensions = '', tooltip }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="form-group">
      <label>
        {label}
        {hint && <span className="hint">{hint}</span>}
        {tooltip && <FieldTooltip text={tooltip} />}
      </label>
      <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
        <input
          type="text"
          className="text-input"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          style={{ flex: 1, ...(mono ? { fontFamily: 'var(--font-mono)', fontSize: 12 } : {}) }}
          disabled={disabled}
        />
        {!disabled && (
          <button
            type="button"
            title={mode === 'files' ? 'Browse for file' : 'Browse for folder'}
            onClick={() => setOpen(true)}
            style={{
              padding: '6px 10px', fontSize: 14, borderRadius: 'var(--radius-sm)',
              border: '1px solid var(--line-2)', background: 'var(--surface-2)',
              cursor: 'pointer', color: 'var(--ink-2)', flexShrink: 0,
            }}
          >
            {mode === 'files' ? '📄' : '📁'}
          </button>
        )}
      </div>
      {open && (
        <Suspense fallback={null}>
          <DirBrowser
            initial={value}
            mode={mode}
            extensions={extensions}
            onSelect={(p) => { onChange(p); setOpen(false) }}
            onClose={() => setOpen(false)}
          />
        </Suspense>
      )}
    </div>
  )
}

/**
 * Collapsible section wrapper
 */
export function CollapsibleSection({ title, children, defaultOpen = true }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="card" style={{ padding: 0, marginBottom: 16 }}>
      <button type="button" className="acc-head" onClick={() => setOpen((v) => !v)}>
        <span className="acc-title">{title}</span>
        <span style={{ color: 'var(--ink-3)', fontSize: 11 }}>{open ? '▲' : '▼'}</span>
      </button>
      {open && <div className="acc-body">{children}</div>}
    </div>
  )
}
