/**
 * src/components/LiveOutputBadge.jsx
 * ----------------------------------
 * Small status pill used in the "Live Output" CollapsibleSection title.
 * Maps a status string to a colour + label.
 */
export default function LiveOutputBadge({ status }) {
  const styles = {
    running:   { bg: 'var(--cobalt-soft)',     fg: 'var(--cobalt-deep)', label: 'Running'   },
    paused:    { bg: 'rgba(230,160,40,0.15)',  fg: 'rgb(180,120,20)',    label: 'Paused'    },
    cancelled: { bg: 'var(--bg-2)',            fg: 'var(--ink-3)',       label: 'Cancelled' },
    completed: { bg: 'rgba(60,180,100,0.12)',  fg: 'var(--ok)',          label: 'Complete'  },
    failed:    { bg: 'rgba(220,70,70,0.10)',   fg: 'var(--bad)',         label: 'Failed'    },
  }
  const s = styles[status] || styles.running
  return (
    <span style={{
      fontSize: 10, fontWeight: 600, padding: '1px 6px',
      borderRadius: 4, background: s.bg, color: s.fg,
      textTransform: 'uppercase', letterSpacing: '0.05em',
    }}>{s.label}</span>
  )
}