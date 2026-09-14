/**
 * src/components/ImageInfoCard.jsx
 * --------------------------------
 * Read-only "IMAGE INFORMATION" card shared by Inference and Preprocessing.
 * Renders Dimensions / Bands / Data type / Format / Geospatial / GSD / CRS
 * from the payload returned by GET /api/inference/image-info.
 */
export default function ImageInfoCard({ meta, title = 'IMAGE INFORMATION' }) {
  if (!meta) return null

  const rows = [
    ['Dimensions', `${meta.width} × ${meta.height}`],
    ['Bands', meta.bands],
    ['Data type', meta.dtype || '—'],
    ['Format', meta.format || '—'],
    ['Geospatial', meta.geospatial ? '✓' : '—'],
  ]
  if (meta.gsd != null) rows.push(['GSD', `${meta.gsd} m`])
  if (meta.crs) rows.push(['CRS', meta.crs])

  return (
    <div style={{
      background: 'var(--bg-2)', border: '1px solid var(--line-2)',
      borderRadius: 'var(--radius-sm)', padding: '10px 14px', marginTop: 10,
    }}>
      <div style={{
        fontSize: 10, fontWeight: 700, color: 'var(--ink-3)',
        textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 8,
      }}>
        {title}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '110px 1fr', rowGap: 4, fontSize: 12 }}>
        {rows.map(([k, v]) => (
          <div key={k} style={{ display: 'contents' }}>
            <div style={{ color: 'var(--ink-3)' }}>{k}</div>
            <div style={{ fontFamily: 'var(--font-mono)', color: 'var(--ink-1)', wordBreak: 'break-all' }}>{v}</div>
          </div>
        ))}
      </div>
    </div>
  )
}