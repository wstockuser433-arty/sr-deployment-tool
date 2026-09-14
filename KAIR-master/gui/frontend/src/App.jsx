import { useState, useEffect, useRef } from 'react'
import { NavLink, Navigate, useLocation } from 'react-router-dom'
import Training from './pages/Training'
import Inference from './pages/Inference'
import Preprocessing from './pages/Preprocessing'
import { checkHealth } from './api/client'
import { JobProvider } from './context/JobContext'
import suparcoLogo from '../assets/suparco logo.jpg'

const SENSOR_PROFILES = [
  {
    key: 'pan', label: 'Panchromatic', sensor: 'General / Pleiades Pan',
    bands: [{ name: 'Pan', range: '480–830 nm', desc: 'Broadband panchromatic' }],
  },
  {
    key: 'rgb', label: 'RGB (3-band)', sensor: 'Standard optical',
    bands: [
      { name: 'R', range: '620–700 nm', desc: 'Red' },
      { name: 'G', range: '520–590 nm', desc: 'Green' },
      { name: 'B', range: '450–510 nm', desc: 'Blue' },
    ],
  },
  {
    key: 'ms4', label: 'Multispectral 4-band', sensor: 'Pleiades · Planet SuperDove',
    bands: [
      { name: 'B', range: '450–510 nm', desc: 'Blue' },
      { name: 'G', range: '520–590 nm', desc: 'Green' },
      { name: 'R', range: '620–700 nm', desc: 'Red' },
      { name: 'NIR', range: '750–950 nm', desc: 'Near Infrared' },
    ],
  },
  {
    key: 'ms8', label: 'Multispectral 8-band', sensor: 'WorldView-2/3',
    bands: [
      { name: 'C',    range: '400–450 nm', desc: 'Coastal' },
      { name: 'B',    range: '450–510 nm', desc: 'Blue' },
      { name: 'G',    range: '510–580 nm', desc: 'Green' },
      { name: 'Y',    range: '585–625 nm', desc: 'Yellow' },
      { name: 'R',    range: '630–690 nm', desc: 'Red' },
      { name: 'RE',   range: '705–745 nm', desc: 'Red Edge' },
      { name: 'NIR1', range: '770–895 nm', desc: 'Near-IR 1' },
      { name: 'NIR2', range: '860–1040 nm', desc: 'Near-IR 2' },
    ],
  },
  {
    key: 's2', label: 'Sentinel-2 (13-band)', sensor: 'Sentinel-2A/B MSI',
    bands: [
      { name: 'B1',  range: '443 nm',  desc: 'Coastal aerosol' },
      { name: 'B2',  range: '490 nm',  desc: 'Blue' },
      { name: 'B3',  range: '560 nm',  desc: 'Green' },
      { name: 'B4',  range: '665 nm',  desc: 'Red' },
      { name: 'B5',  range: '705 nm',  desc: 'Vegetation red edge' },
      { name: 'B6',  range: '740 nm',  desc: 'Vegetation red edge' },
      { name: 'B7',  range: '783 nm',  desc: 'Vegetation red edge' },
      { name: 'B8',  range: '842 nm',  desc: 'NIR' },
      { name: 'B8A', range: '865 nm',  desc: 'Narrow NIR' },
      { name: 'B9',  range: '945 nm',  desc: 'Water vapour' },
      { name: 'B10', range: '1375 nm', desc: 'SWIR — cirrus' },
      { name: 'B11', range: '1610 nm', desc: 'SWIR' },
      { name: 'B12', range: '2190 nm', desc: 'SWIR' },
    ],
  },
]

function ClassesModal({ onClose }) {
  const [selected, setSelected] = useState(SENSOR_PROFILES[0].key)
  const profile = SENSOR_PROFILES.find(p => p.key === selected)

  useEffect(() => {
    const handler = (e) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose])

  return (
    <div onClick={onClose} style={{
      position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.5)', zIndex: 9998,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
    }}>
      <div onClick={e => e.stopPropagation()} style={{
        background: 'var(--surface)', borderRadius: 'var(--radius)', padding: 28,
        width: 580, maxWidth: '95vw', maxHeight: '85vh', overflow: 'auto',
        boxShadow: 'var(--shadow-lg)', border: '1px solid var(--line)',
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
          <div>
            <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--ink)' }}>Sensor Band Profiles</div>
            <div style={{ fontSize: 12, color: 'var(--ink-3)', marginTop: 2 }}>
              Supported satellite sensor configurations and their spectral bands
            </div>
          </div>
          <button onClick={onClose} style={{
            background: 'none', border: 'none', cursor: 'pointer', fontSize: 20,
            color: 'var(--ink-3)', lineHeight: 1, padding: '4px 6px',
          }}>✕</button>
        </div>

        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 20 }}>
          {SENSOR_PROFILES.map(p => (
            <button key={p.key} type="button"
              onClick={() => setSelected(p.key)}
              style={{
                padding: '5px 12px', fontSize: 12, borderRadius: 'var(--radius-sm)',
                border: `1px solid ${selected === p.key ? 'var(--cobalt-deep)' : 'var(--line-2)'}`,
                background: selected === p.key ? 'var(--cobalt-soft)' : 'var(--surface-2)',
                color: selected === p.key ? 'var(--cobalt-deep)' : 'var(--ink-2)',
                fontWeight: selected === p.key ? 600 : 400, cursor: 'pointer',
              }}>
              {p.label}
            </button>
          ))}
        </div>

        {profile && (
          <div>
            <div style={{ fontSize: 12, color: 'var(--ink-3)', marginBottom: 14 }}>
              Sensor: <span style={{ fontWeight: 600, color: 'var(--ink-2)' }}>{profile.sensor}</span>
              &nbsp;·&nbsp; {profile.bands.length} band{profile.bands.length !== 1 ? 's' : ''}
            </div>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: '2px solid var(--line-2)' }}>
                  <th style={{ textAlign: 'left', padding: '6px 10px', color: 'var(--ink-2)', fontWeight: 600 }}>#</th>
                  <th style={{ textAlign: 'left', padding: '6px 10px', color: 'var(--ink-2)', fontWeight: 600 }}>Band</th>
                  <th style={{ textAlign: 'left', padding: '6px 10px', color: 'var(--ink-2)', fontWeight: 600 }}>Range</th>
                  <th style={{ textAlign: 'left', padding: '6px 10px', color: 'var(--ink-2)', fontWeight: 600 }}>Description</th>
                </tr>
              </thead>
              <tbody>
                {profile.bands.map((b, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid var(--line-2)' }}>
                    <td style={{ padding: '7px 10px', color: 'var(--ink-3)', fontFamily: 'monospace' }}>{i + 1}</td>
                    <td style={{ padding: '7px 10px', fontWeight: 600, color: 'var(--cobalt-deep)', fontFamily: 'monospace' }}>{b.name}</td>
                    <td style={{ padding: '7px 10px', color: 'var(--ink-2)', fontFamily: 'monospace' }}>{b.range}</td>
                    <td style={{ padding: '7px 10px', color: 'var(--ink-2)' }}>{b.desc}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div style={{ marginTop: 14, fontSize: 11, color: 'var(--ink-3)', lineHeight: 1.5 }}>
              Use 1-indexed band numbers in the inference band selection. Example: RGB composite
              for Pleiades 4-band → LR bands [3, 2, 1] (R, G, B order).
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

const NAV_ITEMS = [
  {
    to: '/preprocessing',
    label: 'Preprocessing',
    icon: (
      <svg width="15" height="15" viewBox="0 0 15 15" fill="none">
        <rect x="2.5" y="2.5" width="4" height="4" stroke="currentColor" strokeLinejoin="round" />
        <rect x="8.5" y="2.5" width="4" height="4" stroke="currentColor" strokeLinejoin="round" />
        <rect x="8.5" y="8.5" width="4" height="4" stroke="currentColor" strokeLinejoin="round" />
        <rect x="2.5" y="8.5" width="4" height="4" stroke="currentColor" strokeLinejoin="round" />
      </svg>
    ),
  },
  {
    to: '/training',
    label: 'Training',
    icon: (
      <svg width="15" height="15" viewBox="0 0 15 15" fill="none">
        <path d="M2.5 12.5L7.5 4.5L12.5 12.5" stroke="currentColor" strokeLinejoin="round" />
        <path d="M4 10H11" stroke="currentColor" strokeLinejoin="round" />
      </svg>
    ),
  },
  {
    to: '/inference',
    label: 'Inference',
    icon: (
      <svg width="15" height="15" viewBox="0 0 15 15" fill="none">
        <circle cx="6.5" cy="6.5" r="4" stroke="currentColor" />
        <path d="M12.5 12.5L9.5 9.5" stroke="currentColor" strokeLinecap="round" />
      </svg>
    ),
  },
]

export default function App() {
  const location = useLocation()
  const [connected, setConnected] = useState(false)
  const [dropdownOpen, setDropdownOpen] = useState(false)
  const [lastChecked, setLastChecked] = useState('')
  const [isChecking, setIsChecking] = useState(false)
  const [classesOpen, setClassesOpen] = useState(false)
  const dropdownRef = useRef(null)

  const check = (showChecking = false) => {
    if (showChecking) setIsChecking(true)
    return checkHealth()
      .then(() => {
        setConnected(true)
        setLastChecked(new Date().toLocaleTimeString())
      })
      .catch(() => {
        setConnected(false)
        setLastChecked(new Date().toLocaleTimeString())
      })
      .finally(() => {
        if (showChecking) setIsChecking(false)
      })
  }

  useEffect(() => {
    check(true)
    const interval = setInterval(() => check(false), 4000)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    function handleClickOutside(event) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setDropdownOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  return (
    <JobProvider>
    <div className="app">
      {classesOpen && <ClassesModal onClose={() => setClassesOpen(false)} />}

      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sb-brand">
          <img src={suparcoLogo} alt="SUPARCO" className="sb-logo" />
          <h1>Super-Resolution</h1>
        </div>
        <nav className="sb-nav">
          <div className="sb-sec">Workflows</div>
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) => `sb-item ${isActive ? 'active' : ''}`}
            >
              {item.icon}
              {item.label}
            </NavLink>
          ))}
          <div className="sb-sec" style={{ marginTop: 12 }}>Reference</div>
          <button
            type="button"
            className="sb-item"
            onClick={() => setClassesOpen(true)}
            style={{ background: 'none', border: 'none', width: '100%', textAlign: 'left', cursor: 'pointer' }}
          >
            <svg width="15" height="15" viewBox="0 0 15 15" fill="none">
              <rect x="1.5" y="2.5" width="12" height="1" fill="currentColor" rx="0.5" />
              <rect x="1.5" y="5.5" width="8" height="1" fill="currentColor" rx="0.5" />
              <rect x="1.5" y="8.5" width="10" height="1" fill="currentColor" rx="0.5" />
              <rect x="1.5" y="11.5" width="6" height="1" fill="currentColor" rx="0.5" />
            </svg>
            Band Classes
          </button>
        </nav>
        <div className="sb-footer" ref={dropdownRef}>
          {dropdownOpen && (
            <div className="sb-status-dropdown" role="menu">
              <div className="sb-status-detail-item">
                <span>Endpoint</span>
                <span className="val">/api/health</span>
              </div>
              <div className="sb-status-detail-item">
                <span>Status</span>
                <span className="val" style={{ color: connected ? 'var(--ok)' : 'var(--bad)', fontWeight: 600 }}>
                  {connected ? 'Healthy' : 'Unhealthy'}
                </span>
              </div>
              <div className="sb-status-detail-item">
                <span>Last Checked</span>
                <span className="val">{lastChecked || 'Checking...'}</span>
              </div>
              <div className="sb-status-divider" />
              <button
                type="button"
                className="sb-status-refresh-btn"
                onClick={() => check(true)}
                disabled={isChecking}
              >
                <svg
                  className={isChecking ? 'spin' : ''}
                  width="13"
                  height="13"
                  viewBox="0 0 16 16"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M2.5 2v4h4" />
                  <path d="M13.5 14v-4h-4" />
                  <path d="M13.5 6A7 7 0 0 0 4.5 3.5L2.5 6" />
                  <path d="M2.5 10a7 7 0 0 0 9 2.5l2-2.5" />
                </svg>
                {isChecking ? 'Checking...' : 'Check Status'}
              </button>
            </div>
          )}
          <button
            type="button"
            className={`sb-status-trigger ${dropdownOpen ? 'open' : ''}`}
            onClick={() => setDropdownOpen(!dropdownOpen)}
            aria-haspopup="true"
            aria-expanded={dropdownOpen}
          >
            <div className="sb-status-info" style={{ color: 'var(--ink)' }}>
              API Status
            </div>
            <svg
              className="chevron"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <polyline points="6 9 12 15 18 9" />
            </svg>
          </button>
        </div>
      </aside>

      {/* Main content — all pages stay mounted; only visibility changes */}
      <main className="main">
        {location.pathname === '/' && <Navigate to="/training" replace />}
        <div style={{ display: location.pathname === '/training' ? 'block' : 'none' }}>
          <Training />
        </div>
        <div style={{ display: location.pathname === '/inference' ? 'block' : 'none' }}>
          <Inference />
        </div>
        <div style={{ display: location.pathname === '/preprocessing' ? 'block' : 'none' }}>
          <Preprocessing />
        </div>
      </main>
    </div>
    </JobProvider>
  )
}
