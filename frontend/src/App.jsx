import { useState } from 'react'
import './App.css'
import ArchitectureTab from './components/ArchitectureTab'
import ChatTab from './components/ChatTab'
import EvalTab from './components/EvalTab'
import RightsLibraryTab from './components/RightsLibraryTab'

const TABS = [
  { key: 'chat', label: 'Chat' },
  { key: 'library', label: 'Rights Library' },
  { key: 'eval', label: 'Eval' },
  { key: 'architecture', label: 'How It Works' },
]

export default function App() {
  const [tab, setTab] = useState('chat')

  return (
    <div className="app">
      <header className="top">
        <h1>🚦 Haryana Traffic Legal Assistant</h1>
        <p className="tagline">
          Know your rights during a roadside stop — answers cited from official Haryana &amp;
          Indian motor vehicle law.
        </p>
        <nav className="tabs">
          {TABS.map((t) => (
            <button key={t.key} className={tab === t.key ? 'active' : ''} onClick={() => setTab(t.key)}>
              {t.label}
            </button>
          ))}
        </nav>
      </header>

      <main className="content" style={tab === 'eval' || tab === 'architecture' ? { maxWidth: '1100px' } : undefined}>
        {tab === 'chat' && <ChatTab />}
        {tab === 'library' && <RightsLibraryTab />}
        {tab === 'eval' && <EvalTab />}
        {tab === 'architecture' && <ArchitectureTab />}
      </main>
    </div>
  )
}
