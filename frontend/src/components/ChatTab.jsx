import { useRef, useState } from 'react'
import { streamAsk } from '../api'
import PipelineTrace from './PipelineTrace'
import ResponseCard from './ResponseCard'
import { matchSmartSuggestions } from '../suggestions'
import { addToHistory, getHistory } from '../history'

export default function ChatTab() {
  const [question, setQuestion] = useState('')
  const [provider, setProvider] = useState('ollama')
  const [events, setEvents] = useState([])
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)
  const [showSuggestions, setShowSuggestions] = useState(false)
  const stopRef = useRef(null)

  const suggestions = showSuggestions ? matchSmartSuggestions(question, getHistory()) : []

  function ask(overrideQuestion) {
    const q = (overrideQuestion ?? question).trim()
    if (!q || busy) return

    addToHistory(q)
    setShowSuggestions(false)
    stopRef.current?.()
    setEvents([])
    setResult(null)
    setError(null)
    setBusy(true)

    stopRef.current = streamAsk(q, provider, {
      onEvent: (ev) => setEvents((prev) => [...prev, ev]),
      onDone: (data) => {
        setResult(data)
        setBusy(false)
      },
      onError: (message) => {
        setError(message)
        setBusy(false)
      },
    })
  }

  function pickSuggestion(text) {
    setQuestion(text)
    setShowSuggestions(false)
    ask(text)
  }

  return (
    <div>
      <div className="card">
        <div style={{ position: 'relative' }}>
          <textarea
            className="question-box"
            placeholder="e.g. Can the officer ask for my RC? Can my driving licence be seized on the spot?"
            value={question}
            onChange={(e) => {
              setQuestion(e.target.value)
              setShowSuggestions(true)
            }}
            onFocus={() => setShowSuggestions(true)}
            onBlur={() => setTimeout(() => setShowSuggestions(false), 150)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) ask()
            }}
          />
          {suggestions.length > 0 && (
            <div className="suggestion-list">
              {suggestions.map((text, i) => (
                <button
                  key={i}
                  className="suggestion-item"
                  onMouseDown={(e) => e.preventDefault()}
                  onClick={() => pickSuggestion(text)}
                >
                  {text}
                </button>
              ))}
            </div>
          )}
        </div>
        <div className="row">
          <div className="provider-toggle">
            {['ollama', 'gemini'].map((p) => (
              <label key={p} className={provider === p ? 'checked' : ''}>
                <input
                  type="radio"
                  name="provider"
                  value={p}
                  checked={provider === p}
                  onChange={() => setProvider(p)}
                />
                <span>{p === 'ollama' ? 'Ollama (Llama 3.1 8B)' : 'Gemini'}</span>
              </label>
            ))}
          </div>
          <button className="primary" onClick={() => ask()} disabled={busy}>
            {busy ? 'Asking…' : 'Ask'}
          </button>
        </div>
      </div>

      <PipelineTrace events={events} done={!!result || !!error} />

      {error && <div className="error-box">{error}</div>}
      {result && <ResponseCard result={result} />}

      <p style={{ marginTop: '1.5rem', fontSize: '0.75rem', color: 'var(--muted)' }}>
        This assistant explains legal rights and obligations under official Haryana/Indian motor
        vehicle law. It is not a substitute for a lawyer, and says so plainly when no official
        source supports an answer rather than guessing.
      </p>
    </div>
  )
}
