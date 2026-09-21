import { useEffect, useRef, useState } from 'react'
import { resetConversation, restoreConversation, streamAsk } from '../api'
import PipelineTrace from './PipelineTrace'
import ResponseCard from './ResponseCard'
import { matchSmartSuggestions } from '../suggestions'
import { addToHistory, getHistory } from '../history'
import {
  getConversationId,
  loadTranscript,
  saveTranscript,
  startNewConversation,
  toModelMessages,
} from '../conversation'

const OPENERS = [
  'The officer says my tint is illegal — is it?',
  'Can they seize my licence on the spot?',
  'I was riding without a helmet. How bad is this?',
  'What documents must I actually carry?',
]

export default function ChatTab() {
  const [messages, setMessages] = useState(() => loadTranscript())
  const [conversationId, setConversationId] = useState(() => getConversationId())
  const [question, setQuestion] = useState('')
  const [provider, setProvider] = useState('ollama')
  const [events, setEvents] = useState([])
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)
  const [showSuggestions, setShowSuggestions] = useState(false)
  const stopRef = useRef(null)
  const bottomRef = useRef(null)
  const inputRef = useRef(null)

  const suggestions = showSuggestions ? matchSmartSuggestions(question, getHistory()) : []

  // The server's transcript is in-memory and ours is in localStorage, so after
  // a reload or a service restart only ours survives. Push it back once on
  // mount — otherwise the user sees their whole chat on screen while the model
  // answers the next question with no memory of it.
  useEffect(() => {
    const stored = loadTranscript()
    if (stored.length) restoreConversation(conversationId, toModelMessages(stored))
    // Deliberately mount-only: re-pushing after every turn would undo the
    // server's own record of the turn it just handled.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => saveTranscript(messages), [messages])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages, events, busy])

  // Stop an in-flight stream if the user navigates away mid-answer, so a
  // CPU-only Ollama turn isn't left streaming into a dead component.
  useEffect(() => () => stopRef.current?.(), [])

  function ask(overrideQuestion) {
    const q = (overrideQuestion ?? question).trim()
    if (!q || busy) return

    addToHistory(q)
    setShowSuggestions(false)
    stopRef.current?.()
    setEvents([])
    setError(null)
    setBusy(true)
    setQuestion('')
    setMessages((prev) => [...prev, { role: 'user', content: q }])

    stopRef.current = streamAsk(q, provider, {
      conversationId,
      onEvent: (ev) => setEvents((prev) => [...prev, ev]),
      onDone: (data) => {
        // The whole payload is kept, not just the text: ResponseCard renders
        // this turn's own citations, grounding verdict and retrieved chunks,
        // so every answer stays auditable after later turns are added.
        setMessages((prev) => [...prev, { role: 'assistant', content: data.answer, result: data }])
        setBusy(false)
        inputRef.current?.focus()
      },
      onError: (message) => {
        setError(message)
        setBusy(false)
        // Drop the user turn that never got an answer — leaving it would make
        // the next follow-up resolve against a question the model never saw.
        setMessages((prev) => (prev[prev.length - 1]?.role === 'user' ? prev.slice(0, -1) : prev))
      },
    })
  }

  function newChat() {
    stopRef.current?.()
    resetConversation(conversationId)
    setConversationId(startNewConversation())
    setMessages([])
    setEvents([])
    setError(null)
    setBusy(false)
    setQuestion('')
    inputRef.current?.focus()
  }

  const isEmpty = messages.length === 0 && !busy

  return (
    <div className="chat">
      <div className="chat-bar">
        <span className="chat-bar-label">
          {messages.length > 0
            ? `${messages.filter((m) => m.role === 'user').length} question(s) in this conversation`
            : 'New conversation'}
        </span>
        <button className="ghost" onClick={newChat} disabled={isEmpty}>
          New chat
        </button>
      </div>

      <div className="chat-thread">
        {isEmpty && (
          <div className="chat-empty">
            <h3>What's happening at the stop?</h3>
            <p>
              Tell me the situation and I'll tell you where you actually stand — including when the
              law isn't on your side. Ask follow-ups; I'll remember what you've told me.
            </p>
            <div className="chat-openers">
              {OPENERS.map((text) => (
                <button key={text} className="chat-opener" onClick={() => ask(text)}>
                  {text}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m, i) =>
          m.role === 'user' ? (
            <div className="msg-user" key={i}>
              <div className="msg-user-bubble">{m.content}</div>
            </div>
          ) : (
            <div className="msg-assistant" key={i}>
              <ResponseCard result={m.result} />
            </div>
          )
        )}

        {busy && (
          <div className="msg-assistant">
            <PipelineTrace events={events} />
          </div>
        )}

        {error && <div className="error-box">{error}</div>}

        <div ref={bottomRef} />
      </div>

      <div className="chat-composer">
        <div style={{ position: 'relative' }}>
          <textarea
            ref={inputRef}
            className="question-box"
            placeholder={
              messages.length
                ? 'Ask a follow-up — e.g. "what if I refuse?"'
                : 'e.g. Can the officer ask for my RC? Can my driving licence be seized on the spot?'
            }
            value={question}
            onChange={(e) => {
              setQuestion(e.target.value)
              setShowSuggestions(true)
            }}
            onFocus={() => setShowSuggestions(true)}
            // Keep the delay. The suggestion list is IN-FLOW (see App.css), so
            // it no longer covers the Ask button — but that also means closing
            // it collapses its height and shifts the button upward. Closing on
            // blur (i.e. on the button's own mousedown) moved the button out
            // from under the cursor before mouseup, so the click landed on
            // nothing. The delay keeps the layout still until the click lands.
            onBlur={() => setTimeout(() => setShowSuggestions(false), 150)}
            onKeyDown={(e) => {
              // Enter sends, Shift+Enter newlines — the convention every chat
              // UI uses. Ctrl/Cmd+Enter still works, so the old muscle memory
              // from the single-question version isn't broken.
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                ask()
              }
              if (e.key === 'Escape') setShowSuggestions(false)
            }}
          />
          {suggestions.length > 0 && (
            <div className="suggestion-list">
              {suggestions.map((text, i) => (
                <button
                  key={i}
                  className="suggestion-item"
                  onMouseDown={(e) => e.preventDefault()}
                  onClick={() => {
                    setQuestion(text)
                    setShowSuggestions(false)
                    ask(text)
                  }}
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
            {busy ? 'Thinking…' : 'Send'}
          </button>
        </div>
        <p className="chat-disclaimer">
          This assistant explains legal rights and obligations under official Haryana/Indian motor
          vehicle law. It is not a substitute for a lawyer, and says so plainly when no official
          source supports an answer rather than guessing.
        </p>
      </div>
    </div>
  )
}
