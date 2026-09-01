import { useState } from 'react'
import { runEval } from '../api'
import LegalEvidenceView from './LegalEvidenceView'
import { matchSmartSuggestions } from '../suggestions'
import { addToHistory, getHistory } from '../history'

const CELLS = [
  { key: 'ollama_only', title: 'Ollama — raw (no persona, no retrieval)', cls: 'raw', badge: 'RAW MODEL' },
  { key: 'ollama_rag', title: 'Ollama — this app’s pipeline', cls: 'rag', badge: 'PERSONA + RAG' },
  { key: 'gemini_only', title: 'Gemini — raw (no persona, no retrieval)', cls: 'raw', badge: 'RAW MODEL' },
  { key: 'gemini_rag', title: 'Gemini — this app’s pipeline', cls: 'rag', badge: 'PERSONA + RAG' },
]

const GRAPH_ONLY_CELL = {
  key: 'ollama_graph_rag', title: 'Ollama — graph RAG only (vector search skipped)', cls: 'rag', badge: 'PERSONA + GRAPH ONLY',
}

// The two extra local models pulled for the Week 4 model-comparison
// exercise — Eval tab only, never offered on the Ask tab (see
// evaluation/metrics_report.json: starcoder2 essentially ignores the
// injected context, codellama has the highest hallucination rate of the
// four models tested).
const EXTRA_MODEL_CELLS = [
  { key: 'codellama_rag', title: 'CodeLlama 7B — this app’s pipeline', cls: 'rag', badge: 'PERSONA + RAG' },
  { key: 'starcoder2_rag', title: 'StarCoder2 3B — this app’s pipeline', cls: 'rag', badge: 'PERSONA + RAG' },
]

function Cell({ def, cell }) {
  return (
    <div className={`card eval-cell ${def.cls}`}>
      <h3>
        {def.title}
        <span className="mode-badge">{def.badge}</span>
      </h3>
      <div className="answer-text" style={{ fontSize: '0.88rem' }}>{cell.answer}</div>
      {cell.citations?.length > 0 && (
        <div className="citations">
          {cell.citations.map((c, i) => (
            <span className="citation-chip" key={i}>
              {c.act}{c.section ? ` — Sec ${c.section}` : ''}{c.page ? `, p.${c.page}` : ''}
            </span>
          ))}
        </div>
      )}
      <div className="meta-line" style={{ marginTop: '0.6rem', fontSize: '0.75rem', color: 'var(--muted)' }}>
        {cell.model} · {Math.round(cell.latency_ms)} ms
        {cell.grounding?.total_claims > 0 && (
          <span className={cell.grounding.unverified_claims === 0 ? 'grounding-ok' : 'grounding-warn'}>
            {' · '}{cell.grounding.unverified_claims === 0 ? '✓' : '⚠'} grounding {cell.grounding.verified_claims}/{cell.grounding.total_claims}
          </span>
        )}
      </div>
    </div>
  )
}

export default function EvalTab() {
  const [question, setQuestion] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)
  const [showEvidence, setShowEvidence] = useState(false)
  const [showSuggestions, setShowSuggestions] = useState(false)

  const suggestions = showSuggestions ? matchSmartSuggestions(question, getHistory()) : []

  async function run(overrideQuestion) {
    const q = (overrideQuestion ?? question).trim()
    if (!q || busy) return
    addToHistory(q)
    setShowSuggestions(false)
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      const data = await runEval(q)
      setResult(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  function pickSuggestion(text) {
    setQuestion(text)
    setShowSuggestions(false)
    run(text)
  }

  return (
    <div>
      <div className="card">
        <p style={{ marginTop: 0, fontSize: '0.82rem', color: 'var(--muted)' }}>
          Evaluator dashboard — one question, one retrieval pass shown in full below, then four
          generations: each model with no persona/no retrieval (its raw, unguided behavior) versus
          this app's actual pipeline (persona + hybrid RAG context).
        </p>
        <div style={{ position: 'relative' }}>
          <textarea
            className="question-box"
            placeholder="e.g. Can the officer ask for my RC?"
            value={question}
            onChange={(e) => {
              setQuestion(e.target.value)
              setShowSuggestions(true)
            }}
            onFocus={() => setShowSuggestions(true)}
            onBlur={() => setTimeout(() => setShowSuggestions(false), 150)}
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
          <button className="primary" onClick={() => run()} disabled={busy}>
            {busy ? 'Running…' : 'Run comparison'}
          </button>
        </div>
      </div>

      {error && <div className="error-box">{error}</div>}

      {busy && <p className="loading" style={{ marginTop: '1rem' }}>
        Running retrieval once, then 4 generations (2 Ollama + 2 Gemini) — this can take a while on
        local hardware…
      </p>}

      {result && (
        <>
          <div className="card" style={{ marginTop: '1.25rem' }}>
            <h4 style={{ margin: '0 0 0.3rem', fontSize: '0.78rem', color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              Retrieval — inner workings (shared by all 4 cells below)
            </h4>
            <div className="stat-row">
              <div className="stat">
                <span className="stat-label">Embed</span>
                <span className="stat-value">{Math.round(result.retrieval.timing.embed_ms)} ms</span>
              </div>
              <div className="stat">
                <span className="stat-label">Vector search</span>
                <span className="stat-value">{Math.round(result.retrieval.timing.vector_search_ms)} ms</span>
              </div>
              <div className="stat">
                <span className="stat-label">Graph lookup</span>
                <span className="stat-value">{Math.round(result.retrieval.timing.graph_ms)} ms</span>
              </div>
              <div className="stat">
                <span className="stat-label">Context chunks</span>
                <span className="stat-value">{result.retrieval.context.length}</span>
              </div>
            </div>

            <div style={{ marginTop: '0.9rem' }}>
              <span className="stat-label">Matched entities</span>{' '}
              {result.retrieval.matched_entities.length > 0
                ? result.retrieval.matched_entities.map((e, i) => (
                    <span className="citation-chip" key={i}>{e}</span>
                  ))
                : <span style={{ color: 'var(--muted)', fontSize: '0.85rem' }}>none</span>}
            </div>

            {result.retrieval.graph_relationships.length > 0 && (
              <div className="relationship-list">
                {result.retrieval.graph_relationships.slice(0, 8).map((r, i) => (
                  <div className="relationship-item" key={i}>
                    <b>{r.source_name}</b> —{r.relation}→ <b>{r.target_name}</b> ({r.evidence_count} section{r.evidence_count === 1 ? '' : 's'})
                  </div>
                ))}
                {result.retrieval.graph_relationships.length > 8 && (
                  <div className="relationship-item">+ {result.retrieval.graph_relationships.length - 8} more</div>
                )}
              </div>
            )}

            <button className="ghost" style={{ marginTop: '0.8rem' }} onClick={() => setShowEvidence((v) => !v)}>
              {showEvidence ? 'Hide' : 'Show'} ranked context chunks ({result.retrieval.context.length})
            </button>
            {showEvidence && <LegalEvidenceView context={result.retrieval.context} />}
          </div>

          <div className="grid-2x2">
            {CELLS.map((def) => (
              <Cell def={def} cell={result[def.key]} key={def.key} />
            ))}
          </div>

          <h4 style={{ margin: '1.25rem 0 0', fontSize: '0.78rem', color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            Isolating the graph path — same question, graph-only retrieval (no vector search at all)
          </h4>
          <div className="grid-2x2" style={{ gridTemplateColumns: '1fr' }}>
            <Cell def={GRAPH_ONLY_CELL} cell={result.ollama_graph_rag} />
          </div>

          <h4 style={{ margin: '1.25rem 0 0', fontSize: '0.78rem', color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            Other local models on this app's pipeline — Week 4 comparison (not offered on the Ask tab)
          </h4>
          <div className="grid-2x2">
            {EXTRA_MODEL_CELLS.map((def) => (
              <Cell def={def} cell={result[def.key]} key={def.key} />
            ))}
          </div>
        </>
      )}
    </div>
  )
}
