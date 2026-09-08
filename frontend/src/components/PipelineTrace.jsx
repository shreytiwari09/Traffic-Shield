// Visualises the REAL step-by-step progress of the current request as SSE
// events arrive from Orchestration Service. Every node, timing and count
// below is read straight off an event the server actually emitted for a real
// inter-service call — nothing here is a scripted animation or a placeholder
// duration. See services/orchestration_service/routes.py (/v1/ask/stream).
//
// Four layers, in order of how much they explain:
//   1. Flow diagram      — which service handled which step, and where the
//                          request is right now.
//   2. Timing breakdown  — proportional bar showing where the time went.
//   3. Outcome stats     — what retrieval actually found, and whether the
//                          grounding checker verified the answer's claims.
//   4. Raw event log     — the unstyled server messages, collapsed by default,
//                          so the pretty view can always be checked against
//                          what the backend literally said.
import { useMemo, useState } from 'react'

// Each stage names the service that does the work and the port it listens on,
// so the diagram doubles as evidence that these are five separate processes.
// `done` is the event whose arrival proves the stage finished.
const STAGES = [
  { key: 'app', icon: '🌐', label: 'Application', port: ':8000', sub: 'Single front door', done: 'start' },
  { key: 'orch', icon: '🧭', label: 'Orchestration', port: ':8001', sub: 'Sequences the request', done: 'retrieval_start' },
  { key: 'embed', icon: '🔢', label: 'LLM · embed', port: ':8003', sub: 'nomic-embed-text', done: 'embed_done' },
  { key: 'vector', icon: '📐', label: 'Data · vector', port: ':8004', sub: 'Chroma cosine search', done: 'vector_search_done' },
  { key: 'graph', icon: '🕸️', label: 'Retrieval · graph', port: ':8002', sub: 'Entity + relationships', done: 'graph_done' },
  { key: 'fuse', icon: '⚖️', label: 'Retrieval · fuse', port: ':8002', sub: 'One ranked context', done: 'retrieval_done' },
  { key: 'gen', icon: '✍️', label: 'LLM · generate', port: ':8003', sub: 'Ollama / Gemini', done: 'grounding_check', start: 'generation_start' },
  // Label kept to one word so every card in the row stays the same height.
  { key: 'ground', icon: '🛡️', label: 'Grounding', port: ':8001', sub: 'Verify every claim', done: 'grounding_check' },
]

// Colours match the timing-bar segments so the legend needs no repeating.
const SEGMENTS = [
  { key: 'embed', label: 'Embed', cls: 'seg-embed' },
  { key: 'vector', label: 'Vector search', cls: 'seg-vector' },
  { key: 'graph', label: 'Graph lookup', cls: 'seg-graph' },
  { key: 'gen', label: 'Generation', cls: 'seg-gen' },
  { key: 'other', label: 'Fusion & overhead', cls: 'seg-other' },
]

function ms(value) {
  if (typeof value !== 'number') return null
  return value >= 1000 ? `${(value / 1000).toFixed(1)} s` : `${Math.round(value)} ms`
}

// `done` is intentionally not a prop: completion is derivable from the events
// themselves (a `done` or `error` step), so there is no second source of truth
// to keep in sync with the stream.
export default function PipelineTrace({ events }) {
  const [showLog, setShowLog] = useState(false)

  const seen = useMemo(() => {
    const map = {}
    for (const ev of events) map[ev.step] = ev
    return map
  }, [events])

  const errored = !!seen.error
  const finished = !!seen.done

  // Stage status. A stage is 'done' the moment its proving event arrives;
  // 'active' once the pipeline has reached it but not past it. On error, the
  // first unfinished stage is what failed.
  const firstUnfinished = STAGES.findIndex((s) => !seen[s.done])
  function statusOf(stage, index) {
    if (seen[stage.done]) return 'done'
    if (errored) return index === firstUnfinished ? 'error' : 'pending'
    if (index === firstUnfinished && events.length > 0) return 'active'
    return 'pending'
  }

  // Per-stage timing. Note retrieval_done.elapsed_ms is CUMULATIVE from the
  // start of the request (not that step's own duration), so it is deliberately
  // not shown as a stage duration — only embed/vector/graph report their own
  // measured spans, and generation's comes from the final `done` event.
  const timings = {
    embed: seen.embed_done?.elapsed_ms,
    vector: seen.vector_search_done?.elapsed_ms,
    graph: seen.graph_done?.elapsed_ms,
    gen: seen.done?.elapsed_ms,
  }

  const totalMs = seen.done?.total_elapsed_ms
  const breakdown = useMemo(() => {
    if (typeof totalMs !== 'number') return null
    const parts = {
      embed: timings.embed || 0,
      vector: timings.vector || 0,
      graph: timings.graph || 0,
      gen: timings.gen || 0,
    }
    const accounted = parts.embed + parts.vector + parts.graph + parts.gen
    // Guard: retrieval scores graph candidates concurrently, so the measured
    // spans can sum past the wall-clock total. Never render a negative slice.
    parts.other = Math.max(0, totalMs - accounted)
    const scale = Math.max(totalMs, accounted)
    return { parts, scale }
  }, [totalMs, timings.embed, timings.vector, timings.graph, timings.gen])

  // What retrieval actually handed the model, split by which path found it.
  const context = seen.done?.context || seen.retrieval_done?.context || []
  const nVector = context.filter((c) => c.source === 'vector').length
  const nGraph = context.filter((c) => c.source === 'graph').length
  const entities = seen.done?.matched_entities || seen.graph_done?.matched_entities || []
  const grounding = seen.done?.grounding

  if (events.length === 0) return null

  return (
    <section className="pipeline" aria-label="Live pipeline">
      <header className="pipeline-head">
        <div>
          <h4>Live retrieval &amp; generation pipeline</h4>
          <p className="pipeline-sub">
            Real Server-Sent Events from Orchestration Service — one event per actual
            inter-service call, not an animation.
          </p>
        </div>
        {typeof totalMs === 'number' && (
          <div className="pipeline-total">
            <span className="pipeline-total-value">{ms(totalMs)}</span>
            <span className="pipeline-total-label">total</span>
          </div>
        )}
      </header>

      {/* 1 — flow diagram */}
      <ol className="flow">
        {STAGES.map((stage, i) => {
          const status = statusOf(stage, i)
          return (
            <li key={stage.key} className={`flow-node ${status}`}>
              <div className="flow-icon">
                {status === 'done' && <span className="flow-tick">✓</span>}
                {status === 'active' && <span className="flow-spin">◐</span>}
                {status === 'error' && <span className="flow-cross">✕</span>}
                {status === 'pending' && <span className="flow-dot">•</span>}
              </div>
              <div className="flow-body">
                <div className="flow-label">
                  <span className="flow-emoji">{stage.icon}</span>
                  {stage.label}
                </div>
                <div className="flow-sub">{stage.sub}</div>
                <div className="flow-meta">
                  <span className="flow-port">{stage.port}</span>
                  {ms(timings[stage.key]) && <span className="flow-ms">{ms(timings[stage.key])}</span>}
                </div>
              </div>
            </li>
          )
        })}
      </ol>

      {/* 2 — where the time actually went */}
      {breakdown && (
        <div className="timing">
          <div className="timing-bar">
            {SEGMENTS.map((seg) => {
              const value = breakdown.parts[seg.key]
              if (!value) return null
              const pct = (value / breakdown.scale) * 100
              return (
                <div
                  key={seg.key}
                  className={`timing-seg ${seg.cls}`}
                  style={{ width: `${pct}%` }}
                  title={`${seg.label}: ${ms(value)}`}
                />
              )
            })}
          </div>
          <div className="timing-legend">
            {SEGMENTS.map((seg) => {
              const value = breakdown.parts[seg.key]
              if (!value) return null
              return (
                <span key={seg.key} className="timing-key">
                  <i className={`swatch ${seg.cls}`} />
                  {seg.label} <b>{ms(value)}</b>
                </span>
              )
            })}
          </div>
        </div>
      )}

      {/* 3 — what the pipeline actually produced */}
      {(context.length > 0 || grounding) && (
        <div className="pipeline-stats">
          <div className="pstat">
            <span className="pstat-value">{context.length}</span>
            <span className="pstat-label">context sections</span>
          </div>
          <div className="pstat">
            <span className="pstat-value">
              {nVector} <em>/</em> {nGraph}
            </span>
            <span className="pstat-label">vector / graph</span>
          </div>
          <div className="pstat">
            <span className="pstat-value">{entities.length}</span>
            <span className="pstat-label">graph entities</span>
          </div>
          {grounding && (
            <div className={`pstat ${grounding.unverified_claims > 0 ? 'bad' : 'good'}`}>
              <span className="pstat-value">
                {grounding.verified_claims}<em>/</em>{grounding.total_claims}
              </span>
              <span className="pstat-label">claims verified</span>
            </div>
          )}
        </div>
      )}

      {entities.length > 0 && (
        <div className="entity-row">
          {entities.map((e) => (
            <span className="entity-chip" key={e}>{e}</span>
          ))}
        </div>
      )}

      {/* 4 — the unstyled truth, for checking the view above against */}
      <button className="log-toggle" onClick={() => setShowLog((v) => !v)}>
        {showLog ? '▾' : '▸'} {showLog ? 'Hide' : 'Show'} raw event log ({events.length})
      </button>
      {showLog && (
        <div className="log">
          {events.map((ev, i) => (
            <div className={`log-line ${ev.step === 'error' ? 'err' : ''}`} key={i}>
              <span className="log-step">{ev.step}</span>
              <span className="log-msg">{ev.message}</span>
              {typeof ev.elapsed_ms === 'number' && (
                <span className="log-ms">{Math.round(ev.elapsed_ms)} ms</span>
              )}
            </div>
          ))}
        </div>
      )}

      {!finished && !errored && <div className="pipeline-working">Working…</div>}
    </section>
  )
}
