import { useEffect, useMemo, useState } from 'react'
import { fetchEvalReport } from '../api'
import '../metrics.css'

// RAG Evaluation dashboard — reads the OFFLINE harness's metrics_report.json.
//
// Three sections, top to bottom: executive KPIs, model comparison, trace
// inspector. Built on this app's existing CSS custom properties rather than a
// utility framework: the project has an established dark design system
// (index.css) and hand-rolled SVG visuals (PipelineTrace, the timing bars), so
// introducing Tailwind would have meant two parallel styling systems in one
// bundle. The charts are hand-rolled SVG for the same reason — a charting
// library would have roughly tripled the bundle for two charts.
//
// Every number rendered here comes from the report file. Nothing is mocked:
// a metric the harness could not measure renders as an explicit "not measured"
// rather than a zero, which is the entire point of the profiling rewrite.

const PALETTE = ['#5b9bff', '#35d39a', '#a481ff', '#ecc04a', '#f0664a']

function pct(v) {
  return v === null || v === undefined ? null : `${(v * 100).toFixed(1)}%`
}

function scoreTone(v) {
  if (v === null || v === undefined) return 'none'
  if (v >= 0.85) return 'high'
  if (v >= 0.6) return 'medium'
  return 'low'
}

/** Sparkline over per-question scores — shape, not precision. */
function Sparkline({ values, color }) {
  const clean = (values || []).filter((v) => v !== null && v !== undefined)
  if (clean.length < 2) return null
  const w = 90
  const h = 26
  const max = Math.max(...clean, 1)
  const min = Math.min(...clean, 0)
  const span = max - min || 1
  const pts = clean
    .map((v, i) => `${(i / (clean.length - 1)) * w},${h - ((v - min) / span) * h}`)
    .join(' ')
  return (
    <svg className="spark" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" aria-hidden="true">
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.6" vectorEffect="non-scaling-stroke" />
    </svg>
  )
}

function KpiCard({ label, value, tone, hint, spark, sparkColor, footnote }) {
  return (
    <div className={`kpi kpi-${tone}`}>
      <div className="kpi-head">
        <span className="kpi-label">{label}</span>
        <span className="kpi-info" title={hint}>?</span>
      </div>
      <div className="kpi-value">{value ?? <em className="kpi-na">not measured</em>}</div>
      {spark?.length > 1 && <Sparkline values={spark} color={sparkColor} />}
      {footnote && <div className="kpi-foot">{footnote}</div>}
    </div>
  )
}

/** Grouped bar chart: models x RAGAS metrics. Hand-rolled SVG. */
function ModelComparisonChart({ models }) {
  const metrics = [
    { key: 'faithfulness', label: 'Faithfulness' },
    { key: 'answer_relevance', label: 'Relevance' },
    { key: 'test_pass_rate', label: 'Guardrail pass' },
    { key: 'grounded', label: 'Grounded (1−halluc)' },
  ]
  const names = Object.keys(models)
  if (!names.length) return null

  const value = (name, key) => {
    const m = models[name]
    if (key === 'grounded') {
      const h = m.hallucination_rate_grounding
      return h === null || h === undefined ? null : 1 - h
    }
    return m[key]
  }

  const W = 720
  const H = 240
  const padL = 44
  const padB = 46
  const padT = 12
  const groupW = (W - padL) / metrics.length
  const barW = Math.min(26, (groupW - 18) / names.length)

  return (
    <div className="chart-wrap">
      <svg viewBox={`0 0 ${W} ${H}`} className="chart" role="img" aria-label="Model comparison across RAGAS metrics">
        {[0, 0.25, 0.5, 0.75, 1].map((g) => {
          const y = padT + (1 - g) * (H - padT - padB)
          return (
            <g key={g}>
              <line x1={padL} y1={y} x2={W} y2={y} stroke="var(--border)" strokeWidth="1" />
              <text x={padL - 8} y={y + 4} textAnchor="end" className="chart-axis">{g * 100}</text>
            </g>
          )
        })}
        {metrics.map((metric, mi) => {
          const gx = padL + mi * groupW
          return (
            <g key={metric.key}>
              {names.map((name, ni) => {
                const v = value(name, metric.key)
                const plotH = H - padT - padB
                const barH = v === null || v === undefined ? 0 : v * plotH
                const x = gx + (groupW - barW * names.length) / 2 + ni * barW
                return v === null || v === undefined ? (
                  <text key={name} x={x + barW / 2} y={H - padB - 4} textAnchor="middle" className="chart-na">n/a</text>
                ) : (
                  <rect
                    key={name}
                    x={x}
                    y={padT + plotH - barH}
                    width={barW - 3}
                    height={barH}
                    rx="2"
                    fill={PALETTE[ni % PALETTE.length]}
                  >
                    <title>{`${name} — ${metric.label}: ${(v * 100).toFixed(1)}%`}</title>
                  </rect>
                )
              })}
              <text x={gx + groupW / 2} y={H - padB + 18} textAnchor="middle" className="chart-label">
                {metric.label}
              </text>
            </g>
          )
        })}
      </svg>
      <div className="chart-legend">
        {names.map((name, i) => (
          <span key={name} className="legend-item">
            <i className="legend-swatch" style={{ background: PALETTE[i % PALETTE.length] }} />
            {name}
          </span>
        ))}
      </div>
    </div>
  )
}

function Pill({ value, label }) {
  if (value === null || value === undefined) return <span className="pill pill-na">n/a</span>
  return <span className={`pill pill-${scoreTone(value)}`}>{label ?? pct(value)}</span>
}

function TraceRow({ trace, expanded, onToggle }) {
  const guardTone =
    trace.guardrail === 'pass' ? 'high' : trace.guardrail === 'fail' ? 'low' : 'none'
  return (
    <>
      <tr className={`trace-row ${expanded ? 'is-open' : ''}`} onClick={onToggle}>
        <td className="trace-caret">{expanded ? '▾' : '▸'}</td>
        <td className="trace-q">
          <span className="trace-qid">{trace.question_id}</span>
          {trace.question}
        </td>
        <td className="trace-model">{trace.model}</td>
        <td><Pill value={trace.faithfulness} /></td>
        <td><Pill value={trace.answer_relevance} /></td>
        <td><span className={`pill pill-${guardTone}`}>{trace.guardrail}</span></td>
      </tr>
      {expanded && (
        <tr className="trace-detail-row">
          <td colSpan={6}>
            <div className="trace-detail">
              <div className="span-step">
                <div className="span-head"><b>1 · User query</b></div>
                <div className="span-body">{trace.question}</div>
              </div>

              <div className="span-step">
                <div className="span-head">
                  <b>2 · Retrieved context</b>
                  <span className="span-meta">
                    {trace.retrieved_context?.length || 0} chunk(s)
                    {trace.expected_sections?.length
                      ? ` · ground truth: Sec ${trace.expected_sections.join(', ')}`
                      : ''}
                    {trace.matched_entities?.length
                      ? ` · graph entities: ${trace.matched_entities.join(', ')}`
                      : ' · no graph entities matched'}
                  </span>
                </div>
                <div className="span-body">
                  {(trace.retrieved_context || []).map((c, i) => {
                    const isGold = trace.expected_sections?.some(
                      (s) => String(s).toUpperCase() === String(c.section).toUpperCase()
                    )
                    return (
                      <div key={i} className={`chunk ${isGold ? 'chunk-gold' : ''}`}>
                        <div className="chunk-head">
                          <span className="chunk-rank">#{i + 1}</span>
                          {c.act} — Sec {c.section}
                          <span className={`chunk-src src-${c.source}`}>{c.source}</span>
                          {isGold && <span className="chunk-gold-tag">ground truth</span>}
                        </div>
                        <div className="chunk-text">{(c.text || '').slice(0, 420)}…</div>
                      </div>
                    )
                  })}
                </div>
              </div>

              <div className="span-step">
                <div className="span-head">
                  <b>3 · Final LLM response</b>
                  <span className="span-meta">
                    {trace.model}
                    {trace.latency_ms ? ` · ${(trace.latency_ms / 1000).toFixed(1)}s` : ''}
                    {trace.ttft_ms ? ` · TTFT ${trace.ttft_ms.toFixed(0)}ms` : ''}
                    {trace.tokens_per_second ? ` · ${trace.tokens_per_second} tok/s` : ''}
                  </span>
                </div>
                <div className="span-body answer-body">
                  {trace.error ? <span className="err">{trace.error}</span> : trace.answer}
                </div>
              </div>

              <div className="span-step">
                <div className="span-head">
                  <b>4 · LLM-as-a-judge justification</b>
                  <span className="span-meta">RAGAS · statement-level NLI verdicts</span>
                </div>
                <div className="span-body">
                  {trace.judge_faithfulness_evaluations?.length ? (
                    trace.judge_faithfulness_evaluations.map((e, i) => (
                      <div key={i} className={`verdict ${e.verdict ? 'ok' : 'bad'}`}>
                        <span className="verdict-mark">{e.verdict ? '✓' : '✗'}</span>
                        <div>
                          <div className="verdict-statement">{e.statement}</div>
                          <div className="verdict-reason">{e.reason}</div>
                        </div>
                      </div>
                    ))
                  ) : (
                    <p className="muted-note">
                      Not judged. Run <code>python -m evaluation.analyze</code> without
                      <code>--no-judge</code> to populate statement-level verdicts.
                    </p>
                  )}
                  {trace.judge_generated_questions?.length > 0 && (
                    <div className="reverse-q">
                      <b>Reverse-engineered questions</b> (Answer Relevance):
                      <ul>
                        {trace.judge_generated_questions.map((q, i) => <li key={i}>{q}</li>)}
                      </ul>
                    </div>
                  )}
                  {trace.grounding && (
                    <p className="muted-note">
                      Deterministic guardrail (grounding.py): {trace.grounding.verified_claims}/
                      {trace.grounding.total_claims} claims verified
                      {trace.grounding.unverified_sections?.length
                        ? ` · unverified sections: ${trace.grounding.unverified_sections.join(', ')}`
                        : ''}
                    </p>
                  )}
                </div>
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  )
}

export default function MetricsTab() {
  const [report, setReport] = useState(null)
  const [error, setError] = useState(null)
  const [missing, setMissing] = useState(false)
  const [openRow, setOpenRow] = useState(null)
  const [modelFilter, setModelFilter] = useState('all')

  useEffect(() => {
    fetchEvalReport()
      .then(setReport)
      .catch((e) => {
        setMissing(!!e.missing)
        setError(e.message)
      })
  }, [])

  const traces = useMemo(() => {
    if (!report?.traces) return []
    return modelFilter === 'all'
      ? report.traces
      : report.traces.filter((t) => t.model === modelFilter)
  }, [report, modelFilter])

  if (missing) {
    return (
      <div className="card">
        <h3 style={{ marginTop: 0 }}>No evaluation report yet</h3>
        <p className="muted-note">Run the offline harness, then reload this tab:</p>
        <pre className="run-cmd">python -m evaluation.run_eval{'\n'}python -m evaluation.analyze</pre>
        <p className="muted-note">
          Add <code>--no-judge</code> to the second command to skip the LLM-as-judge metrics
          (no API calls, deterministic metrics only).
        </p>
      </div>
    )
  }
  if (error) return <div className="error-box">{error}</div>
  if (!report) return <div className="card">Loading evaluation report…</div>

  const { kpis, models, retrieval_quality: rq, judge, run_meta: meta } = report
  const perQ = rq?.per_question || []
  const gpu = meta?.host?.gpu

  const faithSpark = report.traces.map((t) => t.faithfulness).filter((v) => v != null)
  const relSpark = report.traces.map((t) => t.answer_relevance).filter((v) => v != null)

  return (
    <div className="metrics">
      <div className="metrics-head">
        <div>
          <h2>RAG Evaluation</h2>
          <p className="metrics-sub">
            RAGAS metrics (Es et al., 2024, <code>arXiv:2309.15217</code>) over{' '}
            {meta?.n_questions ?? perQ.length} questions ×{' '}
            {Object.keys(models || {}).length} models.
            {judge?.enabled
              ? ` Judge: ${judge.model} (${judge.n_judged} judged).`
              : ' Judge disabled for this report — Faithfulness and Relevance unmeasured.'}
          </p>
        </div>
        <div className="metrics-stamp">
          <div>{report.generated_at}</div>
          {meta?.graph_backend && <div>graph: {meta.graph_backend}</div>}
        </div>
      </div>

      {/* 1 — Executive KPIs */}
      <section className="kpi-row">
        <KpiCard
          label="Faithfulness"
          value={pct(kpis?.faithfulness)}
          tone={scoreTone(kpis?.faithfulness)}
          hint="Measures hallucination rate based on context. RAGAS: answer split into atomic statements, each verified by NLI against the retrieved chunks. Score = supported / total."
          spark={faithSpark}
          sparkColor="#35d39a"
          footnote={judge?.enabled ? null : 'run analyze without --no-judge'}
        />
        <KpiCard
          label="Answer Relevance"
          value={pct(kpis?.answer_relevance)}
          tone={scoreTone(kpis?.answer_relevance)}
          hint="Measures alignment with user intent. RAGAS: judge reverse-engineers 3 questions the answer resolves, then cosine similarity against the original question. Refusals score 0 by definition."
          spark={relSpark}
          sparkColor="#5b9bff"
          footnote={judge?.enabled ? 'refusals count as 0 by RAGAS definition' : 'run analyze without --no-judge'}
        />
        <KpiCard
          label="Context Precision"
          value={pct(kpis?.context_precision)}
          tone={scoreTone(kpis?.context_precision)}
          hint="Signal-to-noise ranking of the retriever. Rank-weighted Average Precision: the correct section at rank 1 scores 1.0, at rank 7 scores 0.14. Recall alone cannot see this."
          spark={perQ.map((q) => q.context_precision)}
          sparkColor="#a481ff"
          footnote={rq?.mean_recall != null ? `recall ${pct(rq.mean_recall)} · MRR ${rq.mean_reciprocal_rank}` : null}
        />
        <KpiCard
          label="P95 Latency"
          value={kpis?.latency_p95_ms ? `${(kpis.latency_p95_ms / 1000).toFixed(1)}s` : null}
          tone="none"
          hint="95th-percentile end-to-end generation latency across all models and questions."
          footnote={gpu && !gpu.available ? 'CPU-only host — see note below' : null}
        />
      </section>

      {/* Hardware honesty banner — the profiling rewrite's whole point. */}
      {gpu && !gpu.available && (
        <div className="hw-note">
          <b>No GPU on this host.</b> {gpu.reason} GPU metrics are reported as
          unavailable rather than as <code>0</code>, which would read as a measured value.
          {meta?.host && (
            <span className="hw-spec">
              {' '}Measured on {meta.host.cpu_physical_cores} physical /{' '}
              {meta.host.cpu_logical_cores} logical cores,{' '}
              {(meta.host.total_ram_mb / 1024).toFixed(1)} GB RAM.
            </span>
          )}
        </div>
      )}

      {/* 2 — Model comparison */}
      <section className="card metrics-section">
        <h3>Model comparison</h3>
        <ModelComparisonChart models={models || {}} />
        <div className="table-scroll">
          <table className="metrics-table">
            <thead>
              <tr>
                <th>Model</th><th>Faith</th><th>Relev</th><th>Halluc</th><th>Pass</th>
                <th>Latency</th><th>TTFT</th><th>tok/s</th><th>GPU</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(models || {}).map(([name, m]) => (
                <tr key={name}>
                  <td className="mono">{name}</td>
                  <td><Pill value={m.faithfulness} /></td>
                  <td><Pill value={m.answer_relevance} /></td>
                  <td>{pct(m.hallucination_rate_grounding) ?? '—'}</td>
                  <td>{pct(m.test_pass_rate) ?? '—'}</td>
                  <td>{m.latency_ms?.mean ? `${(m.latency_ms.mean / 1000).toFixed(1)}s` : '—'}</td>
                  <td>{m.ttft_ms?.measured ? `${m.ttft_ms.mean.toFixed(0)}ms` : <em className="na">not measured</em>}</td>
                  <td>{m.tokens_per_second_mean ?? '—'}</td>
                  <td>
                    {m.gpu?.available
                      ? `${m.gpu.peak_mem_mb_mean?.toFixed(0)} MB`
                      : <em className="na" title={m.gpu?.note}>n/a</em>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* 3 — Trace inspector */}
      <section className="card metrics-section">
        <div className="section-head">
          <h3>Trace inspector</h3>
          <select
            className="model-filter"
            value={modelFilter}
            onChange={(e) => { setModelFilter(e.target.value); setOpenRow(null) }}
          >
            <option value="all">All models</option>
            {Object.keys(models || {}).map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
        </div>
        <p className="muted-note">Click any row to expand the full pipeline trace for that question.</p>
        <div className="table-scroll">
          <table className="metrics-table trace-table">
            <thead>
              <tr>
                <th /><th>Query</th><th>Model</th><th>Faithfulness</th><th>Relevance</th><th>Guardrail</th>
              </tr>
            </thead>
            <tbody>
              {traces.map((t, i) => {
                const id = `${t.question_id}-${t.model}`
                return (
                  <TraceRow
                    key={id}
                    trace={t}
                    expanded={openRow === id}
                    onToggle={() => setOpenRow(openRow === id ? null : id)}
                  />
                )
              })}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  )
}
