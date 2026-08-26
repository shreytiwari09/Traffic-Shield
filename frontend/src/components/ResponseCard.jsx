import { useState } from 'react'
import LegalEvidenceView from './LegalEvidenceView'

const CONFIDENCE_LABEL = {
  high: 'High confidence — retrieved from official source',
  medium: 'Medium confidence — partial official-source support',
  low: 'Low confidence — weak retrieval match',
  none: 'No official source found',
}

// The LLM's answer follows the system prompt's fixed 4-part shape and uses
// **bold** for its section headers — render those as real emphasis rather
// than showing the literal asterisks.
function renderAnswer(text) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g)
  return parts.map((part, i) =>
    part.startsWith('**') && part.endsWith('**')
      ? <b key={i}>{part.slice(2, -2)}</b>
      : <span key={i}>{part}</span>
  )
}

export default function ResponseCard({ result }) {
  const [showEvidence, setShowEvidence] = useState(false)
  const { answer, citations, provider, model, confidence, context, used_context, matched_entities } = result

  return (
    <div className="card response-card">
      <div className="badges">
        <span className="badge haryana">Haryana Applicable</span>
        <span className={`badge confidence-${confidence}`}>{CONFIDENCE_LABEL[confidence]}</span>
        <span className="badge">{provider} · {model}</span>
        {!used_context && <span className="badge confidence-low">No retrieval used</span>}
      </div>

      <div className="answer-text">{renderAnswer(answer)}</div>

      {citations.length > 0 && (
        <div className="citations">
          <h4>Legal citation</h4>
          {citations.map((c, i) => (
            <span className="citation-chip" key={i} onClick={() => setShowEvidence(true)}>
              {c.act}{c.section ? ` — Sec ${c.section}` : ''}{c.page ? `, p.${c.page}` : ''}
            </span>
          ))}
        </div>
      )}

      {matched_entities?.length > 0 && (
        <p style={{ fontSize: '0.75rem', color: 'var(--muted)', marginTop: '0.6rem' }}>
          Graph-matched concepts: {matched_entities.join(', ')}
        </p>
      )}

      {context?.length > 0 && (
        <button className="ghost" style={{ marginTop: '0.8rem' }} onClick={() => setShowEvidence((v) => !v)}>
          {showEvidence ? 'Hide' : 'Show'} original legal text ({context.length} chunk{context.length === 1 ? '' : 's'})
        </button>
      )}

      {showEvidence && <LegalEvidenceView context={context} />}
    </div>
  )
}
