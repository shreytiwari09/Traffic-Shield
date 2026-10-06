import { useState } from 'react'
import LegalEvidenceView from './LegalEvidenceView'
import { sendFeedback } from '../api'

const CONFIDENCE_LABEL = {
  high: 'High confidence — retrieved from official source',
  medium: 'Medium confidence — partial official-source support',
  low: 'Low confidence — weak retrieval match',
  none: 'No official source found',
}

// The persona is conversational now (no fixed heading template), but models
// still reach for **bold** to stress a provision or an amount — render those
// as real emphasis rather than showing the literal asterisks.
function renderAnswer(text) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g)
  return parts.map((part, i) =>
    part.startsWith('**') && part.endsWith('**')
      ? <b key={i}>{part.slice(2, -2)}</b>
      : <span key={i}>{part}</span>
  )
}

function groundingBadge(grounding) {
  if (!grounding || grounding.total_claims === 0) {
    return <span className="badge">No checkable claims (no section/₹ figures cited)</span>
  }
  const clean = grounding.unverified_claims === 0
  return (
    <span className={`badge ${clean ? 'confidence-high' : 'confidence-low'}`}>
      {clean ? '✓' : '⚠'} Grounding: {grounding.verified_claims}/{grounding.total_claims} claims verified
    </span>
  )
}

// LLMOps feedback loop. Shown only for answers that went through the model
// (a guardrail refusal has no model answer to rate).
function FeedbackBar({ result }) {
  const [state, setState] = useState('idle') // idle | sending | sent | error
  const [rating, setRating] = useState(null)
  const [comment, setComment] = useState('')

  async function submit(nextRating, withComment = '') {
    setRating(nextRating)
    setState('sending')
    try {
      await sendFeedback({
        request_id: result.request_id,
        rating: nextRating,
        comment: withComment || null,
        question: result.question || result.retrieval_query || '',
        answer: result.answer,
        provider: result.provider,
        model: result.model,
        prompt_version: result.prompt_version,
        conversation_id: result.conversation_id,
        cited_sections: (result.citations || []).map((c) => c.section).filter(Boolean),
        grounding_unverified: result.grounding?.unverified_claims ?? null,
      })
      setState('sent')
    } catch (err) {
      console.error('feedback failed', err)
      setState('error')
    }
  }

  if (state === 'sent') {
    return <p className="feedback-bar muted">Thanks — feedback recorded{rating === 'down' ? ' and queued for review' : ''}.</p>
  }

  return (
    <div className="feedback-bar">
      <span className="muted">Was this answer useful?</span>
      <button className="ghost" aria-label="Helpful" disabled={state === 'sending'} onClick={() => submit('up')}>👍</button>
      <button className="ghost" aria-label="Not helpful" disabled={state === 'sending'} onClick={() => setRating('down')}>👎</button>
      {rating === 'down' && (
        <>
          <input
            className="feedback-comment"
            placeholder="What was wrong? (optional)"
            value={comment}
            maxLength={1000}
            onChange={(e) => setComment(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') submit('down', comment) }}
          />
          <button className="ghost" disabled={state === 'sending'} onClick={() => submit('down', comment)}>Send</button>
        </>
      )}
      {state === 'error' && <span className="feedback-error">Could not send — try again.</span>}
      {result.prompt_version && <span className="feedback-meta">prompt {result.prompt_version}</span>}
    </div>
  )
}

export default function ResponseCard({ result }) {
  const [showEvidence, setShowEvidence] = useState(false)
  const { answer, citations, provider, model, confidence, context, used_context, matched_entities, grounding } = result

  return (
    <div className="card response-card">
      <div className="badges">
        <span className="badge haryana">Haryana Applicable</span>
        <span className={`badge confidence-${confidence}`}>{CONFIDENCE_LABEL[confidence]}</span>
        <span className="badge">{provider} · {model}</span>
        {!used_context && <span className="badge confidence-low">No retrieval used</span>}
        {groundingBadge(grounding)}
      </div>

      {grounding?.unverified_claims > 0 && (
        <div className="error-box" style={{ marginBottom: '0.9rem', marginTop: 0 }}>
          This answer states {grounding.unverified_claims === 1 ? 'a claim' : `${grounding.unverified_claims} claims`} not
          found in the cited sources —
          {grounding.unverified_sections.length > 0 && (
            <> Section{grounding.unverified_sections.length > 1 ? 's' : ''} {grounding.unverified_sections.join(', ')}</>
          )}
          {grounding.unverified_sections.length > 0 && grounding.unverified_amounts.length > 0 && ' and '}
          {grounding.unverified_amounts.length > 0 && (
            <> amount{grounding.unverified_amounts.length > 1 ? 's' : ''} ₹{grounding.unverified_amounts.join(', ₹')}</>
          )} — double-check these before relying on them.
        </div>
      )}

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

      {result.request_id && model !== 'guardrail-regex' && <FeedbackBar result={result} />}
    </div>
  )
}
