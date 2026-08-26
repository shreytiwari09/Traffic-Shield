// Renders the REAL step-by-step progress of the current request as SSE
// events arrive from Orchestration Service — every message and elapsed_ms
// here is exactly what the server reported for an actual measured step of
// an actual inter-service call, not a scripted animation. See
// services/orchestration_service/routes.py (/v1/ask/stream) for the source.

function StepIcon({ step, isLatest }) {
  if (step === 'error') return <span className="trace-icon">✕</span>
  if (isLatest) return <span className="trace-icon spin">◐</span>
  return <span className="trace-icon">✓</span>
}

export default function PipelineTrace({ events, done }) {
  if (events.length === 0) return null

  return (
    <div className="pipeline-trace">
      <h4>Live retrieval &amp; generation pipeline</h4>
      {events.map((ev, i) => {
        const isLatest = i === events.length - 1 && !done
        return (
          <div className="trace-step" key={i}>
            <StepIcon step={ev.step} isLatest={isLatest} />
            <span className="trace-msg">{ev.message}</span>
            {typeof ev.elapsed_ms === 'number' && (
              <span className="trace-elapsed">{Math.round(ev.elapsed_ms)} ms</span>
            )}
          </div>
        )
      })}
    </div>
  )
}
