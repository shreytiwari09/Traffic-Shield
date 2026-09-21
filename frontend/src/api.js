// All requests go through the Vite dev proxy to Application Service (:8000),
// which is the single front door for the whole backend — this file never
// talks to Orchestration/Retrieval/etc. directly.

export function streamAsk(question, provider, { conversationId, onEvent, onDone, onError } = {}) {
  // Only the conversation id travels, never the transcript: EventSource can
  // only issue a GET, and a multi-turn transcript in a query string would run
  // into URL length limits. The turns themselves live in Orchestration.
  const params = new URLSearchParams({ question, provider })
  if (conversationId) params.set('conversation_id', conversationId)
  const source = new EventSource(`/api/ask/stream?${params.toString()}`)

  source.onmessage = (ev) => {
    let data
    try {
      data = JSON.parse(ev.data)
    } catch {
      return
    }
    onEvent?.(data)
    if (data.step === 'done') {
      onDone?.(data)
      source.close()
    } else if (data.step === 'error') {
      onError?.(data.message || 'Something went wrong')
      source.close()
    }
  }

  source.onerror = () => {
    onError?.('Connection to the server was lost')
    source.close()
  }

  return () => source.close()
}

export async function restoreConversation(conversationId, messages) {
  if (!conversationId || !messages?.length) return
  try {
    await fetch(`/api/conversations/${encodeURIComponent(conversationId)}/restore`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages }),
    })
  } catch {
    // Best-effort: if this fails the visible chat is still correct, the model
    // just answers the next question without the earlier turns.
  }
}

export async function resetConversation(conversationId) {
  if (!conversationId) return
  try {
    await fetch(`/api/conversations/${encodeURIComponent(conversationId)}`, { method: 'DELETE' })
  } catch {
    // Best-effort. The client starts a brand-new conversation id regardless,
    // so a failed reset only leaves an orphaned transcript server-side, which
    // expires on its own (see conversation.py TTL) — nothing user-visible.
  }
}

export async function runEval(question) {
  const res = await fetch('/api/eval', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  })
  const data = await res.json()
  if (!res.ok) throw new Error(data.detail || 'Eval request failed')
  return data
}

export async function fetchEvalReport() {
  const res = await fetch('/api/eval/report')
  const data = await res.json().catch(() => ({}))
  if (res.status === 404) {
    // Not an error state — the offline harness just hasn't been run yet. The
    // dashboard renders instructions rather than a failure.
    const err = new Error(data.detail || 'No evaluation report yet.')
    err.missing = true
    throw err
  }
  if (!res.ok) throw new Error(data.detail || 'Failed to load the evaluation report')
  return data
}

export async function fetchCategories() {
  const res = await fetch('/api/categories')
  if (!res.ok) throw new Error('Failed to load categories')
  return res.json()
}

export async function fetchCategorySections(slug) {
  const res = await fetch(`/api/categories/${slug}/sections`)
  if (!res.ok) throw new Error('Failed to load this category')
  return res.json()
}
