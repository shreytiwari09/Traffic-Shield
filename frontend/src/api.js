// All requests go through the Vite dev proxy to Application Service (:8000),
// which is the single front door for the whole backend — this file never
// talks to Orchestration/Retrieval/etc. directly.

export function streamAsk(question, provider, { onEvent, onDone, onError }) {
  const params = new URLSearchParams({ question, provider })
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
