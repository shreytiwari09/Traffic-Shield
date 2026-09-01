// Per-browser question history (localStorage) — private to this viewer, per
// the artifact/browser-storage convention: never sent anywhere, survives
// reloads, lost if the user clears site data or uses a different browser.
// That's fine here — it's a UX convenience (what did I actually ask before),
// not something that needs to be reliable or shared.

const KEY = 'traffic-shield:question-history'
const MAX_HISTORY = 30

export function getHistory() {
  try {
    const raw = localStorage.getItem(KEY)
    const parsed = raw ? JSON.parse(raw) : []
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

export function addToHistory(question) {
  const q = question.trim()
  if (!q) return
  try {
    const existing = getHistory().filter((h) => h.toLowerCase() !== q.toLowerCase())
    const updated = [q, ...existing].slice(0, MAX_HISTORY)
    localStorage.setItem(KEY, JSON.stringify(updated))
  } catch {
    // localStorage unavailable (private window, blocked site data, etc.) —
    // suggestions just fall back to the curated list, nothing breaks.
  }
}
