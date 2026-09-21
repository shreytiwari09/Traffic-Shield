// The browser's half of a conversation.
//
// Two things are stored, and they are stored for different reasons:
//
//   * the conversation id — an opaque handle the backend keys its own
//     transcript on (see orchestration_service/conversation.py). The browser
//     never sends the transcript itself on a question, only this id, because
//     the live-progress stream is an SSE GET and a growing transcript in a
//     query string hits URL length limits.
//
//   * the transcript — kept here so a reload doesn't wipe the visible chat.
//
// Those two have different lifetimes, which is the subtle part: this survives
// a reload, the server's copy does not survive a service restart or its TTL.
// So on load we push the transcript back (restoreConversation), otherwise the
// user would see their whole chat on screen and get an answer generated with
// no memory of any of it.

const ID_KEY = 'traffic-shield:conversation-id'
const TRANSCRIPT_KEY = 'traffic-shield:transcript'

// Matches MAX_TURNS in orchestration_service/conversation.py. Kept in sync
// deliberately: storing more here than the server will ever use would make
// the restore push turns straight into the bin.
const MAX_TURNS = 12

function newId() {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID()
  return `c-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`
}

export function getConversationId() {
  try {
    const existing = localStorage.getItem(ID_KEY)
    if (existing) return existing
    const fresh = newId()
    localStorage.setItem(ID_KEY, fresh)
    return fresh
  } catch {
    // Private window / blocked site data: fall back to a per-load id. The chat
    // still works within this page view, it just won't survive a reload.
    return newId()
  }
}

export function startNewConversation() {
  const fresh = newId()
  try {
    localStorage.setItem(ID_KEY, fresh)
    localStorage.removeItem(TRANSCRIPT_KEY)
  } catch {
    // Nothing to clean up if storage is unavailable.
  }
  return fresh
}

export function loadTranscript() {
  try {
    const raw = localStorage.getItem(TRANSCRIPT_KEY)
    const parsed = raw ? JSON.parse(raw) : []
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

export function saveTranscript(messages) {
  try {
    localStorage.setItem(TRANSCRIPT_KEY, JSON.stringify(messages.slice(-MAX_TURNS)))
  } catch {
    // Over quota or unavailable — the in-memory chat is unaffected, it just
    // won't survive a reload. Not worth surfacing to the user.
  }
}

/** The transcript reduced to what the model needs: role + text, nothing else.
 *  The citations, grounding and context attached to each assistant turn are
 *  for the UI only and would just burn prompt tokens. */
export function toModelMessages(messages) {
  return messages
    .filter((m) => (m.role === 'user' || m.role === 'assistant') && m.content)
    .map((m) => ({ role: m.role, content: m.content }))
    .slice(-MAX_TURNS)
}
