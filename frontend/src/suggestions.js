// A curated seed set of real questions this app can answer well, tagged by
// trigger keyword — the baseline for a first-time user with no history yet.
// Not the whole story anymore: matchSmartSuggestions() below blends this
// with the user's own real question history, so suggestions get better the
// more the app is actually used, instead of staying a fixed list forever.
export const SUGGESTIONS = [
  { keywords: ['rc', 'registration'], question: 'Can the officer ask for my RC?' },
  { keywords: ['dl', 'licence', 'license'], question: 'Can my driving licence be seized on the spot?' },
  { keywords: ['tint', 'film', 'mirror', 'window'], question: 'Is my window tint legal?' },
  { keywords: ['fine', 'penalty', 'challan'], question: 'What is the fine for not wearing a helmet?' },
  { keywords: ['helmet'], question: 'Is it mandatory to wear a helmet as a pillion rider?' },
  { keywords: ['seatbelt', 'seat belt'], question: 'What is the fine for not wearing a seatbelt?' },
  { keywords: ['speed', 'speeding'], question: 'What is the fine for overspeeding?' },
  { keywords: ['parking', 'parked'], question: 'Can my car be towed for illegal parking?' },
  { keywords: ['disab', 'handicap'], question: 'What are the rules for a disabled person driving?' },
  { keywords: ['insurance'], question: 'Can the officer ask for my insurance certificate?' },
  { keywords: ['puc', 'pollution'], question: 'Do I need to carry a PUC certificate?' },
  { keywords: ['dashcam', 'cctv', 'camera', 'footage', 'video'], question: 'Is dashcam footage admissible as evidence against me?' },
  { keywords: ['stop', 'signal', 'red light'], question: 'What happens if I jump a red light?' },
  { keywords: ['drunk', 'alcohol', 'drink'], question: 'What is the penalty for drunk driving?' },
  { keywords: ['phone', 'mobile', 'call'], question: 'Can I be fined for using a mobile phone while driving?' },
  { keywords: ['id', 'identity', 'badge'], question: "Can I ask to see the officer's ID before showing my documents?" },
  { keywords: ['seize', 'impound', 'confiscate', 'key', 'keys'], question: 'Can the police seize my vehicle on the spot?' },
  { keywords: ['minor', 'underage', 'age'], question: 'What happens if a minor is caught driving?' },
  { keywords: ['silencer', 'exhaust', 'modif'], question: "Is it illegal to modify my vehicle's silencer?" },
  { keywords: ['number plate', 'plate', 'font'], question: 'Is a fancy number plate font illegal?' },
]

/** Which of the curated topics a piece of text touches on, by keyword. */
function topicsIn(text) {
  const lower = text.toLowerCase()
  return SUGGESTIONS.filter((s) => s.keywords.some((k) => lower.includes(k)))
}

/** Up to `limit` questions whose trigger keyword appears in what's typed so
 * far — the original, simple behavior, kept as the fallback layer. */
export function matchSuggestions(input, limit = 12) {
  const q = input.trim().toLowerCase()
  if (q.length < 2) return []
  const matches = SUGGESTIONS.filter(
    (s) => s.keywords.some((k) => k.includes(q) || q.includes(k)) && s.question.toLowerCase() !== q
  )
  return matches.slice(0, limit)
}

/**
 * The smart version: blends the user's own real question history with the
 * curated list, so suggestions reflect what this citizen actually asks
 * about, not just a fixed 20-question list.
 *
 * With no input typed yet (input === ''): shows the most recent real
 * questions first (quick recall), then curated questions matching topics
 * the user has shown interest in before but hasn't asked yet (inferred from
 * history's keyword overlap with SUGGESTIONS) — genuinely personalized, not
 * random. Falls back to the plain curated list for a first-time user with
 * no history.
 *
 * With input typed: real past questions that match take priority over the
 * curated list (they're proven relevant — the user asked them for real),
 * then curated matches fill any remaining slots.
 */
export function matchSmartSuggestions(input, history, limit = 12) {
  const q = input.trim().toLowerCase()
  const results = []
  const seen = new Set()

  function push(question) {
    const key = question.toLowerCase()
    if (seen.has(key) || key === q) return
    seen.add(key)
    results.push(question)
  }

  if (q.length === 0) {
    // Default view: recent real questions, then curated questions matching
    // this user's inferred interests that they haven't asked yet.
    history.slice(0, 6).forEach(push)
    const interestKeywords = new Set(topicsIn(history.join(' ')).flatMap((s) => s.keywords))
    SUGGESTIONS
      .filter((s) => s.keywords.some((k) => interestKeywords.has(k)))
      .forEach((s) => push(s.question))
    // Still short (new user, no history)? Fill from the plain curated list.
    if (results.length < limit) {
      SUGGESTIONS.forEach((s) => push(s.question))
    }
    return results.slice(0, limit)
  }

  if (q.length < 2) return []

  // Real past questions that match what's being typed — proven relevant.
  history.filter((h) => h.toLowerCase().includes(q)).forEach(push)
  // Curated matches fill the rest.
  matchSuggestions(input, limit).forEach((s) => push(s.question))

  return results.slice(0, limit)
}
