// A small, hand-curated set of real questions this app can actually answer
// well, tagged by trigger keyword — not a generative feature, just a
// keyword-matched FAQ list, built from the same 12 graph concepts + common
// real scenarios this project's own testing surfaced (RC, tint, disability,
// dashcam evidence, etc.).
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
  { keywords: ['seize', 'impound', 'confiscate'], question: 'Can the police seize my vehicle on the spot?' },
  { keywords: ['minor', 'underage', 'age'], question: 'What happens if a minor is caught driving?' },
  { keywords: ['silencer', 'exhaust', 'modif'], question: "Is it illegal to modify my vehicle's silencer?" },
  { keywords: ['number plate', 'plate', 'font'], question: 'Is a fancy number plate font illegal?' },
]

/** Up to `limit` questions whose trigger keyword appears in what's typed so
 * far — a simple substring match, not a search engine, on purpose. */
export function matchSuggestions(input, limit = 5) {
  const q = input.trim().toLowerCase()
  if (q.length < 2) return []
  const matches = SUGGESTIONS.filter(
    (s) => s.keywords.some((k) => k.includes(q) || q.includes(k)) && s.question.toLowerCase() !== q
  )
  return matches.slice(0, limit)
}
