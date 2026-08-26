import { useState } from 'react'

function formatCitation(item) {
  const parts = [item.act]
  if (item.section) parts.push(`Section ${item.section}`)
  const line = parts.join(' — ')
  return item.page ? `${line} (Page ${item.page})` : line
}

function EvidenceItem({ item }) {
  const [copied, setCopied] = useState(false)

  const shareText = `${formatCitation(item)}\n\n${item.text}\n\nSource: ${item.source_pdf}`

  async function copy() {
    try {
      await navigator.clipboard.writeText(shareText)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      // clipboard API unavailable — silently no-op, button just won't confirm
    }
  }

  async function share() {
    if (navigator.share) {
      try {
        await navigator.share({ title: formatCitation(item), text: shareText })
        return
      } catch {
        // user cancelled or share failed — fall through to copy
      }
    }
    copy()
  }

  return (
    <div className="evidence-item">
      <div className="meta">
        <span className="act-line">{formatCitation(item)}</span>
        <span>{item.source_pdf}{item.source ? ` · via ${item.source}` : ''}</span>
      </div>
      <div className="original-text">{item.text}</div>
      <div className="actions">
        <button className="ghost" onClick={copy}>{copied ? 'Copied ✓' : 'Copy'}</button>
        <button className="ghost" onClick={share}>Share</button>
      </div>
    </div>
  )
}

export default function LegalEvidenceView({ context }) {
  if (!context || context.length === 0) return null

  return (
    <div className="evidence">
      <h4>Legal Evidence View — top {context.length} retrieved section(s)</h4>
      {context.map((item, i) => (
        <EvidenceItem item={item} key={i} />
      ))}
    </div>
  )
}
