import { useEffect, useState } from 'react'
import { fetchCategories, fetchCategorySections } from '../api'

function SectionCard({ section }) {
  const [expanded, setExpanded] = useState(false)
  return (
    <div className="evidence-item" onClick={() => setExpanded((v) => !v)} style={{ cursor: 'pointer' }}>
      <div className="meta">
        <span className="act-line">{section.act} — Section {section.section}</span>
        <span>Page {section.page}</span>
      </div>
      <div style={{ fontSize: '0.85rem', marginBottom: expanded ? '0.4rem' : 0 }}>{section.title}</div>
      {expanded && <div className="original-text">{section.content}</div>}
    </div>
  )
}

export default function RightsLibraryTab() {
  const [categories, setCategories] = useState(null)
  const [active, setActive] = useState(null)
  const [sections, setSections] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    fetchCategories()
      .then((data) => setCategories(data.categories))
      .catch((e) => setError(e.message))
  }, [])

  async function openCategory(cat) {
    setActive(cat)
    setSections(null)
    setError(null)
    setLoading(true)
    try {
      const data = await fetchCategorySections(cat.slug)
      setSections(data.sections)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  if (error && !categories) return <div className="error-box">{error}</div>
  if (!categories) return <p className="loading">Loading Rights Library…</p>

  return (
    <div>
      {!active && (
        <div className="category-grid">
          {categories.map((cat) => (
            <button className="category-card" key={cat.slug} onClick={() => openCategory(cat)}>
              <h3>{cat.label}</h3>
              <p>{cat.entity_names.join(', ')}</p>
            </button>
          ))}
        </div>
      )}

      {active && (
        <div>
          <button className="ghost" onClick={() => setActive(null)}>← All categories</button>
          <h3 style={{ marginTop: '0.8rem' }}>{active.label}</h3>
          {loading && <p className="loading">Loading sections…</p>}
          {error && <div className="error-box">{error}</div>}
          {sections && (
            <div className="section-list">
              {sections.map((s) => (
                <SectionCard section={s} key={s.id} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
