import { useState } from 'react'

// Every field here is a fact about the actual running code (ports, files,
// endpoints) — not illustrative/invented. Keep in sync with services/*.
const NODES = [
  {
    id: 'browser', row: 0, icon: '🧑', label: 'Citizen / Evaluator (Browser)', port: null,
    summary: 'Opens the React app, asks a question, browses the Rights Library, or runs an Eval comparison.',
    detail: 'Not a service — the React SPA running in the browser (this app). Talks to exactly one thing: Application Service. Never calls Orchestration/Retrieval/LLM/Data directly.',
    calls: ['app'],
  },
  {
    id: 'app', row: 1, icon: '🚪', label: 'Application Service', port: 8000,
    summary: 'The single front door. Serves the UI and thin passthrough API routes — no logic of its own.',
    detail: 'Owns the citizen-facing UI (Ask/Chat, Rights Library, Eval tabs) and nothing else: no retrieval, no LLM calls, no data access happen here. Every /api/* route (including the /api/ask/stream SSE relay) is a pure passthrough to Orchestration Service.\n\nEndpoints: GET / · GET /api/ask/stream (SSE) · POST /api/ask · POST /api/eval · GET /api/categories(/{slug}/sections)\nFile: services/app_service/',
    calls: ['orchestration'],
    calledBy: ['browser'],
  },
  {
    id: 'orchestration', row: 2, icon: '🎛️', label: 'Orchestration Service', port: 8001,
    summary: 'Coordinates ONE request end to end — calls Retrieval, then LLM, and assembles the result.',
    detail: 'Sequences the whole flow: calls Retrieval Service for context, then LLM Service to generate an answer; computes a confidence score; assembles citations. For the Eval tab it runs retrieval once and 4 generations (2 providers × with/without persona+RAG). Streams real progress events (not simulated) as each step actually completes — that\'s the live pipeline view on the Chat tab.\n\nEndpoints: POST /v1/ask · GET /v1/ask/stream (SSE) · POST /v1/eval · GET /v1/categories(/{slug}/sections)\nFile: services/orchestration_service/',
    calls: ['retrieval', 'llm'],
    calledBy: ['app'],
  },
  {
    id: 'retrieval', row: 3, icon: '🔎', label: 'Retrieval Service', port: 8002,
    summary: 'Hybrid retrieval: vector search + graph lookup, fused by one relevance score.',
    detail: 'Normalizes known abbreviations (RC → registration certificate, etc.), embeds the question (via LLM Service), searches Chroma for similar chunks (via Data Service), matches known legal-concept entities against the question using the flat-JSON graph, scores graph-evidenced sections on the SAME cosine-similarity scale as vector hits (not a blind priority guess), and fuses everything into one ranked context list.\n\nEndpoints: POST /v1/retrieve · GET /v1/categories(/{slug}/sections)\nFile: services/retrieval_service/',
    calls: ['llm', 'data'],
    calledBy: ['orchestration'],
  },
  {
    id: 'llm', row: 3, icon: '🧠', label: 'LLM Service', port: 8003,
    summary: 'The only thing that talks to Ollama or Gemini. Knows nothing about retrieval.',
    detail: 'Two jobs: embed text (always via Ollama nomic-embed-text) and generate an answer — Ollama llama3.1:8b or Gemini, whichever the citizen/evaluator picked — under the fixed legal persona, but ONLY when real retrieval happened. The Eval tab\'s "raw model" cells skip the persona on purpose, so the contrast is honest.\n\nEndpoints: POST /v1/embed · POST /v1/generate\nFile: services/llm_service/',
    calls: ['ollama', 'gemini'],
    calledBy: ['orchestration', 'retrieval'],
  },
  {
    id: 'data', row: 4, icon: '🗃️', label: 'Data Service', port: 8004,
    summary: 'Owns the dataset and the vector store. Read-only at request time.',
    detail: 'An in-memory index over DATA/dataset.jsonl (1,317 legal-section records) plus a read-time wrapper over the Chroma collection that data_pipeline built offline (1,675 embedded chunks). Never writes at request time, never talks to Ollama.\n\nEndpoints: POST /v1/vector-search · GET /v1/records/{id} · GET /v1/embeddings/{id}\nFile: services/data_service/',
    calls: ['chroma'],
    calledBy: ['retrieval'],
  },
  {
    id: 'ollama', row: 4, icon: '💻', label: 'Ollama (local)', port: 11434, external: true,
    summary: 'llama3.1:8b (generation) + nomic-embed-text (embeddings) — running on this machine.',
    detail: 'A local model server on this machine, not one of this app\'s own services — the primary generation/embedding backend. Runs fully offline; no data leaves the machine for Ollama calls.',
    calledBy: ['llm'],
  },
  {
    id: 'gemini', row: 4, icon: '☁️', label: 'Gemini API', port: null, external: true,
    summary: "gemini-3.6-flash — Google's hosted model, the second provider choice.",
    detail: 'Called only when a citizen or evaluator explicitly picks Gemini. Requires GEMINI_API_KEY in .env; the app degrades gracefully (a clear "unavailable" message, not a crash) if it\'s unset.',
    calledBy: ['llm'],
  },
  {
    id: 'chroma', row: 5, icon: '📦', label: 'Chroma (vector store)', port: null, external: true,
    summary: 'Embedded vector database, persisted to chroma_data/ on disk.',
    detail: 'Built offline by data_pipeline\'s Phase 6 — 1,675 chunk embeddings (768-dim, nomic-embed-text). Data Service is the only thing that ever queries it.',
    calledBy: ['data'],
  },
]

const byId = Object.fromEntries(NODES.map((n) => [n.id, n]))
const rows = [...new Set(NODES.map((n) => n.row))].sort((a, b) => a - b)

// The reasoning behind every boundary — including the two concerns that are
// deliberately NOT split into their own service, and why not.
const SERVICE_EXPLANATIONS = [
  {
    name: 'Application Service', status: 'separate', statusLabel: 'SEPARATE SERVICE',
    what: 'The UI (Ask/Chat, Rights Library, Eval tabs) and the single entry point the browser talks to.',
    why: "Kept separate from Orchestration so \"how citizens interact\" (tabs, forms, rendering) never mixes with \"how one request is sequenced.\" The UI can be redesigned — this whole React app was rebuilt mid-project — without touching a single line of orchestration logic.",
  },
  {
    name: 'RAG / Retrieval Service', status: 'separate', statusLabel: 'SEPARATE SERVICE',
    what: 'Turning a question into ranked context: query normalization, vector similarity, graph lookup, fusion/scoring.',
    why: 'Kept separate from LLM Service because retrieval and generation genuinely fail in different ways — a bad retrieval (wrong section found) is a different bug from a bad generation (model misreads good context). LLM Service never needs to know HOW context was found; Retrieval never needs to know which model will read it.',
  },
  {
    name: 'LLM Service', status: 'separate', statusLabel: 'SEPARATE SERVICE',
    what: 'The only thing that talks to Ollama or Gemini — embedding and generation, both providers.',
    why: 'Kept separate from Retrieval so swapping providers, changing the persona prompt, or adding a new model touches zero retrieval code. Also why the Eval tab\'s "raw model" cells can call it directly, bypassing Retrieval entirely, without duplicating any Ollama/Gemini logic.',
  },
  {
    name: 'Data / Knowledge Service', status: 'separate', statusLabel: 'SEPARATE SERVICE',
    what: 'The dataset (1,317 records) and the Chroma vector store, at serving time.',
    why: 'Kept separate from Retrieval because "where the data lives and how it\'s stored" is a different concern from "how to search it well" — Retrieval\'s ranking algorithm has been rewritten twice this project without Data Service changing at all. Caveat: the actual document processing (PDF → chunks → embeddings) runs offline in data_pipeline/, not in this service — a rare, heavy batch job, not a per-request concern, so it deliberately lives outside the live services entirely.',
  },
  {
    name: 'Embedding', status: 'merged', statusLabel: 'NOT SEPARATE — INSIDE LLM SERVICE',
    what: 'Turning text into a vector (nomic-embed-text via Ollama).',
    why: 'Not its own service: embedding is still "talking to Ollama," which LLM Service already owns exclusively. A standalone Embedding Service would mean two services both calling Ollama — splitting one external dependency across a boundary for no actual benefit.',
  },
  {
    name: 'API / Gateway', status: 'merged', statusLabel: 'NOT SEPARATE — INSIDE APPLICATION SERVICE',
    what: 'A single point of entry for all requests.',
    why: 'Not its own service, because there is only one client today (this browser app) — a dedicated gateway earns its place once a second client exists (e.g. a future mobile app, or a separate admin tool) that needs routing/auth/rate-limiting shared across clients. Right now Application Service already is that single front door.',
  },
  {
    name: 'Orchestration Layer', status: 'separate', statusLabel: 'SEPARATE SERVICE',
    what: 'Sequencing one request end to end: call Retrieval, then LLM, assemble citations + confidence.',
    why: "Not folded into Application Service because this app's retrieval is genuinely two-step (vector + graph, fused), and the Eval tab's 4-way comparison (2 providers × with/without RAG) needed real coordination logic that has nothing to do with serving HTML/JSON to a browser.",
  },
]

function Node({ node, expanded, onToggle }) {
  return (
    <div className={`arch-node ${node.external ? 'external' : ''} ${expanded ? 'expanded' : ''}`} onClick={onToggle}>
      <div className="arch-node-head">
        <span className="arch-icon">{node.icon}</span>
        <span className="arch-label">{node.label}</span>
        {node.port && <span className="mode-badge">:{node.port}</span>}
        <span className="arch-caret">{expanded ? '▾' : '▸'}</span>
      </div>
      <p className="arch-summary">{node.summary}</p>
      {expanded && (
        <div className="arch-detail">
          <p style={{ whiteSpace: 'pre-line' }}>{node.detail}</p>
          <div className="arch-links">
            {node.calls?.length > 0 && (
              <div><span className="stat-label">Calls</span> {node.calls.map((id) => byId[id].label).join(', ')}</div>
            )}
            {node.calledBy?.length > 0 && (
              <div><span className="stat-label">Called by</span> {node.calledBy.map((id) => byId[id].label).join(', ')}</div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export default function ArchitectureTab() {
  const [expandedId, setExpandedId] = useState(null)

  return (
    <div>
      <div className="card">
        <h3 style={{ marginTop: 0 }}>How this app is built</h3>
        <p style={{ fontSize: '0.85rem', color: 'var(--muted)', margin: 0 }}>
          One user request flows straight down this diagram: Browser → Application Service →
          Orchestration Service → Retrieval Service (which itself calls LLM Service to embed the
          question, and Data Service to search the vector store) → LLM Service again (to generate
          the final answer) → back up the chain. Click any box for what it owns, its real API
          endpoints, and which file it lives in.
        </p>
      </div>

      <div className="arch-diagram">
        {rows.map((row) => (
          <div className="arch-row" key={row}>
            {NODES.filter((n) => n.row === row).map((node) => (
              <Node
                key={node.id}
                node={node}
                expanded={expandedId === node.id}
                onToggle={() => setExpandedId(expandedId === node.id ? null : node.id)}
              />
            ))}
          </div>
        ))}
      </div>

      <div className="card" style={{ marginTop: '1.25rem' }}>
        <h4 style={{ marginTop: 0 }}>Services in this system — and why each one is separate</h4>
        <p style={{ fontSize: '0.82rem', color: 'var(--muted)', marginTop: 0 }}>
          Not a checklist — the point of splitting anything into its own service is that it can
          change without touching the others. Each entry below is the actual reason that boundary
          exists here, including the two concerns that are deliberately NOT their own service.
        </p>
        <div className="service-explain-list">
          {SERVICE_EXPLANATIONS.map((s) => (
            <div className="service-explain" key={s.name}>
              <div className="service-explain-head">
                <span className="service-explain-name">{s.name}</span>
                {s.status && <span className={`mode-badge ${s.status === 'merged' ? 'warn' : ''}`}>{s.statusLabel}</span>}
              </div>
              <p className="service-explain-what"><b>What it owns:</b> {s.what}</p>
              <p className="service-explain-why"><b>Why this boundary:</b> {s.why}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
