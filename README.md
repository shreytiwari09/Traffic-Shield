# Haryana Traffic Legal Assistant

An LLM application that helps an ordinary citizen understand their legal rights during a roadside stop
by traffic police in Haryana, India — answering with exact citations (Act, Section, Page) from official
Indian/Haryana motor-vehicle and evidence law, refusing to answer rather than inventing a provision it
can't find.

Built as a 5-service microservice architecture with a React frontend, on top of an offline data pipeline
that turns 6 official government PDFs into a searchable, hybrid (vector + graph) knowledge base.

**Live locally at:** `http://localhost:5173/` (React app) once running — see [Setup](#one-time-setup).
**Repo:** https://github.com/Cykikz/Traffic-shield

---

## What it does

A citizen (or an evaluator) asks a question like *"Can the officer ask for my RC?"* or *"What's the fine
for a tinted mirror?"*. The app:

1. Retrieves the actual relevant legal sections (not a paraphrase, not general knowledge) from a corpus of
   6 official Indian/Haryana legal documents.
2. Generates an answer — using either a local model (Ollama, Llama 3.1 8B) or a hosted one (Gemini),
   citizen's choice — under a fixed legal persona that must cite Act/Section/Page for every claim.
3. **Checks its own answer** against the retrieved sources before showing it, flagging any cited section
   number or rupee amount that isn't actually backed by what was retrieved.
4. Shows its work: a live, real-time trace of every step the request actually took, and a full evaluator
   dashboard comparing 5 different ways of answering the same question.

## Architecture

```
Browser (React app, :5173)
  │  /api/*
  ▼
Application Service   (:8000)  — UI + thin API passthrough, the single front door
  │  POST /v1/ask · GET /v1/ask/stream (SSE) · POST /v1/eval · GET /v1/categories
  ▼
Orchestration Service (:8001)  — sequences one request end to end
  │                     │
  │ POST /v1/retrieve   │ POST /v1/generate
  ▼                     ▼
Retrieval Service (:8002)     LLM Service (:8003)
  │  embed / vector+graph       │  only thing that talks to Ollama or Gemini
  ▼  fusion / scoring           ▼
Data Service (:8004)          Ollama (localhost:11434) / Gemini API
  │  Chroma + dataset.jsonl
```

| Service | Owns | Why it's separate |
|---|---|---|
| **Application** | UI, single entry point | UI concerns never mix with request-sequencing logic |
| **Orchestration** | Sequencing one request; SSE progress streaming; the Eval grid's 5-way coordination | Retrieval here is genuinely two-step (vector + graph) and needed its own coordinator |
| **Retrieval / RAG** | Query normalization, vector similarity, graph lookup, fusion/scoring | Retrieval and generation fail in different ways — keeping them separate means neither needs to know how the other works |
| **LLM** | The only thing that talks to Ollama or Gemini | Swapping providers/prompts touches zero retrieval code |
| **Data / Knowledge** | `dataset.jsonl` + Chroma vector store, read-only at request time | "Where data lives" is a different concern from "how to search it well" |

Two rubric-style layers exist but are deliberately **not** separate services, with reasoning built into the
app itself (see the "How It Works" tab): **Embedding** (lives inside LLM Service — it's still "talking to
Ollama") and **API/Gateway** (Application Service already is the single front door; nothing to route
between with only one client).

`data_pipeline/` is a separate **offline** batch tool — it built the dataset/chunks/graph once; the live
services only ever read what it produced, never regenerate it at request time.

## Key features (beyond plain retrieve-then-generate)

- **Hybrid retrieval, one relevance score** — vector search and a flat-JSON GraphRAG substitute both
  compete on the same cosine-similarity scale (graph candidates are scored against their own stored chunk
  embeddings), not a blind priority guess or a rule that always favors one path.
- **Query glossary** — normalizes citizen phrasing ("RC", "tint", "disabled") to the corpus's actual legal
  vocabulary, for both the embedding call and graph entity-matching.
- **Guaranteed fallback context** — the general default-penalty section always surfaces for fine-related
  questions, since a generic catch-all clause can never win on relevance score against a specific question.
- **Grounding / Hallucination Checker** — extracts the section numbers and rupee amounts an answer
  actually states, and verifies each against the real retrieved text (scoped to the section the model
  itself cited, not the whole retrieved batch). Surfaced live: a badge + warning box on every answer, a
  step in the live pipeline trace, and a stat on every Eval-tab cell.
- **Dual model provider** — Ollama (local, free, private) or Gemini (hosted), switchable per question.
- **Persona applied only when it should be** — the citizen-facing Ask flow always uses the legal
  persona+hard-rules prompt; the Eval tab's "raw model" cells deliberately strip it, so the "this app's
  pipeline" vs. "the model on its own" contrast is honest.
- **Prompt Suggestions** — keyword-matched question suggestions while typing.
- **Live pipeline trace** — real Server-Sent Events from Orchestration Service as an actual request
  crosses every service boundary — not a simulated animation.
- **Eval Dashboard** — one retrieval pass shown in full (timing, matched entities, graph relationships,
  ranked chunks), then 5 generations side by side: Ollama raw / Ollama+RAG / Gemini raw / Gemini+RAG /
  Ollama+graph-only-RAG (isolating what the graph path alone contributes).
- **Rights Library** — browse the knowledge base by category (Driver Rights, Police Powers, Documents
  Required, Traffic Signals, Challans & Fines) instead of only Q&A, reusing the same graph entities.
- **How It Works** — an interactive, in-product architecture diagram + a service-by-service explanation of
  *why* each boundary exists, including the two deliberately-merged layers.

## Tech stack

- **Backend**: Python, FastAPI, httpx, pydantic-settings, Chroma (vector store), Ollama SDK/API,
  `google-genai` (Gemini)
- **Frontend**: React + Vite, plain CSS (no UI framework), native `EventSource` for SSE
- **Data pipeline**: pymupdf (PDF parsing), Ollama `nomic-embed-text` (embeddings), a hand-rolled
  section-boundary extractor and flat-JSON GraphRAG builder

## Project structure

```
traffic-shield/
├── DATA/                     # pipeline output: dataset.jsonl, chunks.jsonl, graph/, raw & parsed PDFs
├── trafficshield_kb/         # source PDFs, organized by legal priority
├── data_pipeline/            # offline 7-phase batch pipeline (collect → parse → clean → extract →
│                              # validate → chunk+embed → graph)
├── chroma_data/              # persisted vector store (gitignored, rebuildable)
├── services/
│   ├── shared/                # settings, pydantic schemas, the legal system prompt, confidence heuristic
│   ├── app_service/            # Application Service (+ legacy Jinja2 pages, superseded by frontend/)
│   ├── orchestration_service/   # Orchestration Service (+ grounding.py — the hallucination checker)
│   ├── retrieval_service/       # Retrieval Service (+ glossary.py, fusion.py, graph_store.py)
│   ├── llm_service/             # LLM Service (ollama_client.py, gemini_client.py)
│   └── data_service/            # Data Service (chroma_store.py, dataset_store.py)
├── frontend/                  # React app — Chat, Rights Library, Eval, How It Works
├── scripts/verify_kb.py       # Ex2 smoke test — direct Chroma query, no services needed
└── requirements.txt
```

## One-time setup

```bash
pip install -r requirements.txt
cd frontend && npm install && cd ..

ollama pull llama3.1:8b          # generation model
ollama pull nomic-embed-text     # embedding model
```

**Generate the knowledge base** (builds `DATA/dataset.jsonl`, `DATA/chunks.jsonl`, the Chroma vector store,
and the graph — takes ~1 minute):

```bash
python -m data_pipeline.run_pipeline
```

Verify it's queryable (no services need to be running for this):

```bash
python scripts/verify_kb.py
```

**Gemini (optional):** copy `.env.example` to `.env` and set `GEMINI_API_KEY`. Without it, everything
still works fully on Ollama; Gemini cells/options will show "unavailable" instead of an answer. If a
model name in `.env` gets deprecated (Gemini's free-tier models rotate), swap `GEMINI_MODEL` for another
`*-flash*` model your key has access to.

## Running the app

Start all five backend services, then the frontend — six terminals (or six background processes), from
the repo root:

```bash
uvicorn services.data_service.main:app          --port 8004
uvicorn services.llm_service.main:app           --port 8003
uvicorn services.retrieval_service.main:app     --port 8002
uvicorn services.orchestration_service.main:app --port 8001
uvicorn services.app_service.main:app           --port 8000
cd frontend && npm run dev                       # :5173
```

Then open:

- **http://localhost:5173/** — the real app: Chat, Rights Library, Eval, How It Works
- **http://localhost:8000/** and **/eval** — the original Jinja2 pages, still functional, superseded by
  the React app above
- Each backend service's own Swagger docs — **`:8004/docs`**, **`:8003/docs`**, **`:8002/docs`**,
  **`:8001/docs`** — proof these are real, independently callable APIs, not internal function calls

## Verifying it end to end

1. `GET http://localhost:800{1,2,3,4}/v1/health` on each service — all report `"status": "ok"`.
2. **Chat tab**: ask a real question with Ollama, then Gemini — both should cite a real Act/Section/Page,
   and the live pipeline trace should show real, non-zero timings for every step.
3. **Eval tab**: same question — the two "raw model" cells should differ noticeably from the "this app's
   pipeline" cells (that contrast is the RAG-vs-no-RAG deliverable, demonstrated live); check the grounding
   score on each cell.
4. **Rights Library**: browse a category, confirm real sections load.
5. **How It Works**: click through the architecture nodes and the service-explanation panel.

## Data pipeline

6 official documents → 1,317 legal-section records → 1,671 embedded chunks (768-dim, `nomic-embed-text`)
→ a 14-concept flat-JSON graph (1,390 edges). Sources: Motor Vehicles Act 1988, its 2019 Amendment, Central
Motor Vehicle Rules 1989, Motor Vehicles (Driving) Regulations 2017, Haryana Motor Vehicle Rules 1993, and
Bharatiya Sakshya Adhiniyam 2023 (evidence law — added mid-project since it governs admissibility of
digital evidence like dashcam footage, relevant to a traffic-stop assistant).

## Known limitations (documented, not silently hidden)

- 51 image-only pages in the MV Act's First Schedule (road-sign plates) have no extractable text —
  correctly falls through to "no official source found," not a bug to chase.
- 18 CMVR rules and ~12 chunks are pre-existing PDF-parsing edge cases; see `data_pipeline/README.md`.
- The "graph" is flat JSON (entities.json/relationships.json), not a real graph database — a deliberate,
  documented substitute for Neo4j.
- Retrieval on **compound questions** (bundling two distinct concepts, e.g. "is this legal AND is this
  fine correct") can still favor whichever concept has the stronger keyword signal — query decomposition
  would fix this properly; not yet implemented.
- The Grounding Checker is a text-matching check, not full NLU fact-checking — it catches invented section
  numbers/amounts, not every possible inaccuracy, and matches section numbers without disambiguating by Act.
- No auth, no rate limiting, no Docker — a localhost coursework build. Docker is Exercise 5, deferred.

## Reserved for later

Real Neo4j graph database · a "Legal Update Agent" to monitor Haryana gazettes for amendments · query
decomposition for compound questions · Docker/containerization (Exercise 5) · broader glossary coverage
for colloquial terms (e.g. "dashcam"/"CCTV" don't yet trigger the Electronic Record graph concept, only
its formal name does) · **voice-based question input and spoken answers** — was part of the original
Application Service vision ("voice interaction"), not yet built; needs a speech-tech decision first (see
below) before implementation starts.
