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
  │  fusion / scoring           ▼
  ├──────────────┐            Ollama (localhost:11434) / Gemini API
  ▼              ▼
Data Service  Neo4j (bolt://localhost:7687)
  (:8004)       Cypher: entity match, relationships,
  │  Chroma      evidence ranking, multi-hop expansion
  │  + dataset.jsonl
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

- **Hybrid retrieval, one relevance score** — vector search (Chroma) and graph search (Neo4j) both
  compete on the same cosine-similarity scale (graph candidates are scored against their own stored chunk
  embeddings), not a blind priority guess or a rule that always favors one path.
- **Real graph database** — entity alias matching, relationship lookup and evidence ranking all run as
  Cypher against Neo4j. Because sections are real nodes rather than string ids inside a JSON blob, the
  graph is genuinely traversable: `GRAPH_EXPAND_HOPS=2` reaches evidence through *connected* entities, a
  hop the previous flat-JSON store structurally could not make. (Measured caveat: hop-2 ids are appended
  after all hop-1 ids and `_MAX_GRAPH_EVIDENCE` keeps only the first 20, so on questions that already have
  20+ direct candidates the expansion is truncated away — see `graph_expand_hops` in `settings.py`.)
  The JSON store is retained as a verified fallback — `scripts/verify_neo4j.py` asserts both backends
  return identical entities, relationships and ranked evidence, so the swap is proven equivalent rather
  than assumed.
- **Multi-turn conversation** — the Chat tab is a real conversation, not a series of unrelated questions.
  Follow-ups work (*"and if I refuse?"*), including for **retrieval**: a follow-up that would embed to
  nothing useful on its own is resolved against the previous turn before it is searched on, so the model
  never gets correct conversation history alongside a context block about something else.
- **Query glossary** — normalizes citizen phrasing ("RC", "tint", "disabled") to the corpus's actual legal
  vocabulary, for both the embedding call and graph entity-matching.
- **Guaranteed fallback context** — the general default-penalty section always surfaces for fine-related
  questions, since a generic catch-all clause can never win on relevance score against a specific question.
- **Grounding / Hallucination Checker** — extracts the section numbers and rupee amounts an answer
  actually states, and verifies each against the real retrieved text (scoped to the section the model
  itself cited, not the whole retrieved batch). Surfaced live: a badge + warning box on every answer, a
  step in the live pipeline trace, and a stat on every Eval-tab cell.
- **Tells you when you're being charged for a non-offence** — "no source found" is split into two very
  different cases, because collapsing them made the assistant useless in the situation where it matters
  most. If the answer needs a legal fact that wasn't retrieved, it refuses honestly. But if the user is
  being *accused* of something no retrieved provision makes an offence at all ("he's fining me for not
  wearing sunglasses"), refusing would leave them paying for nothing — so it says so and hands them the
  lever: make the officer name the section and write it on the challan. It states the limit of what it
  knows ("nothing in the official sources I have makes this an offence"), never "this is legal" — and
  when genuinely unsure which case it is in, it falls back to refusing, so a missed retrieval can never
  become a false all-clear on a real offence.
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

- **Backend**: Python, FastAPI, httpx, pydantic-settings, Chroma (vector store), Neo4j (graph store,
  via the official `neo4j` Bolt driver), Ollama SDK/API, `google-genai` (Gemini)
- **Frontend**: React + Vite, plain CSS (no UI framework), native `EventSource` for SSE
- **Data pipeline**: pymupdf (PDF parsing), Ollama `nomic-embed-text` (embeddings), a hand-rolled
  section-boundary extractor and GraphRAG builder (emits JSON; `scripts/load_neo4j.py` loads it into Neo4j)

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
│   ├── orchestration_service/   # Orchestration Service (+ grounding.py — the hallucination checker,
│   │                            #   conversation.py — multi-turn memory + follow-up query resolution)
│   ├── retrieval_service/       # Retrieval Service (+ glossary.py, fusion.py; graph_store.py dispatches
│   │                            #   to neo4j_store.py or graph_store_json.py)
│   ├── llm_service/             # LLM Service (ollama_client.py, gemini_client.py)
│   └── data_service/            # Data Service (chroma_store.py, dataset_store.py)
├── frontend/                  # React app — Chat, Rights Library, Eval, How It Works
│   ├── Dockerfile             #   multi-stage: Vite build -> nginx serving static + /api proxy
│   └── nginx.conf             #   container-side equivalent of the Vite dev proxy
├── evaluation/                # RAG evaluation harness (see "Evaluation" below)
│   ├── run_eval.py            #   streamed sweep: TTFT, tok/s, peak RAM, dynamic GPU detection
│   ├── judge.py               #   RAGAS LLM-as-judge: Faithfulness + Answer Relevance
│   ├── retrieval_metrics.py   #   rank-aware Context Precision, Recall, MRR
│   ├── profiling.py           #   hardware capability detection + peak sampling
│   └── analyze.py             #   -> metrics_report.json, consumed by the RAG Metrics tab
├── scripts/
│   ├── verify_kb.py           # Ex2 smoke test — direct Chroma query, no services needed
│   ├── load_neo4j.py          # builds the Neo4j graph from DATA/graph/ (offline, idempotent)
│   └── verify_neo4j.py        # asserts the Neo4j and JSON graph backends behave identically
├── Dockerfile                 # one shared image for all five Python services
├── docker-compose.yml         # five services + Neo4j + frontend, healthcheck-ordered
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

### Graph database (Neo4j)

The graph half of hybrid retrieval runs on Neo4j. Like Ollama, the **server is infrastructure this app
points at**, not something it embeds — `pip install neo4j` gives you the *driver* only.

**Start a server** (any one of these):

```bash
docker compose up neo4j -d                      # easiest, if you use the Docker path below
# or Neo4j Desktop (https://neo4j.com/download/), create a local DBMS
# or Community Server: https://neo4j.com/deployment-center/ — unzip, then:
#   bin/neo4j-admin dbms set-initial-password <password>
#   bin/neo4j console
```

Neo4j 5.x requires **Java 17 or 21** — it warns and misbehaves on newer JDKs. If your default `java` is
something else, point `JAVA_HOME` at a 17/21 JDK for the Neo4j process only.

**Then set the password in `.env` and load the graph:**

```bash
# .env:  NEO4J_PASSWORD=<the password you set>
python -m scripts.load_neo4j        # builds Entity/Section nodes + RELATES/EVIDENCED_BY edges
python -m scripts.verify_neo4j      # proves it matches the JSON store exactly
```

`load_neo4j.py` is idempotent — rerun it after any pipeline rerun. Use `--wipe` if the rerun *removed*
entities, since a plain reload MERGEs and would leave the deleted ones behind.

**Don't want Neo4j at all?** Set `GRAPH_BACKEND=json` and everything still works on the original in-process
flat-JSON store — you only lose multi-hop expansion. If Neo4j is configured but unreachable, the service
falls back to JSON automatically rather than losing the graph path entirely, and reports it at
`GET :8002/v1/health` as `"fell_back": true`.

Verify it's queryable (no services need to be running for this):

```bash
python scripts/verify_kb.py
```

**Gemini (optional):** copy `.env.example` to `.env` and set `GEMINI_API_KEY`. Without it, everything
still works fully on Ollama; Gemini cells/options will show "unavailable" instead of an answer. If a
model name in `.env` gets deprecated (Gemini's free-tier models rotate), swap `GEMINI_MODEL` for another
`*-flash*` model your key has access to.

## Running the app — with Docker

One command brings up all five services plus the frontend:

```bash
docker compose up --build
```

Then open **http://localhost:5173/**. Same ports as the manual setup below, so every URL in this README
works unchanged.

**Ollama must already be running on the host** (`ollama serve`, with `llama3.1:8b` and `nomic-embed-text`
pulled) — it is intentionally not containerized, see [Known limitations](#known-limitations-documented-not-silently-hidden).
The knowledge base must also already be built: `chroma_data/` and `DATA/` are bind-mounted, not baked into
the image, so run `python -m data_pipeline.run_pipeline --only 6` on the host first if `chroma_data/` is
missing.

Neo4j **is** containerized here (unlike Ollama — the graph is ~1.3k nodes, so there is no multi-GB
download or CPU-inference penalty to avoid). Its graph is not baked into the image either; load it from
the host once after the first `up`, which is what bolt on `7687` is published for:

```bash
python -m scripts.load_neo4j
python -m scripts.verify_neo4j
```

Notes on how it's wired:

- **One image, five services.** All five FastAPI services share `services/` and the same pinned
  requirements, so `Dockerfile` is built once and compose overrides `command:` per service.
- **No code changed to containerize.** Inter-service URLs already came from `services/shared/settings.py`,
  so compose just sets the env vars that were always there — `localhost:8004` becomes `data_service:8004`.
- **The frontend is a production build**, served by nginx (`frontend/Dockerfile` is multi-stage), with
  nginx proxying `/api/*` to Application Service exactly as the Vite dev proxy does. `vite.config.js` is
  untouched, so `npm run dev` on the host still works identically.
- **Startup is ordered by healthchecks**, not guesswork: retrieval waits for data + llm, orchestration
  waits for retrieval, app waits for orchestration, frontend waits for app.
- `docker compose ps` shows health per service; `docker compose logs -f <service>` tails one of them.

Your `GEMINI_API_KEY` is passed in from `.env` by compose at runtime. It is never copied into an image —
`.env` is in `.dockerignore`.

## Running the app — manually

Start Neo4j and Ollama first (see above), then all five backend services and the frontend — six terminals
(or six background processes), from the repo root:

```bash
uvicorn services.data_service.main:app          --port 8004
uvicorn services.llm_service.main:app           --port 8003
uvicorn services.retrieval_service.main:app     --port 8002
uvicorn services.orchestration_service.main:app --port 8001
uvicorn services.app_service.main:app           --port 8000
cd frontend && npm run dev                       # :5173
```

Confirm the graph backend actually connected — this is the one thing that fails *quietly*, since
retrieval falls back to the JSON store rather than erroring:

```bash
curl localhost:8002/v1/health
# {"graph": {"backend": "neo4j", "fell_back": false, ...}}   <- what you want
# {"graph": {"backend": "json",  "fell_back": true,  ...}}   <- Neo4j unreachable, check neo4j_error
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
→ a graph of 1,331 entities (14 alias-bearing domain concepts + the 1,317 sections) and 1,390 relationships,
emitted as JSON by the pipeline and loaded into Neo4j by `scripts/load_neo4j.py`. Sources: Motor Vehicles Act 1988, its 2019 Amendment, Central
Motor Vehicle Rules 1989, Motor Vehicles (Driving) Regulations 2017, Haryana Motor Vehicle Rules 1993, and
Bharatiya Sakshya Adhiniyam 2023 (evidence law — added mid-project since it governs admissibility of
digital evidence like dashcam footage, relevant to a traffic-stop assistant).

## Evaluation

The offline harness scores the pipeline on the four RAGAS metrics
(Es, S., James, J., Espinosa-Anke, L., & Schockaert, S., 2024, *"RAGAS: Automated Evaluation of Retrieval
Augmented Generation"*, EACL 2024, [arXiv:2309.15217](https://arxiv.org/abs/2309.15217)). Rank weighting for
Context Precision follows the standard IR Average Precision formulation (Manning, Raghavan & Schütze,
*Introduction to Information Retrieval*, 2008, ch. 8).

```bash
python -m evaluation.run_eval        # streamed sweep over 30 questions x 4 models
python -m evaluation.analyze         # adds the LLM-as-judge metrics -> metrics_report.json
python -m evaluation.analyze --no-judge   # deterministic metrics only, zero API calls
```

Results render in the app's **RAG Metrics** tab (`http://localhost:5173/`): executive KPIs, a model
comparison chart, and a trace inspector that expands each question into its full pipeline — query,
retrieved chunks (with the ground-truth chunk highlighted and its rank visible), the answer, and the
judge's per-statement verdicts.

| Metric | How it is computed |
|---|---|
| **Faithfulness** | Judge splits the answer into atomic pronoun-free statements, then returns an NLI verdict per statement against the retrieved context. Score = supported / total. |
| **Answer Relevance** | Judge reverse-engineers 3 questions the answer resolves; score is mean cosine similarity to the real question (`nomic-embed-text`). Noncommittal answers score 0 by definition. |
| **Context Precision** | Rank-weighted Average Precision over retrieved chunks — the correct section at rank 1 scores 1.0, at rank 7 scores 0.14. |
| **Context Recall** | Fraction of hand-verified ground-truth sections the retriever surfaced. |

Two deliberate choices worth knowing:

- **The judge must not be a model under test.** `JUDGE_MODEL` defaults to Gemini; every verdict is cached
  by content hash in `judge_cache.json`, so an interrupted sweep resumes instead of re-spending quota.
- **Retrieval relevance is not LLM-judged.** `ground_truth.json` already records the exact sections a
  correct answer must cite, verified against the corpus text — an objective, reproducible, zero-cost
  signal that a judge would only re-derive worse.

**Hardware metrics are capability-detected, never assumed.** If the host has no GPU, the report says so
(`"gpu": {"available": false, "reason": "..."}`) rather than emitting a number. This matters: the previous
committed report carried `gpu_mem_used_mb: 5350.7` recorded on entirely different hardware, and reported
2063.6 MB of "GPU memory" for *Gemini* — a network call that touches no local GPU. TTFT and tokens/sec are
measured from the real token stream, and tokens/sec states its own basis (`decode_window` vs
`total_elapsed`) because a provider that returns an answer in two chunks has no meaningful decode window.

## Known limitations (documented, not silently hidden)

- 51 image-only pages in the MV Act's First Schedule (road-sign plates) have no extractable text —
  correctly falls through to "no official source found," not a bug to chase.
- 18 CMVR rules and ~12 chunks are pre-existing PDF-parsing edge cases; see `data_pipeline/README.md`.
- Follow-up questions are resolved for retrieval by a **heuristic**, not an LLM rewrite: a message that
  looks like a continuation gets the previous question prepended before it is embedded. An LLM-based
  standalone-question rewriter would be more accurate, but on this project's CPU-only Ollama path it means
  a second generation (~175s) on every follow-up. Both failure modes are cheap — a false positive slightly
  dilutes the embedding, a false negative degrades to plain single-turn behavior. See
  `services/orchestration_service/conversation.py`.
- Conversation memory is **in-process**, so it is lost when Orchestration restarts. The browser keeps its
  own transcript and pushes it back on load, which covers reloads and restarts, but two Orchestration
  replicas behind a load balancer would not share memory — a real deployment would need Redis or similar.
- Retrieval on **compound questions** (bundling two distinct concepts, e.g. "is this legal AND is this
  fine correct") can still favor whichever concept has the stronger keyword signal — query decomposition
  would fix this properly; not yet implemented.
- The Grounding Checker is a text-matching check, not full NLU fact-checking — it catches invented section
  numbers/amounts, not every possible inaccuracy, and matches section numbers without disambiguating by Act.
- No auth and no rate limiting — a localhost coursework build.
- Ollama is **not** containerized: it runs on the host and the containers reach it via
  `host.docker.internal`. Deliberate — bundling an 8B model would mean shipping ~5GB of weights into a
  volume and running inference inside the VM. It's treated as infrastructure the app depends on, the way
  you'd point at a database rather than embed one.

## Reserved for later

A "Legal Update Agent" to monitor Haryana gazettes for amendments · query
decomposition for compound questions · an LLM-based follow-up rewriter (viable once generation is not
CPU-bound) · broader glossary coverage
for colloquial terms (e.g. "dashcam"/"CCTV" don't yet trigger the Electronic Record graph concept, only
its formal name does) · **voice-based question input and spoken answers** — was part of the original
Application Service vision ("voice interaction"), not yet built; needs a speech-tech decision first (see
below) before implementation starts.
