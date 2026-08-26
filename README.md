# Haryana Traffic Legal Assistant

An LLM application that helps an ordinary citizen understand their legal rights during a roadside stop
by traffic police in Haryana, India — answering with exact citations (Act, Section, Page) from official
Motor Vehicles Act / Central & Haryana Motor Vehicle Rules, never fabricating a provision.

Built as a 5-service architecture (Application, Orchestration, Retrieval, LLM, Data) demonstrating
Exercises 1–4 of the coursework pipeline: **Application → API → Ollama → Response** (Ex1) →
**Knowledge Base: Chunking → Embeddings → Vector Representation** (Ex2) →
**Retrieval/RAG: Query Embedding → Vector Similarity → Context → Response** (Ex3) →
**Services + Orchestration** (Ex4). Exercise 5 (Docker) is intentionally out of scope for this stage.

## Architecture

```
Browser
  │
  ▼
Application Service  (:8000)  — Ask tab (citizen) + Eval tab (evaluator)
  │  POST /v1/ask | /v1/eval
  ▼
Orchestration Service (:8001) — sequences one request end to end
  │                     │
  │ POST /v1/retrieve   │ POST /v1/generate
  ▼                     ▼
Retrieval Service (:8002)   LLM Service (:8003)
  │  embed / vector-search    │  talks to Ollama AND Gemini
  ▼                           ▼
Data Service (:8004)        Ollama (localhost:11434) / Gemini API
  │  Chroma + dataset.jsonl
```

- **Data Service** — owns `DATA/dataset.jsonl` and the Chroma vector store built from `DATA/chunks.jsonl`.
- **LLM Service** — the only thing that talks to Ollama or Gemini. Embeds text; generates answers under
  the fixed legal system prompt (`services/shared/prompts.py`). Knows nothing about retrieval.
- **Retrieval Service** — hybrid retrieval: vector search (via Data Service) + a flat-JSON graph lookup
  (`DATA/graph/entities.json` / `relationships.json`, loaded at startup — the deliberate stand-in for a
  real graph database; Neo4j is reserved for future work) + fusion/ranking.
- **Orchestration Service** — the Ask tab (single provider + always-on RAG) and the Eval tab (the 2×2
  comparison grid: {Ollama, Gemini} × {no context, hybrid RAG context}).
- **Application Service** — the UI. Two tabs, two personas: **Ask** (citizen-facing, pick Ollama or
  Gemini, answer is always grounded) and **Eval** (evaluator-facing, see all four combinations side by
  side — the interactive version of a model/retrieval comparison).

`data_pipeline/` stays a separate offline batch tool (unchanged) — it built `DATA/dataset.jsonl`,
`DATA/chunks.jsonl`, and `DATA/graph/*.json` already. The services only *read* what it produced.

## One-time setup

```bash
pip install -r requirements.txt
ollama pull llama3.1:8b          # generation model (skip if already pulled)
ollama pull nomic-embed-text     # embedding model — needed once, for Exercise 2
```

**Generate real embeddings** (today `DATA/chunks.jsonl` has empty `"embedding": []` on every chunk —
nothing has been persisted into Chroma yet):

```bash
python -m data_pipeline.run_pipeline --only 6
```

Verify the knowledge base is actually queryable (no services need to be running for this):

```bash
python scripts/verify_kb.py
```

**Gemini (optional):** copy `.env.example` to `.env` and set `GEMINI_API_KEY`. Without it, the Ask tab
still works fully on Ollama, and the Eval tab's Gemini cells will show "unavailable" instead of an answer.

## Running the app

Start all five services (five terminals, or five background processes), **in this order**, from the repo
root:

```bash
uvicorn services.data_service.main:app          --port 8004
uvicorn services.llm_service.main:app           --port 8003
uvicorn services.retrieval_service.main:app     --port 8002
uvicorn services.orchestration_service.main:app --port 8001
uvicorn services.app_service.main:app           --port 8000
```

Then open:

- **http://localhost:8000/** — Ask tab (citizen-facing)
- **http://localhost:8000/eval** — Eval tab (model/RAG comparison grid)
- **http://localhost:8004/docs**, **:8003/docs**, **:8002/docs**, **:8001/docs** — each service's own
  Swagger UI, proof these are real, independently callable APIs, not just internal function calls.

## Verifying it end to end

1. `GET http://localhost:800{1,2,3,4}/v1/health` on each service — all should report `"status": "ok"`.
2. Ask tab: ask a real question (e.g. *"Can the officer ask for my RC?"*) with Ollama, then with Gemini —
   both should return an answer citing a real Act/Section/Page from `DATA/dataset.jsonl`.
3. Eval tab: ask the same question — the two "no retrieval" cells should visibly hedge/refuse ("I don't
   have an official source…") while the two "hybrid RAG" cells cite real provisions. That contrast *is*
   Exercise 3's RAG-vs-no-RAG deliverable, now demonstrated inside Exercise 4's service topology.

## Known limitations (documented, not bugs to chase)

- 51 image-only pages in the MV Act's First Schedule (road-sign plates) have no extractable text —
  questions about specific sign designs correctly fall through to "no official source found."
- 18 CMVR rules and ~12 chunks are pre-existing edge cases from PDF parsing; see `data_pipeline/README.md`.
- The "graph" is 1,372 flat-JSON edges (5 hand-defined + 1,367 keyword/alias-derived `MENTIONS` edges),
  not a real graph database — an honest, deliberate substitute for Neo4j at this stage.
- No auth, no rate limiting, no Docker — this is a localhost coursework build; Docker is Exercise 5.

## Reserved for later

Real Neo4j graph database · a "Legal Update Agent" to monitor Haryana gazettes/notifications for
amendments · Docker/containerization (Exercise 5). None of these are built here, all noted so they aren't
silently dropped from the project's original vision.
