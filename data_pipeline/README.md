# TrafficShield Data Pipeline

Implements `DATA/Haryana_Traffic_Data_Blueprint (1).md` — the database-building
stage that turns the official PDFs in `trafficshield_kb/` into a validated,
section-level legal dataset with vector chunks and a GraphRAG index.

Scope is data only. No app or model development happens here.

## Running it

```bash
python -m data_pipeline.run_pipeline                  # all seven phases
python -m data_pipeline.run_pipeline --skip-embeddings  # no Ollama needed
python -m data_pipeline.run_pipeline --only 4         # re-run one phase
python -m data_pipeline.run_pipeline --from 4         # resume from Phase 4
```

Phases 1–5 and 7 need only the PDFs. Phase 6 needs a running Ollama with an
embedding model pulled (`ollama pull nomic-embed-text`).

Check whether the corpus can actually answer a driver's questions:

```bash
python -m data_pipeline.audit_coverage -v
```

## Phases

| # | Phase | Output |
|---|---|---|
| 1 | Collection | `data/raw_pdfs/` + `manifest.json` (SHA-256 per document) |
| 2 | PDF parsing | `data/parsed_markdown/` — headings, tables, `<!-- page: N -->` markers |
| 3 | Cleaning | `data/cleaned_sections/` — TOC/index/blank/duplicate pages removed |
| 4 | Section extraction | `data/dataset.jsonl` — one record per legal section |
| 5 | Metadata validation | `data/validation_report.json`, `rejected_records.jsonl` |
| 6 | Vectorization | `data/chunks.jsonl` + Chroma collection `trafficshield_sections` |
| 7 | GraphRAG | `data/graph/entities.json`, `relationships.json` |

Every phase writes its stats into `data/pipeline_report.json`.

## Record schema

The blueprint's eight required fields, plus fields the project brief needs for
authority-ranked retrieval:

```json
{
  "id": "HR_MVA_130",
  "act": "Motor Vehicles Act, 1988",
  "section": "130",
  "title": "Duty to produce licence and certificate of registration",
  "content": "(1) The driver of a motor vehicle in any public place shall…",
  "page": 79,
  "source_pdf": "motor_vehicles_act_1988.pdf",
  "state": "Haryana",

  "status": "in_force",
  "unit": "section",
  "jurisdiction": "India",
  "authority": "Government of India",
  "document_type": "Act",
  "priority": 1,
  "char_count": 2332
}
```

`status` is `omitted` for provisions repealed by amendment. They are kept
deliberately: the assistant should be able to say a section no longer applies
rather than simply not know about it.

## How section extraction works

Documents are split at legal-section boundaries, never at a character count.
The five documents print their headers three different ways, so Phase 4 keeps
one grammar per style (`config.SOURCES[...]["parser_style"]`):

```
act          27. Power of Central Government to make rules.—The Central …
haryana      11.
             Mutilated license [Section 28(2)(c)].—(1) If at any time …
rule_heading 3. General
             The provisions of sub-section (1) of section 3 shall not apply …
regulation   6. Lane traffic.
             (1) Where any road is marked by lanes …
```

Three details in these PDFs cause most of the difficulty, and each has a
matching defence in `phase4_sections.py`:

- **Amended sections hide behind footnote markers** — `3[129. Wearing of
  protective headgear.—`. Without allowing that prefix, every currently
  amended provision is missed.
- **Footnotes are numbered like rules** — `1. Substituted by G.S.R. 338(E)…`.
  These are rejected by wording, since otherwise they outnumber the real rules.
- **A single bad candidate used to discard everything after it.** A fee table
  reading as "section 100" once cost 38 consecutive Haryana rules. Selection is
  now a longest-increasing-subsequence over candidates, so an outlier loses to
  the many headers that agree with each other.

## Coverage

Measured by `audit_coverage.py` against the sections a driver asserting their
rights would need — 16/16 scenarios have every governing provision present with
full text.

| Document | Sections | Numbered-provision coverage |
|---|---|---|
| Motor Vehicles Act, 1988 | 256 | 216/217 |
| Haryana Motor Vehicle Rules, 1993 | 233 | 227/230 |
| Central Motor Vehicles Rules, 1989 | 545 | 146/164 |
| MV (Driving) Regulations, 2017 | 39 | 39/39 |
| MV (Amendment) Act, 2019 | 74 | amendment items |

Known limitations, all traceable to the source PDFs:

- **51 pages of the MV Act (121–171) have no text layer.** They are the First
  Schedule's road-sign plates — pictorial, not prose. Recorded per document as
  `image_only_pages` in `pipeline_report.json`.
- **18 CMVR rules are not extractable**, including rule 139, which the PDF
  prints mid-line with its title duplicated inside a footnote bracket and no
  delimiter before the body. The MV Act provisions on the same points
  (ss. 130, 158) are present in full.
- **12 of 1,494 chunks exceed the 800-token band** (max 880). All are CMVR form
  templates whose dotted fill-in rules inflate the token estimate; no
  prose section is affected.

## Token estimation

`tiktoken` is not a dependency, and would be the wrong tokenizer for an Ollama
model regardless, so `text_utils.estimate_tokens` approximates from characters
and words. Chunk sizes in reports are estimates, not exact counts.
