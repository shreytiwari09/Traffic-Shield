# Exercise 5 — RAG Pipeline Analysis

This traces the pipeline (Question → Retrieval → Context → Response) through real
cases hit while building and hardening this app — not synthetic examples. Every
trace below is from an actual run against the live services, and every bug
described was found by reading the corpus text directly, not by trusting either
the app's own output or a third-party "fact-check."

## How the pipeline works, in order

1. **Question** arrives at the Orchestration Service (`/v1/ask` or `/v1/eval`).
2. Orchestration calls the **Retrieval Service** (`/v1/retrieve`).
3. Retrieval Service:
   - embeds the question (Ollama `nomic-embed-text`),
   - runs a ChromaDB vector search over `DATA/chunks.jsonl` embeddings,
   - separately alias-matches the question text against `entities.json` to find
     mentioned legal concepts (e.g. "helmet", "RC", "police officer"),
   - looks up `relationships.json` for graph-linked sections for those entities,
   - computes real cosine similarity for graph candidates too (not just vector
     hits) so both paths compete on the same score,
   - applies a small keyword-boost and a curated core-section boost
     (`core_sections.py`) for hand-verified always-relevant sections,
   - fuses, dedupes, and returns the top-K (K=8) chunks as `context`.
4. Orchestration builds the persona system prompt with that `context` inlined
   and sends it to the chosen **LLM Service** provider (Ollama or Gemini).
5. The model generates an answer, which is expected to cite only what's in
   `context`.
6. The **Grounding Checker** (`grounding.py`) regex-extracts every section/rule
   number and rupee amount the answer claims, and checks each against the
   actually-retrieved context — this is the hallucination-rate signal used
   throughout this evaluation.

## Case 1 — Good retrieval, good answer: "Can the officer ask for my RC?"

- **Question → Context**: vector search alone surfaces MVA Sec 158(1)(b) (RC is
  explicitly listed as demandable) with a strong cosine score; the entity
  matcher also fires on "RC"/"Registration Certificate" and pulls in Sec 130.
- **Context → Response**: the model correctly cites 158(1)(b) as the operative
  provision. This is the pipeline working as designed — vector and graph paths
  agree, and the answer stays inside the retrieved text.
- Why this one works reliably: "RC" has almost no vocabulary mismatch between
  the question and the statute's own wording, so vector search alone is
  already strong; the graph path is redundant-but-confirming rather than
  load-bearing here.

## Case 2 — Retrieval failure fixed by curation: helmet fine

- **Original failure**: the correct section (MVA Sec 129 / Sec 194D) scored
  just under the vector top-K cutoff (~0.6368 against a ~0.638 threshold) on
  some phrasings of the question ("helmet fine" vs. the statute's own wording),
  so it was a *valid graph candidate* but still lost its context slot to
  weaker-but-higher-scoring chunks.
- **Root cause**: being retrievable in principle (present in the graph, present
  in Chroma) was not the same as being *ranked* highly enough to survive
  fusion — a pure scoring/ranking problem, not a missing-data problem.
- **Fix**: `core_sections.py` — a small, hand-verified table of
  "always consider as a candidate" sections per entity (e.g. `"Helmet":
  ["HR_MVA_129", "HR_MVA_194D"]`), boosted +0.03 in `fusion.py` so it competes
  fairly rather than being force-included regardless of relevance.
- **Result after fix**: same question now reliably retrieves both the
  requirement (129) and the penalty (194D) sections together, and the model's
  answer correctly states both the ₹1,000 fine and the 3-month licence
  disqualification.
- **Takeaway**: retrieval bugs in this system were more often ranking/scoring
  bugs than missing-data bugs — the law was already indexed; it just didn't
  win its slot.

## Case 3 — Hallucination *despite* correct context: vehicle keys

- **Question**: "Can a police officer take my keys from my vehicle without
  permission?"
- **Context retrieved**: no section in the corpus explicitly addresses vehicle
  keys. Retrieval correctly surfaced the closest real material: Sec 130/206/207
  (production and detention powers during a stop) — a reasonable, honest
  result given no exact-match provision exists.
- **What the model did wrong**: instead of scoping its answer to the retrieved
  130/206/207 material (or admitting no exact provision addresses keys), the
  Ollama-generated answer cited Sec 213 — a gazetted-officer's power to search
  a *premises* — and applied it to a roadside *vehicle stop*.
- **Why this matters as a distinct failure mode**: this is not fabrication in
  the usual sense. Section 213 is 100% real, verbatim corpus text (verified
  directly against the source, disputing an external "fact-check" that
  initially called it invented). The failure is **context misapplication**:
  real law, wrong situation. The persona prompt's hard rules at the time only
  guarded against *inventing* text, not against citing real text that doesn't
  actually apply to the facts described.
- **Fix applied**: added an explicit "check applicability, not just existence"
  rule to `TRAFFIC_LEGAL_SYSTEM_PROMPT` — before citing any provision, check it
  actually applies to the citizen's specific situation, not just that it
  exists and is the most-similar text available. This is a prompt-level
  mitigation; retrieval-side fixes cannot address this class of error because
  retrieval did its job correctly here.

## Case 4 — Pure fabrication (not misapplication): mirror tint fine

- **Question**: "How much fine for using a mirror film with 20 percent black
  tint?"
- **Context retrieved**: Sec 100 (VLT/tint standard) and Sec 177 (general
  default penalty, ₹500 first offence / ₹1,500 subsequent) — the honest,
  correct answer, since no tint-specific fine section exists in the corpus.
- **What the model did wrong**: repeatedly fabricated a ₹10,000 figure that
  appears nowhere in the retrieved context or the corpus at all — unlike Case
  3, this wasn't real text applied to the wrong situation; the number simply
  didn't exist anywhere in the source material.
- **Why the grounding checker didn't originally catch this**: an early version
  of `grounding.py` checked a claimed amount against the *entire* retrieved
  context blob rather than only the section the model actually cited under —
  so a coincidental ₹1,000 appearing in an unrelated context chunk was enough
  to mark a fabricated ₹1,000 claim "verified." Fixed by scoping
  amount-verification to only the context item(s) whose section number the
  model cited (`cited_text` in `grounding.py`).
- **Takeaway**: the grounding checker itself needed the same discipline as the
  generation prompt — checking against "the retrieved batch" is looser than
  checking against "the specific thing actually cited," and the looser version
  produces false negatives on exactly the cases it exists to catch.

## Case 5 — Invented restriction not present in any source: "only traffic police"

- **Question**: "Can a police officer charge me 1000 rupees for no helmet?
  Since they are not traffic police."
- **Context retrieved**: Sec 129/194D — correct, and the text does not mention
  any "traffic wing" restriction; enforcement authority is not narrowed to a
  specific police branch.
- **What the model did wrong**: on multiple runs, Ollama's answer invented a
  "only traffic police can enforce this" restriction that appears nowhere in
  the retrieved context.
- **Proven non-determinism**: the same question, given the identical retrieved
  context, produced *different* answers across repeated runs with zero code
  changes in between — sometimes correct (any uniformed officer may enforce),
  sometimes fabricating the traffic-police-only restriction. This was
  established by direct repeated testing, not assumed.
- **Why this is the hardest failure mode**: it is neither a retrieval problem
  (context was correct and complete) nor a straightforwardly promptable one —
  it is generation-level unreliability inherent to the small local model under
  repeated sampling. Prompt and retrieval fixes reduce the *rate* of this
  failure but cannot eliminate it deterministically, which is the central,
  still-open finding of this evaluation (see Exercise 4's trade-off analysis
  and the per-model hallucination-rate numbers in `metrics_report.json`).

## Summary of failure modes observed, by pipeline stage

| Stage | Failure mode | Example | Fixable by |
|---|---|---|---|
| Retrieval — ranking | Correct section retrievable but loses its context slot to a near-miss score | Helmet fine (Case 2) | Retrieval-side (curated boost) |
| Retrieval — vocabulary mismatch | Question phrasing doesn't match statute wording | Tint, disability, RC edge cases | Retrieval-side (keyword/phrase boost) |
| Generation — context misapplication | Real, retrieved text applied to the wrong factual situation | Vehicle keys / Sec 213 (Case 3) | Prompt-side (applicability rule) |
| Generation — pure fabrication | A number/claim invented with no basis anywhere in context | Tint fine ₹10,000 (Case 4) | Prompt-side + grounding checker as a safety net |
| Generation — invented restriction | An unsupported qualifier added to an otherwise-correct citation | "Traffic police only" (Case 5) | Not reliably fixable by prompt/retrieval alone — inherent to small-model sampling variance |
| Grounding checker itself | False positive/negative on amount or section-format checks | Whole-context-blob amount check; missing "Rule N" pattern | Checker-side (scoping, regex coverage) |

The clearest conclusion: retrieval-side and prompt-side fixes closed real,
fixable gaps (Cases 1–4), but Case 5 — a small local model changing its answer
run-to-run given *identical* correct context — is a generation-reliability
ceiling that this architecture cannot fully engineer around. This is exactly
the kind of finding the model-comparison evaluation (Exercises 1–4) is meant
to surface quantitatively: does Gemini's hallucination rate on the same
questions/context differ meaningfully from Ollama's, and by how much.
