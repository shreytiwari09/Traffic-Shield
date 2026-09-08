# Exercise 4 — Analysis of Results

Companion to `metrics_report.json` (Exercise 3). Exercise 3 produced the numbers;
this file argues what they mean. Every claim below cites a figure that is actually in
`metrics_report.json` or `results.jsonl` — nothing here is asserted from impression.

Metric definitions are **not** repeated here; they are stated once, authoritatively,
in the module docstring of `analyze.py` (lines 7–33). Read that first — "correctness"
and "relevance" have no standard meaning for a legal Q&A app, so the definitions are
part of the result.

---

## 1. Run provenance and what it limits

Stated up front because it changes how two columns may be read.

| Fact | Value |
|---|---|
| Questions attempted | 27 (`questions.json`) |
| Questions with results | **26** — Q9 lost to a Retrieval Service HTTP 500 mid-run |
| Scored for correctness | 21 (5 excluded as "open", see §5) |
| Models | 4 |
| Result rows | 104 (26 × 4), **0 generation errors** |
| Machine | A group member's workstation (NVIDIA GPU present), not the CPU-only laptop this repo is currently set up on |

Q9 ("What is the penalty for drunk driving?") is missing because retrieval failed, not
because generation failed, and not because it was dropped for convenience. The loss is
**uniform across all four models**, so it does not bias the comparison — it only
shrinks the sample. `run_log.txt` records the failure verbatim.

The `gpu_mem_used_mb` column is **not attributable per model**. It is an `nvidia-smi`
whole-device snapshot, and Gemini — a network call that uses no local GPU — reports
`2063.6`, byte-identical to StarCoder2's figure. That identical value is the machine's
floor, not consumption. The same applies to `cpu_percent` and `ram_used_mb`, which are
whole-host `psutil` readings: with a 3.0–4.8 % spread across models, the differences
are inside noise and **no resource conclusion should be drawn from them**. This is a
limitation of the harness, recorded rather than hidden.

---

## 2. The comparison table

Retrieval is identical for every model (same service, same K, same corpus), so any
difference below is attributable to model choice — which is precisely what Exercise 1
asked us to isolate.

**Shared retrieval baseline:** mean recall **0.553**, hit rate **0.632** (n=19).

> These are the *corrected* figures. `metrics_report.json` as generated reports 0.595 /
> 0.667 (n=21) because of the empty-list scoring bug described in §8 item 3, which awards
> the two refusal questions a free perfect recall. Every retrieval figure in this document
> uses the corrected values; re-running a fixed `analyze.py` will reproduce them.

| Model | Latency mean | Latency p95 | Hallucination | Correctness | Relevance | Prompt tok | Completion tok |
|---|---|---|---|---|---|---|---|
| gemini-3.5-flash-lite | **4,231 ms** | **8,981 ms** | **0.075** | **0.333** | 0.470 | 2,659 | 185 |
| llama3.1:8b | 14,487 ms | 18,288 ms | 0.220 | 0.238 | **0.803** | 2,526 | 200 |
| codellama:7b | 21,822 ms | 33,618 ms | 0.500 | 0.095 | 0.784 | 2,974 | 207 |
| starcoder2:3b | 6,554 ms | 8,450 ms | 1.000 | 0.000 | 0.486 | **22** ⚠ | 325 |

Correctness verdicts (denominator 21 scorable):

| Model | pass | fail | open (excluded) | claims checked |
|---|---|---|---|---|
| gemini-3.5-flash-lite | 7 | 14 | 5 | 40 |
| llama3.1:8b | 5 | 16 | 5 | 50 |
| codellama:7b | 2 | 19 | 5 | 44 |
| starcoder2:3b | 0 | 21 | 5 | 5 |

---

## 3. The StarCoder2 row is not a valid comparison

This must be said before any ranking, because reporting StarCoder2 as a fair fourth
arm would be a false result.

**StarCoder2 never received the retrieved context.** Its mean prompt length is
**22.3 tokens** against 2,526–2,974 for every other model. Per-question values are
17, 19, 18, 19, 20, 21, 17, 25 — the bare question and nothing else. The persona
prompt and all retrieved legal text were absent.

The cause is not the harness. `run_eval.py` sends a byte-identical `system_message` to
every Ollama model through the same `/api/chat` call. `starcoder2:3b` is a **base model
whose Ollama chat template has no system-role slot**, so Ollama silently discards the
system message instead of erroring. The harness recorded a successful HTTP 200 for all
26 questions, which is why this passed unnoticed at run time.

The resulting output is degenerate rather than merely wrong:

```
Q: Can the officer ask for my RC?
A: "\n\n- [x] 5739638\n- [ ] 12345\n\n"

Q: Can my driving licence be seized on the spot?
A: "The answer is yes. Anyone can. It happens 10 times a year, around 8 hours
    per year. The official statistics are based on UK drivers' licences..."
```

The second answer invents UK statistics for a question about Haryana law — the
signature of a model working from parametric memory with no context at all.

Consequently:

- Its `hallucination_rate` of 1.000 rests on **5 checked claims total** (against 40–50
  for the others). It is not a measurement of hallucination under RAG; the model rarely
  emitted anything checkable.
- Its `0.000` correctness measures *"a base model with no context"*, not *"a model
  inside this application's pipeline"*.
- Its 6,554 ms latency is **not** comparable — it processed ~22 prompt tokens while the
  others processed ~2,700. Its apparent speed advantage over llama3.1:8b is an artefact
  of doing ~1 % of the prompt work.

**This is still a real finding, correctly labelled.** It shows that *chat-template
compatibility is a deployment constraint independent of model quality*: a model can be
installed, respond with HTTP 200, and produce fluent text while silently ignoring your
entire RAG pipeline. A system that only checked for errors would have shipped this. It
is reported here as a **failed experimental arm plus an infrastructure finding**, not
as evidence that StarCoder2 is a poor legal-reasoning model — this run does not support
that claim either way.

The ranking in §4 is therefore a **three-model comparison**.

---

## 4. Answering the Exercise 4 questions

### Which model is most accurate?

**Gemini 3.5 Flash Lite** — 7/21 (0.333), against llama3.1:8b's 5/21 (0.238) and
codellama:7b's 2/21 (0.095). Gemini is 1.4× llama3.1 and 3.5× codellama.

### Which hallucinates least?

**Gemini**, decisively — 0.075 vs 0.220 (llama3.1) and 0.500 (codellama). Gemini leaves
roughly 1 claim in 13 unverified; codellama leaves **1 in 2**. Because this is the same
`grounding.py` the live app uses, this number is directly meaningful: it is the rate at
which the app's own checker would flag that model's output to a citizen.

### Which gives better retrieval-based responses?

Gemini again, and the margin is understated by the table. All three context-receiving
models got the *same* context at ~2,500–2,970 prompt tokens, so this measures how well
each **uses** identical evidence. Gemini converts that evidence into a verified answer
most often (0.333) while inventing least (0.075).

### Which has lower latency?

**Gemini** — 4,231 ms mean, 3.4× faster than llama3.1:8b and 5.2× faster than
codellama:7b. The p95 gap is wider still: 8,981 ms vs 33,618 ms for codellama, i.e.
codellama's tail is **3.7×** worse. For a roadside-stop app, tail latency is the number
that matters; a citizen holding a phone in front of an officer experiences p95, not the
mean.

### Which needs fewest computational resources?

**Unanswerable from this data, and it would be dishonest to rank it.** See §1: the
CPU/RAM/GPU columns are whole-host snapshots with a 3.0–4.8 % CPU spread. The only
defensible statement is qualitative: Gemini consumes no local compute at all because
inference happens off-device — which is a genuine architectural difference, not a
measured one.

### Is the most accurate model also the most efficient?

**Yes — and that is the most interesting result in this table.** Gemini is
simultaneously the most accurate (0.333), the least hallucinating (0.075), and the
fastest (4,231 ms). The textbook quality-vs-latency trade-off **does not appear** among
these models.

### So is there a trade-off at all?

Yes, but not on the axis the question anticipates. Two real ones:

**(a) Local/private/free vs. hosted/fast/accurate.** Gemini wins every measured quality
and latency metric, but it is a hosted API: it requires network access at the roadside,
sends a citizen's legal question to a third party, and is subject to quota (the harness
needs a 5-second inter-call spacing, `_GEMINI_MIN_INTERVAL_S`, purely to respect
free-tier limits). llama3.1:8b is 3.4× slower and hallucinates 2.9× more, but runs
entirely offline and privately. **For this specific application — a citizen possibly
with poor connectivity, asking a legally sensitive question — that is not obviously the
wrong trade.** The correct engineering answer is the dual-provider design the app
already implements, not a single winner.

**(b) Specialisation is a cost, not a bonus.** codellama:7b is **strictly dominated** by
llama3.1:8b: slower (21,822 vs 14,487 ms), worse tail (33,618 vs 18,288 ms), 2.3× the
hallucination (0.500 vs 0.220), and 40 % of the correctness (0.095 vs 0.238). There is
no axis on which it is preferable. A code-tuned model applied to statutory prose is
worse than a general instruction-tuned model of comparable size — the specialisation
actively hurts. This is worth stating plainly given the brief requires Code Llama in the
pipeline: **the required model is measurably the wrong tool for this use case**, and
demonstrating that with evidence is a more useful result than pretending otherwise.

---

## 5. Retrieval is the ceiling, not generation

The strongest system-level finding, and it is not visible from any single model row.

Correctness requires citing ≥1 expected section. If retrieval never surfaced that
section, no model can satisfy the condition. Corrected retrieval **hit rate is 0.632**, so
**0.632 is a hard upper bound on correctness for every model**.

Gemini scores 0.333 — **52.7 % of the achievable ceiling**. Read together:

- ~37 % of scorable questions are lost **before generation starts**, identically for all
  four models;
- of the ~63 % that remain reachable, the best model converts about half.

So roughly half of the remaining headroom is a generation problem and half is a
retrieval problem. Swapping in a better model cannot lift the system past 0.632;
**improving retrieval raises the ceiling for all models at once.** If there is one place
to spend the next unit of effort, the retrieval figures say it is retrieval — not model
selection. Exercise 5's `rag_pipeline_analysis.md` traces the specific retrieval failures
behind that gap, and its Case 2 (the helmet fine losing its slot at 0.6368 against a
~0.638 cutoff) is exactly what a lost ceiling point looks like in practice.

The 5 "open" questions are excluded from the denominator rather than scored zero, because
they have no verified ground-truth section. Scoring them 0 would understate every model
equally; excluding them keeps the denominator honest at 21.

---

## 6. Where the relevance metric misleads

`analyze.py` flags `relevance` as a weak lexical-overlap proxy and instructs the write-up
to spot-check by hand rather than trust it. Done, and **it should be disregarded as a
quality signal.**

It inverts the ranking: llama3.1:8b scores 0.803 and Gemini 0.470, the reverse of every
other quality metric. The cause is not verbosity. The metric is
`|question_words ∩ answer_words| / |question_words|` (`analyze.py:78-84`) — the denominator
is the *question's* words, which is fixed, so a longer answer can only ever match **more**
of them. Length cannot lower the score.

Breaking the per-answer scores down by whether the model refused:

| Model | mean | refusals | mean on refusals | non-refusals | mean on non-refusals |
|---|---|---|---|---|---|
| llama3.1:8b | 0.803 | **0** | — | 26 | 0.803 |
| gemini-3.5-flash-lite | 0.470 | **11** | 0.124 | 14 | **0.741** |
| codellama:7b | 0.784 | 1 | 0.667 | 25 | 0.789 |

**All 8 of Gemini's zero-relevance answers are refusals.** A correct refusal — "I don't
have an official Haryana/Indian source that confirms this, so I can't answer reliably" —
contains none of the question's content words, so it scores exactly 0.000. Gemini's mean is
dragged from 0.741 down to 0.470 entirely by refusing 11 times. llama3.1 **never refuses
once across all 26 questions**, including the two `out_of_scope` questions where refusing is
the only correct behaviour — and is rewarded for it by this metric.

So `relevance` does not measure relevance. It measures *willingness to answer*, and it
**penalises the single safety behaviour this application exists to provide**: declining to
state law it cannot source. The same Gemini refusal that earns a correctness `pass` on Q20
earns a relevance `0.000`. Compared like-for-like on non-refusal answers only, the gap
nearly disappears (0.741 vs 0.803).

Conclusion: `relevance` is actively misleading for this application, not merely weak, and no
claim in this document rests on it. Retaining and refuting it is deliberate — it documents
which of our own metrics survived scrutiny. A replacement must either exclude refusals from
the denominator or score them against the *expected* behaviour rather than the question text.

---

## 7. Conclusion

On quantitative evidence, across 26 questions and identical retrieval:

1. **Gemini 3.5 Flash Lite is the best-performing model on every metric that survived
   scrutiny** — correctness 0.333, hallucination 0.075, latency 4,231 ms.
2. **llama3.1:8b is the best local model** and the right default for an offline,
   privacy-preserving deployment: 0.238 correctness at 2.9× Gemini's hallucination rate,
   but with zero network dependency. The app's dual-provider design is the correct
   response to this, and is justified by these numbers rather than by preference.
3. **codellama:7b should not be used for this application.** It is dominated by
   llama3.1:8b on latency, tail latency, hallucination and correctness simultaneously.
4. **The StarCoder2 arm is void** and yields an infrastructure finding instead: a base
   model without a system-role template will silently discard an entire RAG context while
   still returning HTTP 200.
5. **Retrieval, not model choice, is the binding constraint.** The 0.632 retrieval hit
   rate caps every model, and the best model reaches only 52.7 % of it.

The headline is that the anticipated quality-vs-speed trade-off is absent here; the real
trade-off is architectural — local privacy against hosted accuracy — and the largest
available gain is in retrieval, where a fix benefits all four models at once.

---

## 8. What a re-run should fix

Recorded so the limitations above are actionable rather than merely admitted:

1. **Re-run Q9** to restore 27/27. The Retrieval 500 was the ChromaDB telemetry race
   fixed in commit `1a725ce`; it should no longer reproduce.
2. **Assert prompt length per arm.** A single check that `prompt_tokens > 500` would have
   caught the StarCoder2 failure at run time. This is the highest-value fix.
3. **Fix the retrieval-quality denominator.** `analyze.py:119` skips questions whose
   `expected_sections` is `None`, but the two `expect_refusal` questions (Q19, Q20) use an
   empty **list**, which is not `None`. They fall through to the `if exp else 1.0` branch
   and are awarded a free `recall = 1.0` / `hit = True` — for questions where there is no
   correct section to retrieve at all. Corrected, retrieval quality is
   **mean_recall 0.553 / hit_rate 0.632 (n=19)**, not 0.595 / 0.667 (n=21). Fix is
   `if not expected: continue`. This document already uses the corrected figures
   throughout; §5's ceiling argument is unaffected in substance (Gemini's 0.333 is 52.7 %
   of the ceiling rather than exactly half). Re-running a fixed `analyze.py` will bring
   `metrics_report.json` into line with what is written here.
4. **Fix the p95 index.** `analyze.py:167` uses `int(n * 0.95) - 1`, which for n=26 is
   index 23 — the 92.3rd percentile, not the 95th. Every p95 in this document is therefore
   slightly low (llama3.1's true p95 is 19,516 ms, reported as 18,288 ms). The relative
   ranking between models is unaffected. Use `math.ceil(0.95 * n) - 1`.
3. **Replace StarCoder2** with an instruction-tuned model of similar size (e.g.
   `mistral:7b-instruct`, `qwen2.5:3b-instruct`) to obtain a genuine fourth arm, or drive
   it through `/api/generate` with the context folded into the user prompt.
4. **Sample per-process resource usage**, not whole-host, if the resource comparison is to
   mean anything.
5. **Replace lexical-overlap relevance** with a rubric or judge-scored measure, or drop the
   metric.
