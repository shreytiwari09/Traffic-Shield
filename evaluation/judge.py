"""
LLM-as-a-judge implementation of the two RAGAS generation-quality metrics.

METHOD AND ATTRIBUTION
The prompts and the scoring arithmetic below follow the RAGAS framework:

    Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2024).
    "RAGAS: Automated Evaluation of Retrieval Augmented Generation."
    Proceedings of EACL 2024 (System Demonstrations). arXiv:2309.15217.

Two metrics, each a two-step prompt chain rather than a single "rate this
0-10" call. That decomposition is the entire point of the method: asking a
model for a holistic quality score produces an unreliable, uncalibrated
number, whereas asking it to perform many small, verifiable, near-binary
judgements produces a score you can audit claim by claim.

  FAITHFULNESS (hallucination)
    Step 1 — break the answer into atomic, pronoun-free statements.
    Step 2 — NLI verdict per statement: can it be inferred from the context
             ALONE (1) or not (0)?
    Score  = supported statements / total statements.

  ANSWER RELEVANCE
    Step 1 — reverse-engineer 3 questions the answer actually resolves, and
             flag whether the answer is noncommittal (evasive/refusing).
    Step 2 — mean cosine similarity between the original question's embedding
             and those 3 generated questions' embeddings.
    Score  = 0.0 when noncommittal, else that mean similarity.

WHY THIS REPLACES WHAT WAS HERE BEFORE
`analyze.py` previously scored "relevance" as content-word overlap between
question and answer. That metric was actively misleading on this project's own
data: Gemini scored 0.47 while codellama scored 0.784, even though Gemini had
the lowest hallucination rate (0.075 vs 0.50) and the highest pass rate. Word
overlap rewards restating the question and rambling. It is retained in the
report as `relevance_lexical_overlap_legacy` purely so the two can be compared,
never as the headline number.

Faithfulness here also does something `grounding.py` structurally cannot.
grounding.py regex-matches section numbers and rupee amounts, so it is blind to
an invented legal *duty* that cites no number — a real failure mode for this
app. Both are reported: grounding.py stays the fast, deterministic in-app
guardrail; this is the offline, semantic measurement. They answer different
questions and disagreeing is informative, not a bug.

A NOTE ON THE JUDGE
The judge must be a different, stronger model than the ones under test — a
model grading its own output is not an evaluation. Judged answers are cached on
disk by content hash, because a full sweep is ~27 questions x 4 models x 3
judge calls and free-tier quota will not survive being spent twice on
identical inputs.
"""

import asyncio
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.shared.settings import settings  # noqa: E402

CACHE_PATH = Path(__file__).parent / "judge_cache.json"

# ---------------------------------------------------------------------------
# Prompts — reproduced from the RAGAS framework (Es et al., 2024, arXiv:2309.15217)
# ---------------------------------------------------------------------------

STATEMENT_EXTRACTION_SYSTEM = """You are an expert natural language processing assistant. Your task is to break down the provided answer into one or more fully understandable atomic statements.

Strict Rules:
1. Ensure that no pronouns are used in any statement; replace them with the actual entities they refer to.
2. Each statement must contain only one single factual claim.
3. Format the outputs strictly in JSON."""

STATEMENT_EXTRACTION_USER = """Question: {question}
Answer: {answer}

Extract the statements from the Answer and output them in the following JSON format:
{{
  "statements": [
    "statement 1",
    "statement 2"
  ]
}}"""

NLI_VERIFICATION_SYSTEM = """You are an expert fact-checker. Your task is to verify whether a given set of statements can be logically inferred from the provided context.

Strict Rules:
1. For each statement, return a verdict of 1 if the statement can be directly inferred based ONLY on the context.
2. Return a verdict of 0 if the statement cannot be directly inferred, is an exaggeration, or introduces external knowledge.
3. Do not use outside knowledge.
4. Format the outputs strictly in JSON."""

NLI_VERIFICATION_USER = """Context:
{context}

Statements:
{statements}

Evaluate each statement and output in the following JSON format:
{{
  "evaluations": [
    {{
      "statement": "the exact statement text",
      "reason": "brief explanation of why it is supported or not",
      "verdict": 1 or 0
    }}
  ]
}}"""

REVERSE_QUESTION_SYSTEM = """You are an expert system designed to reverse-engineer questions. Your task is to generate a diverse set of questions that the provided answer is attempting to resolve.

Additionally, identify if the answer is noncommittal. A noncommittal answer is one that is evasive, vague, ambiguous, or states that it cannot answer.

Strict Rules:
1. Generate exactly 3 distinct questions.
2. Output a "noncommittal" score: 1 if the answer is evasive/cannot answer, and 0 if the answer is committal and attempts to provide facts.
3. Format the outputs strictly in JSON."""

REVERSE_QUESTION_USER = """Answer:
{answer}

Generate 3 questions and determine the noncommittal status. Output in the following JSON format:
{{
  "generated_questions": [
    "question 1",
    "question 2",
    "question 3"
  ],
  "noncommittal": 0 or 1
}}"""


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
def _load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


_cache = _load_cache()


def _save_cache() -> None:
    CACHE_PATH.write_text(json.dumps(_cache, indent=2), encoding="utf-8")


def _key(*parts: str) -> str:
    return hashlib.sha256("␟".join(parts).encode("utf-8")).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Judge transport
# ---------------------------------------------------------------------------
class JudgeUnavailable(RuntimeError):
    pass


_gemini_lock = asyncio.Lock()
_last_gemini_call = 0.0


def _extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a judge reply.

    Models wrap JSON in prose or ```json fences even when told not to, and a
    whole evaluation run should not be lost to that. Returns None rather than
    raising so one unparseable reply degrades to "unscored" instead of
    aborting the sweep.
    """
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    if start == -1:
        return None
    # Scan for the matching brace rather than regexing — nested objects
    # (the "evaluations" array) break any non-recursive pattern.
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


async def _ask_gemini(system: str, user: str) -> str:
    global _last_gemini_call
    import time as _time

    from google import genai
    from google.genai import types

    if not settings.gemini_api_key:
        raise JudgeUnavailable("GEMINI_API_KEY is not set — the judge cannot run.")

    # Serialised with a floor between calls: the judge issues hundreds of
    # requests in a sweep and the free tier rate-limits well before that.
    async with _gemini_lock:
        wait = settings.judge_min_interval_seconds - (_time.perf_counter() - _last_gemini_call)
        if wait > 0:
            await asyncio.sleep(wait)
        client = genai.Client(api_key=settings.gemini_api_key)

        def _call() -> str:
            response = client.models.generate_content(
                model=settings.judge_model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    # Judging is a classification task, not a creative one —
                    # temperature 0 makes a re-run reproduce the same verdicts.
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
            return response.text or ""

        try:
            return await asyncio.to_thread(_call)
        except Exception as exc:
            text = str(exc)
            if "RESOURCE_EXHAUSTED" in text or "429" in text:
                raise JudgeUnavailable(
                    "Judge model quota exhausted. Results so far are cached in "
                    "evaluation/judge_cache.json — rerun later to resume."
                ) from exc
            raise JudgeUnavailable(f"Judge call failed: {text}") from exc
        finally:
            _last_gemini_call = _time.perf_counter()


async def _ask_ollama(system: str, user: str) -> str:
    async with httpx.AsyncClient(timeout=settings.ollama_timeout_seconds) as client:
        r = await client.post(
            f"{settings.ollama_base_url}/api/chat",
            json={
                "model": settings.judge_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.0},
                "keep_alive": settings.ollama_keep_alive,
            },
        )
        if r.status_code != 200:
            raise JudgeUnavailable(f"Judge (ollama) HTTP {r.status_code}: {r.text[:200]}")
        return r.json().get("message", {}).get("content", "")


async def ask_judge(system: str, user: str) -> dict | None:
    cache_key = _key(settings.judge_provider, settings.judge_model, system, user)
    if cache_key in _cache:
        return _cache[cache_key]

    raw = (
        await _ask_gemini(system, user)
        if settings.judge_provider == "gemini"
        else await _ask_ollama(system, user)
    )
    parsed = _extract_json(raw)
    if parsed is not None:
        _cache[cache_key] = parsed
        _save_cache()
    return parsed


# ---------------------------------------------------------------------------
# Embeddings (Answer Relevance step 2)
# ---------------------------------------------------------------------------
async def embed(text: str) -> list[float]:
    """Embeds via the same Ollama nomic-embed-text used by the app's retrieval
    path, so relevance similarity lives in the same vector space the pipeline
    itself retrieves in."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(
            f"{settings.ollama_base_url}/api/embed",
            json={
                "model": settings.ollama_embedding_model,
                "input": [text],
                "keep_alive": settings.ollama_keep_alive,
            },
        )
        r.raise_for_status()
        vectors = r.json().get("embeddings", [])
        return vectors[0] if vectors else []


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# ---------------------------------------------------------------------------
# Metric 1 — Faithfulness
# ---------------------------------------------------------------------------
async def faithfulness(question: str, answer: str, context_chunks: list[str]) -> dict:
    """RAGAS Faithfulness: supported atomic statements / total statements."""
    if not answer or not answer.strip():
        return {"score": None, "reason": "empty answer", "statements": []}
    if not context_chunks:
        return {"score": None, "reason": "no retrieved context to verify against", "statements": []}

    extraction = await ask_judge(
        STATEMENT_EXTRACTION_SYSTEM,
        STATEMENT_EXTRACTION_USER.format(question=question, answer=answer),
    )
    statements = (extraction or {}).get("statements") or []
    if not statements:
        return {"score": None, "reason": "judge extracted no statements", "statements": []}

    verification = await ask_judge(
        NLI_VERIFICATION_SYSTEM,
        NLI_VERIFICATION_USER.format(
            context="\n\n".join(context_chunks),
            statements=json.dumps(statements, indent=2),
        ),
    )
    evaluations = (verification or {}).get("evaluations") or []
    if not evaluations:
        return {"score": None, "reason": "judge returned no verdicts", "statements": statements}

    verdicts = [int(e.get("verdict", 0) or 0) for e in evaluations]
    supported = sum(1 for v in verdicts if v == 1)
    return {
        "score": round(supported / len(verdicts), 3),
        "supported": supported,
        "total": len(verdicts),
        # Kept per-statement so the dashboard's trace inspector can show WHY a
        # score was given. An evaluation number nobody can audit is not much
        # better than no evaluation.
        "evaluations": evaluations,
        "statements": statements,
    }


# ---------------------------------------------------------------------------
# Metric 2 — Answer Relevance
# ---------------------------------------------------------------------------
async def answer_relevance(question: str, answer: str) -> dict:
    """RAGAS Answer Relevance: mean cosine(original question, 3 reverse-
    engineered questions), forced to 0.0 when the answer is noncommittal."""
    if not answer or not answer.strip():
        return {"score": None, "reason": "empty answer"}

    result = await ask_judge(REVERSE_QUESTION_SYSTEM, REVERSE_QUESTION_USER.format(answer=answer))
    if not result:
        return {"score": None, "reason": "judge returned no questions"}

    noncommittal = int(result.get("noncommittal", 0) or 0)
    generated = result.get("generated_questions") or []

    # A refusal is maximally UNRELEVANT under RAGAS by definition. Note that
    # for this app a refusal is often the CORRECT behavior (the persona
    # requires it when no official source supports an answer), so a low
    # relevance score on an out-of-scope question is a success, not a failure.
    # The report keeps `noncommittal` alongside the score so the two are never
    # confused, and analyze.py reports relevance excluding refusals as well.
    if noncommittal == 1:
        return {
            "score": 0.0,
            "noncommittal": 1,
            "generated_questions": generated,
            "reason": "answer is noncommittal (evasive/refusal) — RAGAS defines relevance as 0.0",
        }

    if not generated:
        return {"score": None, "reason": "judge generated no questions"}

    q_vec = await embed(question)
    sims = []
    for gq in generated[:3]:
        sims.append(cosine(q_vec, await embed(gq)))

    return {
        "score": round(sum(sims) / len(sims), 3) if sims else None,
        "noncommittal": 0,
        "generated_questions": generated[:3],
        "similarities": [round(s, 3) for s in sims],
    }
