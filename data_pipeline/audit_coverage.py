"""
Coverage audit — can the dataset actually answer a driver's rights questions?

Phase 5 checks that records are well-formed; it says nothing about whether the
corpus contains the law a driver stopped at a checkpoint would need. This
audit asks that question directly: for each real traffic-stop scenario it
retrieves from dataset.jsonl with plain TF-IDF (no embeddings, so it runs
without Ollama) and reports whether the provisions that ought to govern the
scenario are present and substantive.

    python -m data_pipeline.audit_coverage
    python -m data_pipeline.audit_coverage --verbose   # show retrieved snippets

Expected sections are named per scenario, so a regression in Phase 4 that
silently loses a provision shows up here as a FAIL rather than as a quietly
worse answer at query time.
"""

import argparse
import math
import re
from collections import Counter

from data_pipeline.config import DATASET_PATH
from data_pipeline.text_utils import read_jsonl

_TOKEN = re.compile(r"[a-z]+")

# Each scenario names the provisions a correct answer must be able to cite.
# "expect" entries are record ids; a scenario passes when every one is present
# in the dataset with substantive text.
SCENARIOS: list[dict] = [
    {
        "question": "A police officer has asked me to produce my driving licence. Do I have to?",
        "expect": ["HR_MVA_130", "HR_MVA_158"],
    },
    {
        "question": "Can the officer keep or impound my licence and documents?",
        "expect": ["HR_MVA_206", "HR_MVA_19"],
    },
    {
        "question": "I do not have the papers with me right now. How long do I get to produce them?",
        "expect": ["HR_MVA_158", "HR_MVA_130"],
    },
    {
        "question": "The officer wants me to pay a fine on the spot in cash. Is that lawful?",
        "expect": ["HR_MVA_200"],
    },
    {
        "question": "I was stopped for not wearing a helmet on my motorcycle.",
        "expect": ["HR_MVA_129", "HR_MVA_194D"],
    },
    {
        "question": "I was challaned for not wearing a seat belt.",
        "expect": ["HR_MVA_194B"],
    },
    {
        "question": "The officer wants me to take a breath test for alcohol.",
        "expect": ["HR_MVA_185", "HR_MVA_203"],
    },
    {
        "question": "My vehicle is being towed away for wrong parking.",
        "expect": ["HR_MVA_127"],
    },
    {
        "question": "I am accused of driving over the speed limit.",
        "expect": ["HR_MVA_112", "HR_MVA_183"],
    },
    {
        "question": "Who has the power to stop my vehicle and demand it be weighed?",
        "expect": ["HR_MVA_132", "HR_MVA_114"],
    },
    {
        "question": "I am being penalised for driving without a valid licence.",
        "expect": ["HR_MVA_3", "HR_MVA_181"],
    },
    {
        "question": "I was stopped for driving without insurance.",
        "expect": ["HR_MVA_146", "HR_MVA_196"],
    },
    {
        "question": "What is the general penalty if no specific penalty is prescribed?",
        "expect": ["HR_MVA_177"],
    },
    {
        "question": "Can I appeal against the licensing authority's order in Haryana?",
        "expect": ["HR_HMVR_8"],
    },
    {
        "question": "What are a driver's duties at a traffic signal and towards other road users?",
        "expect": ["HR_MVDR_3", "HR_MVDR_12"],
    },
    {
        "question": "The officer says my vehicle documents must be carried in original.",
        "expect": ["HR_MVA_130", "HR_MVA_158"],
        # CMVR rule 139 is the subordinate rule on the same point. It is not
        # extractable: the source PDF prints it mid-line with its title
        # duplicated inside a footnote bracket and no delimiter before the
        # body, so no section grammar can anchor on it. The MV Act sections
        # above are the operative provisions and are present in full.
        "known_gaps": {
            "HR_CMVR_139": "mangled in source PDF — duplicated title, no title/body delimiter",
        },
    },
]

# A provision can be genuinely brief; this floor only catches truncation.
MIN_SUBSTANTIVE_CHARS = 140


def build_index(records: list[dict]):
    """Document-frequency table over the corpus."""
    doc_freq: Counter = Counter()
    tokenised: list[set[str]] = []
    for record in records:
        terms = set(_TOKEN.findall(f"{record['title']} {record['content']}".lower()))
        tokenised.append(terms)
        doc_freq.update(terms)
    return doc_freq, tokenised


def search(query: str, records: list[dict], doc_freq: Counter, top_k: int = 5) -> list[tuple[float, dict]]:
    """Rank records against *query* with TF-IDF, weighting title matches."""
    total = len(records)
    query_terms = [t for t in _TOKEN.findall(query.lower()) if len(t) > 2]

    scored: list[tuple[float, dict]] = []
    for record in records:
        title = record["title"].lower()
        body = record["content"].lower()
        body_counts = Counter(_TOKEN.findall(body))

        score = 0.0
        for term in query_terms:
            frequency = body_counts.get(term, 0)
            if not frequency and term not in title:
                continue
            idf = math.log((total + 1) / (doc_freq.get(term, 0) + 1)) + 1
            score += idf * (1 + math.log(frequency)) if frequency else 0.0
            if term in title:
                score += idf * 2.0  # a title hit is a strong signal in statute
        if score:
            scored.append((score, record))

    scored.sort(key=lambda pair: -pair[0])
    return scored[:top_k]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit rights-scenario coverage")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print the retrieved sections for each scenario")
    args = parser.parse_args(argv)

    records = list(read_jsonl(DATASET_PATH))
    by_id = {r["id"]: r for r in records}
    doc_freq, _ = build_index(records)

    print(f"Corpus: {len(records)} sections from "
          f"{len({r['source_pdf'] for r in records})} official documents\n")

    passed = 0
    retrieval_hits = 0
    known_gaps: dict[str, str] = {}
    for scenario in SCENARIOS:
        expected = scenario["expect"]
        known_gaps.update(scenario.get("known_gaps", {}))
        missing = [sid for sid in expected if sid not in by_id]
        thin = [
            sid for sid in expected
            if sid in by_id and len(by_id[sid]["content"]) < MIN_SUBSTANTIVE_CHARS
        ]

        results = search(scenario["question"], records, doc_freq)
        retrieved_ids = [r["id"] for _, r in results]
        top_hit = any(sid in retrieved_ids for sid in expected)
        retrieval_hits += bool(top_hit)

        ok = not missing and not thin
        passed += ok
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {scenario['question']}")

        for sid in expected:
            record = by_id.get(sid)
            if not record:
                print(f"         MISSING  {sid}")
            else:
                mark = "!" if len(record["content"]) < MIN_SUBSTANTIVE_CHARS else " "
                found = "retrieved" if sid in retrieved_ids else "not in top-5"
                print(f"        {mark} {sid:14} p{str(record['page']):>4} "
                      f"{record['title'][:48]:50} {len(record['content']):>5} chars  ({found})")

        for sid, reason in scenario.get("known_gaps", {}).items():
            print(f"         GAP      {sid:14} not extractable — {reason}")

        if args.verbose:
            print("         top TF-IDF matches:")
            for score, record in results:
                snippet = " ".join(record["content"].split())[:110]
                print(f"           {score:6.1f}  {record['id']:14} {record['title'][:40]:42} {snippet}…")
        print()

    print("=" * 78)
    print(f"Scenarios with all expected provisions present: {passed}/{len(SCENARIOS)}")
    print(f"Scenarios where a keyword search surfaces one in the top 5: "
          f"{retrieval_hits}/{len(SCENARIOS)}")
    if known_gaps:
        print(f"\nKnown unextractable provisions ({len(known_gaps)}):")
        for sid, reason in known_gaps.items():
            print(f"  {sid}: {reason}")
    print("\nNote: the second figure uses TF-IDF only. Phase 6 embeddings are what")
    print("the application actually retrieves with; this is a floor, not a ceiling.")
    return 0 if passed == len(SCENARIOS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
