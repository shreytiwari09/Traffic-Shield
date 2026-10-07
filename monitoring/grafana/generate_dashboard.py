"""Generates monitoring/grafana/dashboards/trafficshield.json.

The dashboard is code, not hand-clicked JSON: edit this file and re-run
    python monitoring/grafana/generate_dashboard.py
so panel changes are reviewable in a diff (Grafana re-reads it within ~10s).
"""

import json
from pathlib import Path

DS = {"type": "prometheus", "uid": "prometheus"}
panels = []
pid = [0]


def nid():
    pid[0] += 1
    return pid[0]


def row(title, y):
    panels.append({"type": "row", "title": title, "id": nid(), "collapsed": False,
                   "gridPos": {"h": 1, "w": 24, "x": 0, "y": y}, "panels": []})


def stat(title, expr, x, y, w=4, unit="none", decimals=None, thresholds=None, desc="", mappings=None):
    fc = {"unit": unit, "color": {"mode": "thresholds"},
          "thresholds": {"mode": "absolute", "steps": thresholds or [{"color": "text", "value": None}]}}
    if decimals is not None:
        fc["decimals"] = decimals
    if mappings:
        fc["mappings"] = mappings
    panels.append({"type": "stat", "title": title, "id": nid(), "datasource": DS, "description": desc,
                   "gridPos": {"h": 4, "w": w, "x": x, "y": y},
                   "fieldConfig": {"defaults": fc, "overrides": []},
                   "options": {"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                               "colorMode": "value", "graphMode": "none", "textMode": "value",
                               "justifyMode": "center"},
                   "targets": [{"datasource": DS, "expr": expr, "refId": "A", "instant": True}]})


def ts(title, targets, x, y, w=12, h=8, unit="none", desc="", stack=False, bars=False, overrides=None,
       minv=None, maxv=None):
    custom = {"lineWidth": 2, "fillOpacity": 80 if bars else 0, "showPoints": "never", "spanNulls": True,
              "drawStyle": "bars" if bars else "line",
              "stacking": {"mode": "normal" if stack else "none", "group": "A"}, "axisSoftMin": 0}
    d = {"unit": unit, "custom": custom, "color": {"mode": "palette-classic"}, "noValue": "0"}
    if minv is not None:
        d["min"] = minv
    if maxv is not None:
        d["max"] = maxv
    panels.append({"type": "timeseries", "title": title, "id": nid(), "datasource": DS, "description": desc,
                   **({"interval": "1m"} if bars else {}),
                   "gridPos": {"h": h, "w": w, "x": x, "y": y},
                   "fieldConfig": {"defaults": d, "overrides": overrides or []},
                   "options": {"legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
                               "tooltip": {"mode": "multi", "sort": "desc"}},
                   "targets": [{"datasource": DS, "expr": e, "legendFormat": lg, "refId": chr(65 + i)}
                               for i, (e, lg) in enumerate(targets)]})


def fixed(name, color):
    return {"matcher": {"id": "byName", "options": name},
            "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": color}}]}


GOOD, BAD, WARN = "green", "red", "orange"
HEALTH = 'route!~"/v1/health|/health"'

# Row 1: headline KPIs
row("Health & headline AI KPIs (selected time range)", 0)
stat("Services up", 'sum(max_over_time(up{job="trafficshield"}[30s])) or vector(0)', 0, 1,
     thresholds=[{"color": BAD, "value": None}, {"color": WARN, "value": 1}, {"color": GOOD, "value": 5}],
     desc="Services answering Prometheus scrapes, out of 5.")
stat("Answers served", "sum(increase(ts_answer_confidence_total[$__range])) or vector(0)", 4, 1, decimals=0,
     desc="Questions that reached the LLM and produced an answer.")
stat("Hallucination rate",
     'sum(increase(ts_grounding_claims_total{result="unverified"}[$__range])) '
     "/ clamp_min(sum(increase(ts_grounding_claims_total[$__range])), 1)",
     8, 1, unit="percentunit", decimals=1,
     thresholds=[{"color": GOOD, "value": None}, {"color": WARN, "value": 0.25}, {"color": BAD, "value": 0.35}],
     desc="Share of section numbers / rupee amounts in answers NOT found in the retrieved law. "
          "Offline baseline for llama3.1:8b: 22.5%.")
stat("High-confidence answers",
     'sum(increase(ts_answer_confidence_total{level="high"}[$__range])) '
     "/ clamp_min(sum(increase(ts_answer_confidence_total[$__range])), 1)",
     12, 1, unit="percentunit", decimals=0,
     desc="Answers backed by a strong vector match AND graph corroboration.")
stat("Guardrail blocks", 'sum(increase(ts_guardrail_decisions_total{outcome="blocked"}[$__range])) or vector(0)',
     16, 1, decimals=0, thresholds=[{"color": "text", "value": None}, {"color": WARN, "value": 1}],
     desc="Prompt-injection / illegal-conduct requests refused before reaching retrieval or the LLM.")
# 1 = Neo4j, 0 = JSON by configuration, 2 = JSON because Neo4j was unreachable.
stat("Graph backend", "max(ts_graph_backend_neo4j) + 2 * max(ts_graph_backend_fell_back)", 20, 1,
     thresholds=[{"color": WARN, "value": None}, {"color": GOOD, "value": 1}, {"color": BAD, "value": 2}],
     mappings=[{"type": "value", "options": {"0": {"text": "JSON"}, "1": {"text": "Neo4j"},
                                             "2": {"text": "JSON fallback"}}}],
     desc="Which graph store retrieval is using: Neo4j, the JSON store by configuration, "
          "or the JSON store because Neo4j was unreachable (red).")

# Row 2: service golden signals
row("Service golden signals", 5)
ts("Request rate by service",
   [(f"sum by (service) (rate(ts_http_requests_total{{{HEALTH}}}[1m]))", "{{service}}")],
   0, 6, unit="reqps", desc="Health-check and scrape traffic excluded so real user requests are visible.")
ts("p95 request latency by service",
   [(f"histogram_quantile(0.95, sum by (le, service) (rate(ts_http_request_duration_seconds_bucket{{{HEALTH}}}[5m])))",
     "{{service}}")],
   12, 6, unit="s", desc="Orchestration/app latency is dominated by LLM generation (CPU-only Ollama).")

# Row 3: AI pipeline performance
row("AI pipeline performance", 14)
ts("LLM generation latency (p50 / p95) by model", [
    ('histogram_quantile(0.50, sum by (le, model) (rate(ts_llm_generation_seconds_bucket{status="ok"}[15m])))',
     "p50 {{model}}"),
    ('histogram_quantile(0.95, sum by (le, model) (rate(ts_llm_generation_seconds_bucket{status="ok"}[15m])))',
     "p95 {{model}}"),
], 0, 15, unit="s")
ts("Retrieval stage latency (p95)",
   [("histogram_quantile(0.95, sum by (le, stage) (rate(ts_retrieval_stage_seconds_bucket[15m])))", "{{stage}}")],
   12, 15, unit="s",
   desc="embed = Ollama nomic-embed-text, vector_search = Chroma, graph = Neo4j lookup + candidate scoring.")

# Row 4: AI quality & safety
row("AI quality & safety", 23)
ts("Grounding: verified vs unverified claims", [
    ('sum(increase(ts_grounding_claims_total{result="verified"}[1m]))', "verified"),
    ('sum(increase(ts_grounding_claims_total{result="unverified"}[1m]))', "unverified"),
], 0, 24, w=8, bars=True, stack=True, overrides=[fixed("verified", GOOD), fixed("unverified", BAD)],
   desc="Each section/rupee claim in an answer, checked against the retrieved legal text.")
ts("Answer confidence", [("sum by (level) (increase(ts_answer_confidence_total[1m]))", "{{level}}")],
   8, 24, w=8, bars=True, stack=True,
   overrides=[fixed("high", GOOD), fixed("medium", "yellow"), fixed("low", WARN), fixed("none", BAD)])
ts("Guardrail decisions", [("sum by (outcome) (increase(ts_guardrail_decisions_total[1m]))", "{{outcome}}")],
   16, 24, w=8, bars=True, stack=True,
   overrides=[fixed("allowed", GOOD), fixed("redacted", "blue"), fixed("blocked", BAD)],
   desc="allowed = clean question; redacted = PII (Aadhaar/phone/plate/email) removed; "
        "blocked = injection or illegal request.")

# Row 5: drift & errors
row("Drift & errors", 32)
ts("Retrieval top similarity score (p10 / median)", [
    ("histogram_quantile(0.50, sum by (le) (rate(ts_retrieval_top_score_bucket[15m])))", "median"),
    ("histogram_quantile(0.10, sum by (le) (rate(ts_retrieval_top_score_bucket[15m])))", "p10 (worst queries)"),
], 0, 33, minv=0, maxv=1,
   desc="Best cosine similarity per query. A falling trend means users ask things the legal corpus "
        "does not cover (query drift).")
ts("Error rate (5xx) by service",
   [('sum by (service) (rate(ts_http_requests_total{status=~"5.."}[5m])) '
     "/ clamp_min(sum by (service) (rate(ts_http_requests_total[5m])), 1e-9)", "{{service}}")],
   12, 33, unit="percentunit", minv=0)

# Row 6: LLMOps — prompt canary, cost, feedback
row("LLMOps: prompt versions, canary, cost & user feedback", 41)
stat("Tokens used", "sum(increase(ts_llm_tokens_total[$__range])) or vector(0)", 0, 42, decimals=0, unit="short",
     desc="Prompt + completion tokens reported by the providers over the selected range.")
stat("LLM API cost", "sum(increase(ts_llm_cost_usd_total[$__range])) or vector(0)", 4, 42, unit="currencyUSD",
     decimals=4, desc="Gemini spend from real token counts x the per-1M prices in settings (estimates until "
                      "set to your plan). Local Ollama models cost $0 in API terms.")
stat("User satisfaction",
     'sum(increase(ts_user_feedback_total{rating="up"}[$__range])) '
     "/ clamp_min(sum(increase(ts_user_feedback_total[$__range])), 1)",
     8, 42, unit="percentunit", decimals=0,
     thresholds=[{"color": BAD, "value": None}, {"color": WARN, "value": 0.6}, {"color": GOOD, "value": 0.8}],
     desc="Share of thumbs-up among all citizen ratings. Thumbs-down answers feed evaluation/feedback_to_eval.py.")
stat("Feedback received", "sum(increase(ts_user_feedback_total[$__range])) or vector(0)", 12, 42, decimals=0)
stat("Canary traffic share",
     'sum(increase(ts_prompt_routing_total{arm="canary"}[$__range])) '
     "/ clamp_min(sum(increase(ts_prompt_routing_total[$__range])), 1)",
     16, 42, unit="percentunit", decimals=0,
     desc="Share of Ask requests routed to the candidate prompt (CANARY_PERCENT).")
stat("Answers by prompt version", "count(count by (prompt_version) (increase(ts_prompt_routing_total[$__range]) > 0))",
     20, 42, decimals=0, desc="How many prompt versions served traffic in the range (2 = canary live).")

ts("Hallucination rate: stable vs canary prompt", [
    ('sum by (prompt_version) (increase(ts_grounding_claims_total{result="unverified", prompt_version!=""}[15m])) '
     '/ clamp_min(sum by (prompt_version) (increase(ts_grounding_claims_total{prompt_version!=""}[15m])), 1)', "{{prompt_version}}")],
   0, 46, w=8, unit="percentunit", minv=0, maxv=1,
   desc="The canary decision metric: promote the candidate prompt only if this is not worse than stable.")
ts("High-confidence share by prompt version", [
    ('sum by (prompt_version) (increase(ts_answer_confidence_total{level="high", prompt_version!=""}[15m])) '
     '/ clamp_min(sum by (prompt_version) (increase(ts_answer_confidence_total{prompt_version!=""}[15m])), 1)', "{{prompt_version}}")],
   8, 46, w=8, unit="percentunit", minv=0, maxv=1)
ts("User feedback by prompt version", [
    ('sum by (prompt_version, rating) (increase(ts_user_feedback_total[1m]))', "{{prompt_version}} {{rating}}")],
   16, 46, w=8, bars=True, stack=True)
ts("Tokens per minute by model", [
    ("sum by (model, type) (rate(ts_llm_tokens_total[5m])) * 60", "{{model}} {{type}}")],
   0, 54, unit="short", desc="Prompt tokens dominate: ~2k tokens of retrieved law per question.")
ts("LLM API cost per hour by model", [
    ("sum by (model) (rate(ts_llm_cost_usd_total[15m])) * 3600", "{{model}}")],
   12, 54, unit="currencyUSD")

# Row 7: scheduled online evaluation (evaluation/scheduled_eval.py)
row("Scheduled online evaluation (latest run)", 62)
for i, (metric_name, title, unit, good_high) in enumerate([
    ("context_precision", "Context precision", "percentunit", True),
    ("context_recall", "Context recall", "percentunit", True),
    ("mrr", "MRR", "none", True),
    ("hallucination_rate", "Hallucination rate", "percentunit", False),
    ("test_pass_rate", "Test pass rate", "percentunit", True),
]):
    stat(title, f'max(ts_eval_score{{metric="{metric_name}"}})', i * 4, 63, unit=unit, decimals=2,
         desc="From the latest line of evaluation/eval_history.jsonl (real questions through the live pipeline).")
stat("Last eval run", "time() - max(ts_eval_last_run_timestamp_seconds)", 20, 63, unit="s", decimals=0,
     thresholds=[{"color": GOOD, "value": None}, {"color": WARN, "value": 86400 * 1.5},
                 {"color": BAD, "value": 86400 * 3}],
     desc="Seconds since the last scheduled eval (nightly via scripts/register_nightly_eval.ps1).")

dash = {"title": "Traffic Shield — AI Ops", "uid": "trafficshield-aiops", "schemaVersion": 39, "version": 1,
        "editable": True, "refresh": "5s", "time": {"from": "now-30m", "to": "now"},
        "tags": ["trafficshield", "llmops"], "timezone": "browser", "panels": panels,
        "templating": {"list": []}, "annotations": {"list": []}}
out = Path(__file__).resolve().parent / "dashboards" / "trafficshield.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(dash, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(len(panels), "panels")
