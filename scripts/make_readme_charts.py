"""
Builds the README figures in assets/charts/ from the committed evaluation
outputs, so every chart is reproducible from data in the repo:

    evaluation/metrics_report.json                     -> model comparison (run 1)
    evaluation/experiments/<latest>_ablation/report.json -> RAG / retrieval ablation (run 2)

Run:  python scripts/make_readme_charts.py      (needs requirements-dev.txt)

Colours are the first categorical slots of a colour-blind-validated palette, in
fixed order; two of them sit below 3:1 contrast on the light surface, so every
bar carries a visible value label and the README gives the same numbers as a
table.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "charts"

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 11, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 1,
    "axes.axisbelow": True, "legend.frameon": False,
})


def _label(ax, x, value, fmt):
    ax.annotate("n/a" if value is None else fmt(value), (x, value or 0), xytext=(0, 4),
                textcoords="offset points", ha="center", va="bottom", fontsize=9.5, color=INK)


def grouped_bars(path: Path, title: str, groups: list[str], series: dict[str, list], subtitle: str = "",
                 percent: bool = True, ymax: float | None = None, notes: dict[str, list] | None = None) -> None:
    """One bar per (group, series); series keep their colour across groups.
    `notes` appends a small-print suffix (e.g. a sample size) to a bar's label."""
    n = len(series)
    width = min(0.8 / n, 0.22)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i, (name, values) in enumerate(series.items()):
        xs = [g + (i - (n - 1) / 2) * (width + 0.02) for g in range(len(groups))]
        ax.bar(xs, [v or 0 for v in values], width=width, color=SERIES[i], label=name, linewidth=0)
        suffixes = (notes or {}).get(name, [""] * len(values))
        for x, v, sfx in zip(xs, values, suffixes):
            _label(ax, x, v, (lambda v, s=sfx: f"{v:.0%}{s}") if percent else (lambda v, s=sfx: f"{v:g}{s}"))
    ax.set_xticks(range(len(groups)), groups)
    ax.tick_params(axis="x", length=0, labelsize=11, labelcolor=INK)
    if percent:
        ax.set_ylim(0, ymax or 1.1)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(loc="upper left", ncols=n, bbox_to_anchor=(0, 1.02), fontsize=10)
    fig.suptitle(title, x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.01, 0.905, subtitle, fontsize=10, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def small_multiples(path: Path, title: str, names: list[str], panels: list[tuple[str, list, str]]) -> None:
    """One panel per measure (different units never share an axis)."""
    fig, axes = plt.subplots(1, len(panels), figsize=(10, 3.6))
    for ax, (label, values, fmt) in zip(axes, panels):
        xs = range(len(names))
        ax.bar(xs, [v or 0 for v in values], width=0.5, color=SERIES[:len(names)], linewidth=0)
        for x, v in zip(xs, values):
            _label(ax, x, v, lambda v, f=fmt: f.format(v))
        ax.set_xticks(list(xs), names)
        ax.tick_params(axis="x", length=0, labelcolor=INK)
        ax.set_title(label, loc="left", fontsize=11, color=INK_2)
        ax.set_ylim(0, max(v or 0 for v in values) * 1.25 or 1)
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
    fig.suptitle(title, x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def model_comparison() -> None:
    report = json.loads((ROOT / "evaluation" / "metrics_report.json").read_text(encoding="utf-8"))
    # starcoder2:3b is excluded: that arm is void (its prompt template dropped the
    # retrieved context — see evaluation/model_comparison_analysis.md §3).
    models = {"llama3.1:8b (local)": "llama3.1:8b", "codellama:7b (local)": "codellama:7b",
              "gemini-3.5-flash-lite (API)": "gemini-3.5-flash-lite"}
    m = {label: report["models"][key] for label, key in models.items()}
    grouped_bars(
        OUT / "model_comparison.png", "Model choice: same retrieval, same prompt, 30 questions",
        ["Correct answer\n(pass rate)", "Faithful to sources\n(RAGAS faithfulness)", "Claims verified\n(1 − hallucination)"],
        {label: [v["test_pass_rate"], v["faithfulness"], 1 - v["hallucination_rate_grounding"]] for label, v in m.items()},
        subtitle="Run 1 (2026-09-09). Higher is better on every group.",
    )
    small_multiples(
        OUT / "model_efficiency.png", "Model choice: speed (CPU-only host for the local models)",
        ["llama3.1", "codellama", "gemini"],
        [("Median latency (s)", [v["latency_ms"]["median"] / 1000 for v in m.values()], "{:.1f}s"),
         ("Tokens per second", [v["tokens_per_second_mean"] for v in m.values()], "{:.1f}"),
         ("Mean prompt tokens", [v["tokens"]["mean_prompt"] for v in m.values()], "{:,.0f}")],
    )


def ablation() -> None:
    runs = sorted((ROOT / "evaluation" / "experiments").glob("*_ablation/report.json"))
    if not runs:
        print("no ablation report yet — skipping ablation charts")
        return
    report = json.loads(runs[-1].read_text(encoding="utf-8"))
    run_name = runs[-1].parent.name
    r = report["retrieval"]
    grouped_bars(
        OUT / "retrieval_ablation.png", "Retrieval: what the graph adds to vector search",
        ["Context precision", "Context recall", "Hit rate", "MRR"],
        {"Vector only": [r["vector_only"][k] for k in ("context_precision", "context_recall", "hit_rate", "mrr")],
         "Graph only": [r["graph_only"][k] for k in ("context_precision", "context_recall", "hit_rate", "mrr")],
         "Hybrid (the app)": [r["hybrid"][k] for k in ("context_precision", "context_recall", "hit_rate", "mrr")]},
        subtitle=f"Run 2 ({run_name}). {r['hybrid']['n']} questions with verified answer sections. Higher is better.",
    )
    g = report["generation"]
    arms = {"Raw model": g["raw"], "Persona prompt only": g["persona"], "Persona + RAG (the app)": g["rag"]}

    def verified(arm):
        rate = arm["statute_unverified_claim_rate"]
        return None if rate is None else 1 - rate

    grouped_bars(
        OUT / "rag_ablation.png", "Generation: what the prompt and retrieval each add (Gemini, 30 questions)",
        ["Correct answer\n(statute-checked pass rate)", "Cites a law section", "Claims verified\nin statute text"],
        {name: [a["statute_pass_rate"], a["answers_citing_a_section"], verified(a)] for name, a in arms.items()},
        subtitle=f"Run 2 ({run_name}). Every arm scored the same way against the real statute text.",
        # A rate over 1 claim is not comparable to one over 60: show the base.
        notes={name: ["", "", f"\nof {a['claims_checked']}"] for name, a in arms.items()},
    )
    small_multiples(
        OUT / "rag_efficiency.png", "Generation: what RAG costs",
        ["Raw", "Persona", "RAG"],
        [("Prompt tokens / answer", [a["mean_prompt_tokens"] for a in arms.values()], "{:,.0f}"),
         ("Answer tokens / answer", [a["mean_completion_tokens"] for a in arms.values()], "{:,.0f}"),
         ("Latency / answer (s)", [(a["mean_latency_ms"] or 0) / 1000 for a in arms.values()], "{:.1f}s")],
    )


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    model_comparison()
    ablation()
    print(f"charts written to {OUT}")
