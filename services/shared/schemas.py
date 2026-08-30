"""
Pydantic request/response contracts shared by all five services — the exact
shapes traced in the architecture plan, kept in one place so a payload built
by one service is guaranteed to deserialize correctly in the next.
"""

from typing import Literal

from pydantic import BaseModel, Field

Provider = Literal["ollama", "gemini"]


# ---------------------------------------------------------------------------
# Data Service
# ---------------------------------------------------------------------------
class VectorSearchRequest(BaseModel):
    embedding: list[float]
    top_k: int = 10


class VectorSearchResult(BaseModel):
    chunk_id: str
    record_id: str
    text: str
    distance: float
    metadata: dict


class VectorSearchResponse(BaseModel):
    results: list[VectorSearchResult]


# ---------------------------------------------------------------------------
# LLM Service
# ---------------------------------------------------------------------------
class EmbedRequest(BaseModel):
    text: str


class EmbedResponse(BaseModel):
    embedding: list[float]
    model: str
    dimensions: int


class ContextItem(BaseModel):
    text: str
    act: str | None = None
    section: str | None = None
    page: int | None = None
    source_pdf: str | None = None
    score: float | None = None
    source: str | None = None  # "vector" | "graph"


class GenerateRequest(BaseModel):
    question: str
    context: list[ContextItem] = Field(default_factory=list)
    provider: Provider = "ollama"
    # True (default, always used by the Ask tab): the fixed legal persona +
    # hard rules + context. False (Eval tab's no-retrieval cells only): no
    # system prompt at all — the model's raw, unguided behavior, so the
    # comparison against the RAG+persona cells is an honest apples-to-oranges
    # contrast on purpose (raw model vs. this app's actual pipeline), not the
    # same persona artificially starved of context.
    use_persona: bool = True


class GenerateResponse(BaseModel):
    answer: str
    provider: Provider
    model: str
    used_context: bool
    latency_ms: float


# ---------------------------------------------------------------------------
# Retrieval Service
# ---------------------------------------------------------------------------
class RetrieveRequest(BaseModel):
    question: str
    top_k: int = 5
    # "hybrid" (default, used everywhere except the new Eval cell): vector +
    # graph, fused by relevance score. "graph_only": skip vector search
    # entirely — used by the Eval tab's "Ollama + Graph RAG" comparison cell,
    # to show what pure graph-based retrieval looks like on its own.
    mode: Literal["hybrid", "graph_only"] = "hybrid"


class GraphRelationship(BaseModel):
    relation: str
    source_name: str
    target_name: str
    evidence_count: int


class RetrievalTiming(BaseModel):
    """Real measured durations of each real sub-step — used to drive the
    frontend's live pipeline view honestly (not simulated numbers)."""
    embed_ms: float
    vector_search_ms: float
    graph_ms: float


class RetrieveResponse(BaseModel):
    context: list[ContextItem]
    matched_entities: list[str]
    graph_relationships: list[GraphRelationship]
    timing: RetrievalTiming


# ---------------------------------------------------------------------------
# Rights Library (category browsing over the existing graph entities)
# ---------------------------------------------------------------------------
class Category(BaseModel):
    slug: str
    label: str
    entity_names: list[str]


class CategorySection(BaseModel):
    id: str
    act: str
    section: str
    title: str
    page: int
    source_pdf: str
    content: str


class CategorySectionsResponse(BaseModel):
    category: Category
    sections: list[CategorySection]


# ---------------------------------------------------------------------------
# Orchestration Service
# ---------------------------------------------------------------------------
class Citation(BaseModel):
    act: str | None = None
    section: str | None = None
    page: int | None = None
    source_pdf: str | None = None


class AskRequest(BaseModel):
    question: str
    provider: Provider = "ollama"


Confidence = Literal["high", "medium", "low", "none"]


class GroundingCheck(BaseModel):
    """Result of checking the answer's cited section numbers and rupee
    amounts against what was actually in the retrieved context — see
    services/orchestration_service/grounding.py for what this can and can't
    catch."""
    total_claims: int
    verified_claims: int
    unverified_claims: int
    unverified_sections: list[str] = Field(default_factory=list)
    unverified_amounts: list[int] = Field(default_factory=list)


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    provider: Provider
    model: str
    used_context: bool
    confidence: Confidence
    context: list[ContextItem]  # the actual top-k chunks — Legal Evidence View / Response Card need these
    matched_entities: list[str]
    grounding: GroundingCheck


class EvalRequest(BaseModel):
    question: str


class EvalCell(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    provider: Provider
    model: str
    used_context: bool
    latency_ms: float
    grounding: GroundingCheck


class EvalResponse(BaseModel):
    # The shared retrieval step behind all 4 cells — surfaced once so the
    # Eval tab's dashboard can show the real inner workings (timing, matched
    # entities, graph relationships, the actual ranked chunks) rather than
    # just four text boxes.
    retrieval: RetrieveResponse
    ollama_only: EvalCell
    ollama_rag: EvalCell
    gemini_only: EvalCell
    gemini_rag: EvalCell
    # Graph-only retrieval (vector search skipped entirely) + Ollama +
    # persona — isolates what the graph path alone contributes, separate
    # from the hybrid retrieval used by ollama_rag above.
    ollama_graph_rag: EvalCell
