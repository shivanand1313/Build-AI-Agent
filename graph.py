"""
LangGraph graph definition for the Literature Scout Agent.

Graph topology:
                    ┌──────────────────────────────────────────────┐
                    │                                              │
  START → generate_queries → search_papers → read_papers          │
                                                  │               │
                                         extract_structured ◄─────┘
                                                  │
                               ┌──── citation_round == 0 ────┐
                               │  (has seeds)                 │
                               ▼                              ▼
                       follow_citations              store_to_kb → END
                               │
                        (citation_round = 1)
                               │
                    citation_papers written to state
                               │
                               └──► extract_structured
                                    (citation_round == 1 now)
                                          │
                                    store_to_kb → END

The conditional edge prevents infinite loops:
  - First time: citation_round == 0, has seeds → go to follow_citations
  - Second time: citation_round == 1 → go directly to store_to_kb
"""

from langgraph.graph import StateGraph, START, END

from state import ScoutState
from nodes import (
    generate_queries,
    search_papers,
    read_papers,
    extract_structured,
    follow_citations,
    store_to_kb
)


def route_after_extraction(state: ScoutState) -> str:
    """
    Conditional edge function called after extract_structured.

    Routes to follow_citations on the FIRST pass only.
    Requires both:
      - citation_round == 0 (haven't followed citations yet)
      - citation_seeds not empty (have S2 IDs to follow)

    Returns node name (string) — LangGraph uses this to pick the next node.
    """
    is_first_round = state.get('citation_round', 0) == 0
    has_seeds = bool(state.get('citation_seeds'))

    if is_first_round and has_seeds:
        return "follow_citations"
    return "store_to_kb"


def build_scout_graph():
    """
    Build and compile the Literature Scout LangGraph.
    Returns a compiled graph ready to invoke.
    """
    builder = StateGraph(ScoutState)

    # ── Add nodes ──────────────────────────────────────────────────────────
    builder.add_node("generate_queries", generate_queries)
    builder.add_node("search_papers", search_papers)
    builder.add_node("read_papers", read_papers)
    builder.add_node("extract_structured", extract_structured)
    builder.add_node("follow_citations", follow_citations)
    builder.add_node("store_to_kb", store_to_kb)

    # ── Add edges ──────────────────────────────────────────────────────────

    # Linear pipeline: start → generate → search → read → extract
    builder.add_edge(START, "generate_queries")
    builder.add_edge("generate_queries", "search_papers")
    builder.add_edge("search_papers", "read_papers")
    builder.add_edge("read_papers", "extract_structured")

    # Conditional after extraction: first pass → citations, second pass → store
    builder.add_conditional_edges(
        "extract_structured",
        route_after_extraction,
        {
            "follow_citations": "follow_citations",
            "store_to_kb": "store_to_kb"
        }
    )

    # After citation following, extract those papers too (same node, different state)
    builder.add_edge("follow_citations", "extract_structured")

    # Terminal
    builder.add_edge("store_to_kb", END)

    return builder.compile()
