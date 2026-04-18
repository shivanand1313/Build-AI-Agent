"""
Literature Scout — Entry Point

Usage:
    python run.py                          # Run with default aerospace problem
    python run.py --query "your problem"   # Run with custom problem
    python run.py --inspect               # Inspect what's currently in ChromaDB
    python run.py --query-kb "search term" # Semantic search over stored papers
"""

import argparse
import json
from graph import build_scout_graph
from knowledge_base import AerospaceKnowledgeBase
from state import ScoutState


def run_scout(problem_statement: str) -> ScoutState:
    """
    Run the full Literature Scout pipeline for a given problem statement.
    Returns the final state after all nodes complete.
    """
    graph = build_scout_graph()

    # ── Initial state — every field must be present ──────────────────────────
    # Annotated[List, operator.add] fields must start as [] (not missing)
    # Plain List fields can also start as []
    initial_state: ScoutState = {
        'problem_statement': problem_statement.strip(),
        'search_queries': [],
        'raw_papers': [],
        'papers_to_read': [],
        'papers_read': [],
        'extracted_papers': [],
        'citation_seeds': [],
        'citation_papers': [],
        'citation_round': 0,
        'stored_count': 0,
        'messages': []
    }

    print(f"\n{'═' * 65}")
    print(f"  LITERATURE SCOUT — Aerospace Research Agent")
    print(f"{'═' * 65}")
    print(f"\n  Problem: {problem_statement[:100]}")
    print(f"\n{'─' * 65}")

    final_state = graph.invoke(initial_state)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'═' * 65}")
    print("  SCOUT COMPLETE — Summary")
    print(f"{'─' * 65}")
    for msg in final_state.get('messages', []):
        print(f"  ✓  {msg}")

    extracted = final_state.get('extracted_papers', [])
    print(f"\n  Papers extracted: {len(extracted)}")
    print(f"  Papers stored:    {final_state.get('stored_count', 0)}")
    print(f"{'═' * 65}\n")

    if extracted:
        print("  SAMPLE — First extracted paper:")
        print(f"{'─' * 65}")
        p = extracted[0]
        print(f"  Title:      {p['title']}")
        print(f"  Problem:    {p.get('problem_addressed', 'N/A')}")
        print(f"  Method:     {p.get('proposed_method', 'N/A')}")
        print(f"  Constraints:{p.get('constraints', [])}")
        print(f"{'═' * 65}\n")

    return final_state


def inspect_kb():
    """Print all papers currently stored in ChromaDB."""
    kb = AerospaceKnowledgeBase()
    papers = kb.get_all()

    print(f"\n{'═' * 65}")
    print(f"  KNOWLEDGE BASE INSPECTION — {len(papers)} papers stored")
    print(f"{'═' * 65}")

    if not papers:
        print("  (empty — run the scout first)")
        return

    for i, p in enumerate(papers, 1):
        print(f"\n  [{i}] {p.get('title', 'Unknown title')}")
        print(f"       Year: {p.get('year', '?')} | Source: {p.get('source', '?')}")
        print(f"       Problem: {p.get('problem_addressed', 'N/A')[:80]}")
        print(f"       Method:  {p.get('proposed_method', 'N/A')[:80]}")
        constraints = p.get('constraints', [])
        if constraints:
            print(f"       Constraints: {constraints[0][:70]}")

    print(f"\n{'═' * 65}")


def query_kb(search_term: str, n_results: int = 5):
    """Run a semantic search over the knowledge base."""
    kb = AerospaceKnowledgeBase()
    results = kb.query(search_term, n_results=n_results)

    print(f"\n{'═' * 65}")
    print(f"  KB SEARCH: '{search_term}'")
    print(f"{'─' * 65}")

    if not results:
        print("  No results (KB may be empty — run the scout first)")
        return

    for i, p in enumerate(results, 1):
        score = p.get('relevance_score', 0)
        print(f"\n  [{i}] (score: {score:.3f}) {p.get('title', '?')}")
        print(f"       {p.get('problem_addressed', 'N/A')[:90]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Literature Scout Agent")
    parser.add_argument('--query', type=str, help="Problem statement to research")
    parser.add_argument('--inspect', action='store_true', help="Inspect KB contents")
    parser.add_argument('--query-kb', type=str, help="Semantic search over KB")
    args = parser.parse_args()

    if args.inspect:
        inspect_kb()

    elif args.query_kb:
        query_kb(args.query_kb)

    else:
        # Default: run scout on aerospace harness routing problem
        # (matches Shiv's primary use case for this system)
        problem = args.query or """
        Automated wire harness routing optimization in aircraft fuselage assembly.
        Minimize total wire length while satisfying electrical separation requirements
        between power, signal, and coaxial bundles, avoiding structural interference,
        and meeting aerospace standard AS50881 constraints on bend radius and clamp spacing.
        """
        run_scout(problem)
