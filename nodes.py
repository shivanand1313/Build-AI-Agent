"""
LangGraph Node Functions for the Literature Scout Agent.

Each function:
  - Takes ScoutState as input
  - Returns a PARTIAL dict (only the keys it updates)
  - LangGraph merges the partial dict into state automatically

Node execution order:
  generate_queries → search_papers → read_papers → extract_structured
       → [conditional] follow_citations → extract_structured (again)
       → store_to_kb

The citation loop runs ONCE: citation_round 0 → 1, then routes to store.
"""

import json
import time
from typing import List

import ollama

from state import ScoutState, Paper, ExtractedPaper
from tools.search import search_arxiv, search_semantic_scholar, deduplicate_papers
from tools.pdf_reader import fetch_and_parse_pdf, extract_key_sections
from tools.citation_graph import get_references
from knowledge_base import AerospaceKnowledgeBase

# Shared KB instance — all nodes reference the same store
kb = AerospaceKnowledgeBase()


# ── Node 1: Query Generator ──────────────────────────────────────────────────

def generate_queries(state: ScoutState) -> dict:
    """
    Generates 4 search queries from different angles of the problem statement.

    Why multi-angle? A single query clusters around one vocabulary.
    Aerospace routing papers use different terms than operations research papers
    that solve the same underlying problem. We need both.

    Angle 1: Core technical terminology (as an expert would search)
    Angle 2: Methods/algorithms (what approach solves this class of problem)
    Angle 3: Domain-specific (aerospace, aviation, manufacturing context)
    Angle 4: Analogous problem in a different field (broadest net)
    """
    problem = state['problem_statement']

    prompt = f"""You are an aerospace research assistant performing a literature survey.

Given this problem statement, generate exactly 4 search queries for academic paper databases.
Each query must approach the problem from a DIFFERENT angle to maximize coverage:
  1. Core technical terms an expert in this field would use
  2. Methods/algorithms that solve this class of problem  
  3. Aerospace/aviation/manufacturing domain-specific angle
  4. Analogous problem in a related field (e.g., robotics, logistics, civil engineering)

Problem: {problem}

Respond with ONLY a valid JSON array of 4 strings. No explanation, no markdown, no extra text.
Example format: ["query one here", "query two here", "query three here", "query four here"]"""

    try:
        response = ollama.chat(
            model='llama3.2',
            messages=[{'role': 'user', 'content': prompt}]
        )
        raw = response['message']['content'].strip()
        queries = _parse_json_safely(raw, fallback=None)

        if not isinstance(queries, list) or len(queries) < 2:
            raise ValueError("LLM did not return a valid list")

        queries = [str(q).strip() for q in queries if q][:5]

    except Exception as e:
        print(f"  [QueryGen] LLM failed ({e}), using fallback queries")
        # Deterministic fallback — always works even if Ollama is down
        queries = [
            problem.strip(),
            f"{problem.strip()} optimization algorithm",
            f"{problem.strip()} aerospace manufacturing",
            f"{problem.strip()} survey review"
        ]

    print(f"\n[Node 1/6] Generated {len(queries)} search queries:")
    for i, q in enumerate(queries, 1):
        print(f"  Q{i}: {q[:80]}")

    return {
        'search_queries': queries,
        'messages': [f"Generated {len(queries)} search queries from problem statement"]
    }


# ── Node 2: Paper Searcher ────────────────────────────────────────────────────

def search_papers(state: ScoutState) -> dict:
    """
    Searches arxiv + Semantic Scholar for each generated query.
    Deduplicates across all results before returning.

    Returns up to 3 papers per query per source = ~24 papers max before dedup.
    After dedup typically 10-18 unique papers.
    """
    queries = state['search_queries']
    all_papers: List[Paper] = []

    print(f"\n[Node 2/6] Searching {len(queries)} queries across arxiv + Semantic Scholar...")

    for i, query in enumerate(queries, 1):
        print(f"  Searching Q{i}: {query[:60]}...")
        arxiv_results = search_arxiv(query, max_results=3)
        s2_results = search_semantic_scholar(query, max_results=3)
        all_papers.extend(arxiv_results + s2_results)
        time.sleep(0.3)  # Small delay between query bursts

    unique_papers = deduplicate_papers(all_papers)
    print(f"  Found {len(all_papers)} total → {len(unique_papers)} unique papers")

    return {
        'raw_papers': unique_papers,
        'papers_to_read': unique_papers,   # Replaced (current batch to process)
        'messages': [f"Search complete: {len(unique_papers)} unique papers from {len(queries)} queries"]
    }


# ── Node 3: PDF Reader ────────────────────────────────────────────────────────

def read_papers(state: ScoutState) -> dict:
    """
    Fetches and parses PDFs for papers in papers_to_read queue.

    Uses papers_to_read (not raw_papers) so this node works correctly
    for BOTH the initial papers AND the citation papers — same node,
    different input queue each time it runs.

    Falls back to abstract-only if PDF is unavailable/fails.
    """
    papers_to_read = state.get('papers_to_read', [])
    papers_read: List[Paper] = []

    print(f"\n[Node 3/6] Reading {len(papers_to_read)} papers (PDF + abstract fallback)...")

    for paper in papers_to_read:
        pdf_url = paper.get('pdf_url')
        print(f"  → {paper['title'][:55]}...")

        full_text = None
        if pdf_url:
            full_text = fetch_and_parse_pdf(pdf_url)

        if full_text:
            trimmed = extract_key_sections(full_text)
            text_source = "PDF"
        else:
            # Abstract fallback — still useful for structured extraction
            trimmed = paper.get('abstract', '')
            text_source = "abstract"

        papers_read.append({**paper, 'full_text': trimmed})
        print(f"    ✓ {text_source} ({len(trimmed)} chars)")

    return {
        'papers_read': papers_read,
        'messages': [f"Read {len(papers_read)} papers (PDF where available, abstract fallback)"]
    }


# ── Node 4: Structured Extractor ──────────────────────────────────────────────

def extract_structured(state: ScoutState) -> dict:
    """
    Uses llama3.2 to extract structured JSON from each paper.

    Input sources:
      - papers_read (from initial search)
      - citation_papers (from citation graph, fetched inline)

    Skips papers already in extracted_papers (idempotent across citation rounds).

    The extraction schema is FIXED — changing it breaks the feasibility critic.
    """
    # Gather all papers that have been read
    papers_to_extract = state.get('papers_read', []) + state.get('citation_papers', [])

    # Skip already-extracted (handles re-entry after citation round)
    already_extracted = {p['paper_id'] for p in state.get('extracted_papers', [])}
    pending = [p for p in papers_to_extract if p['paper_id'] not in already_extracted]

    print(f"\n[Node 4/6] Extracting structured data from {len(pending)} papers...")

    extracted: List[ExtractedPaper] = []
    citation_seeds: List[str] = []

    for paper in pending:
        text = paper.get('full_text') or paper.get('abstract') or ''
        if not text.strip():
            print(f"  ✗ Skipped (no text): {paper['title'][:50]}")
            continue

        result = _extract_one_paper(paper, text)
        if result:
            extracted.append(result)
            # Collect S2 IDs from first-round papers for citation following
            if paper.get('s2_id') and state.get('citation_round', 0) == 0:
                citation_seeds.append(paper['s2_id'])
            print(f"  ✓ {paper['title'][:55]}")
        else:
            print(f"  ✗ Extraction failed: {paper['title'][:50]}")

    print(f"  Extracted {len(extracted)}/{len(pending)} papers successfully")

    return {
        'extracted_papers': extracted,
        'citation_seeds': citation_seeds,
        'messages': [f"Structured extraction: {len(extracted)}/{len(pending)} papers succeeded"]
    }


def _extract_one_paper(paper: dict, text: str) -> ExtractedPaper | None:
    """Run LLM extraction for a single paper. Returns None on failure."""

    prompt = f"""Extract structured information from this research paper.
Respond ONLY with a valid JSON object — no explanation, no markdown fences.

Paper Title: {paper['title']}
Authors: {', '.join(paper.get('authors', [])[:3])}
Year: {paper.get('year', 'unknown')}

Paper Content:
{text[:6000]}

Return this exact JSON structure (all fields required):
{{
  "problem_addressed": "One sentence: what specific problem does this paper solve?",
  "proposed_method": "One sentence: what is the core approach or algorithm?",
  "key_assumptions": [
    "assumption 1 that must hold for this method to work",
    "assumption 2",
    "assumption 3"
  ],
  "results_metrics": "Key quantitative results: accuracy %, speedup, error rate, etc. Use 'Not reported' if absent.",
  "constraints": [
    "constraint or limitation that restricts applicability",
    "constraint 2",
    "constraint 3"
  ]
}}

Be specific and technical. Do not use generic statements like 'the method works well'."""

    try:
        response = ollama.chat(
            model='llama3.2',
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.1}  # Low temp for consistent JSON output
        )
        raw = response['message']['content'].strip()
        extraction = _parse_json_safely(raw, fallback=None)

        if not extraction or not isinstance(extraction, dict):
            return None

        # Validate required fields exist
        required = ['problem_addressed', 'proposed_method', 'key_assumptions',
                    'results_metrics', 'constraints']
        if not all(k in extraction for k in required):
            return None

        return ExtractedPaper(
            paper_id=paper['paper_id'],
            title=paper['title'],
            authors=paper.get('authors', []),
            year=paper.get('year'),
            source=paper.get('source', ''),
            abstract=paper.get('abstract', ''),
            problem_addressed=str(extraction['problem_addressed']),
            proposed_method=str(extraction['proposed_method']),
            key_assumptions=_ensure_list(extraction['key_assumptions']),
            results_metrics=str(extraction['results_metrics']),
            constraints=_ensure_list(extraction['constraints'])
        )

    except Exception as e:
        print(f"    LLM error: {e}")
        return None


# ── Node 5: Citation Graph Follower ───────────────────────────────────────────

def follow_citations(state: ScoutState) -> dict:
    """
    Follows references of top papers to discover foundational work.

    Uses up to 3 seed papers (to stay within S2 rate limits).
    Fetches PDFs inline here (unlike the main path where read_papers is separate)
    because citation papers go directly to extract_structured next.

    Skips papers already in raw_papers (already seen).
    """
    seeds = state.get('citation_seeds', [])[:3]   # Cap at 3 seeds
    seen_ids = {p['paper_id'] for p in state.get('raw_papers', [])}

    print(f"\n[Node 5/6] Following citation graph from {len(seeds)} seed papers...")

    citation_papers: List[Paper] = []

    for s2_id in seeds:
        refs = get_references(s2_id, limit=5)
        for ref in refs:
            if not ref['paper_id'] or ref['paper_id'] in seen_ids:
                continue
            seen_ids.add(ref['paper_id'])

            # Fetch PDF inline for citation papers
            if ref.get('pdf_url'):
                print(f"  Reading citation: {ref['title'][:55]}...")
                full_text = fetch_and_parse_pdf(ref['pdf_url'])
                ref['full_text'] = extract_key_sections(full_text or '')
            else:
                ref['full_text'] = ref.get('abstract', '')

            citation_papers.append(ref)

    print(f"  Found {len(citation_papers)} new papers via citation graph")

    return {
        'citation_papers': citation_papers,
        'citation_round': state.get('citation_round', 0) + 1,
        'messages': [f"Citation graph: {len(citation_papers)} foundational papers discovered"]
    }


# ── Node 6: Knowledge Base Writer ─────────────────────────────────────────────

def store_to_kb(state: ScoutState) -> dict:
    """
    Writes all ExtractedPaper records to ChromaDB.
    Uses upsert — safe to run multiple times (idempotent).
    """
    papers = state.get('extracted_papers', [])
    print(f"\n[Node 6/6] Storing {len(papers)} papers to ChromaDB...")

    stored = kb.store_papers_batch(papers)

    print(f"  Stored {stored}/{len(papers)} papers")
    print(f"  Total in KB: {kb.count()} papers")

    return {
        'stored_count': stored,
        'messages': [f"ChromaDB: stored {stored}/{len(papers)} papers (total in KB: {kb.count()})"]
    }


# ── Utilities ─────────────────────────────────────────────────────────────────

def _parse_json_safely(text: str, fallback):
    """
    Parse JSON from LLM output robustly.
    Handles: raw JSON, ```json fences, trailing commas (via repair attempt).
    """
    # Strip markdown code fences
    if '```' in text:
        parts = text.split('```')
        for part in parts:
            part = part.strip()
            if part.startswith('json'):
                part = part[4:].strip()
            try:
                return json.loads(part)
            except json.JSONDecodeError:
                continue

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try finding JSON boundaries manually
    try:
        start_chars = {'{': '}', '[': ']'}
        for start_char, end_char in start_chars.items():
            if start_char in text:
                start = text.index(start_char)
                end = text.rindex(end_char) + 1
                return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        pass

    return fallback


def _ensure_list(value) -> List[str]:
    """Ensure a value is a list of strings."""
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [value]
    return []
