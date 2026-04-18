"""
Search tools: arxiv API + Semantic Scholar API

Both are free with no API key required for basic usage.
S2 has a rate limit of ~100 req/5min unauthenticated — we sleep between calls.

Key design: we return the same Paper schema from both APIs so the rest of the
graph doesn't care which source a paper came from.
"""

import time
import requests
from typing import List, Dict

try:
    import arxiv
    ARXIV_AVAILABLE = True
except ImportError:
    ARXIV_AVAILABLE = False
    print("WARNING: arxiv package not installed. Run: pip install arxiv")


def search_arxiv(query: str, max_results: int = 5) -> List[Dict]:
    """
    Search arxiv using the arxiv Python client.
    Returns up to max_results papers matching the query.
    """
    if not ARXIV_AVAILABLE:
        return []

    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        papers = []
        for result in client.results(search):
            papers.append({
                'paper_id': result.get_short_id(),
                'title': result.title,
                'authors': [a.name for a in result.authors],
                'abstract': result.summary.replace('\n', ' '),
                'year': result.published.year if result.published else None,
                'pdf_url': result.pdf_url,
                'source': 'arxiv',
                's2_id': None,         # Will try to resolve via S2 if needed
                'full_text': None
            })
        return papers

    except Exception as e:
        print(f"  [arxiv] Search failed for '{query[:40]}': {e}")
        return []


def search_semantic_scholar(query: str, max_results: int = 5) -> List[Dict]:
    """
    Search Semantic Scholar Graph API.
    Automatically resolves arxiv IDs to PDF URLs when openAccessPdf is missing.

    S2 API docs: https://api.semanticscholar.org/api-docs/graph
    """
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        'query': query,
        'limit': max_results,
        'fields': 'paperId,title,authors,year,abstract,openAccessPdf,externalIds'
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        papers = []
        for p in data.get('data', []):
            # Resolve PDF URL: prefer S2's openAccessPdf, fall back to arxiv
            pdf_url = None
            if p.get('openAccessPdf'):
                pdf_url = p['openAccessPdf'].get('url')
            arxiv_id = p.get('externalIds', {}).get('ArXiv')
            if not pdf_url and arxiv_id:
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

            papers.append({
                'paper_id': p.get('paperId', ''),
                'title': p.get('title', ''),
                'authors': [a.get('name', '') for a in p.get('authors', [])],
                'abstract': (p.get('abstract') or '').replace('\n', ' '),
                'year': p.get('year'),
                'pdf_url': pdf_url,
                'source': 'semantic_scholar',
                's2_id': p.get('paperId'),
                'full_text': None
            })

        time.sleep(0.5)   # Be a good citizen — S2 rate limit
        return papers

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            print(f"  [S2] Rate limited. Sleeping 10s...")
            time.sleep(10)
        else:
            print(f"  [S2] HTTP {e.response.status_code} for '{query[:40]}'")
        return []
    except Exception as e:
        print(f"  [S2] Search failed for '{query[:40]}': {e}")
        return []


def deduplicate_papers(papers: List[Dict]) -> List[Dict]:
    """
    Remove duplicate papers by title (case-insensitive).
    Prefers arxiv entries over S2 entries (they have more reliable PDF URLs).
    """
    seen_titles = {}
    for paper in papers:
        title_key = paper['title'].lower().strip()
        if not title_key:
            continue
        if title_key not in seen_titles:
            seen_titles[title_key] = paper
        elif paper['source'] == 'arxiv' and seen_titles[title_key]['source'] != 'arxiv':
            # Prefer arxiv entry
            seen_titles[title_key] = paper

    return list(seen_titles.values())
