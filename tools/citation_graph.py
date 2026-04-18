"""
Citation Graph tool: walk the reference graph of key papers.

Strategy: use REFERENCES (backward citations) not CITATIONS (forward citations).

Why references and not citations?
  - References = foundational papers this work builds on. These are older,
    more established, and more likely to contain the core method we need.
  - Citations = papers that cite this one. These are newer but potentially
    scattered across many unrelated topics.

For a literature scout, references give us the "textbook" knowledge tree.
We go one hop deep only — this is controlled by citation_round in state.

API: Semantic Scholar Graph API (free, no key needed)
"""

import time
import requests
from typing import List, Dict


BASE_URL = "https://api.semanticscholar.org/graph/v1/paper"
PAPER_FIELDS = "paperId,title,authors,year,abstract,openAccessPdf,externalIds"


def get_references(s2_paper_id: str, limit: int = 8) -> List[Dict]:
    """
    Fetch papers that this paper cites (its reference list).
    These are foundational/upstream papers — high value for literature surveys.
    """
    url = f"{BASE_URL}/{s2_paper_id}/references"
    params = {
        'fields': PAPER_FIELDS,
        'limit': limit
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        papers = []
        for item in data.get('data', []):
            p = item.get('citedPaper', {})
            if not p.get('paperId') or not p.get('title'):
                continue  # Skip incomplete records

            pdf_url = _resolve_pdf_url(p)
            papers.append({
                'paper_id': p.get('paperId', ''),
                'title': p.get('title', ''),
                'authors': [a.get('name', '') for a in p.get('authors', [])],
                'abstract': (p.get('abstract') or '').replace('\n', ' '),
                'year': p.get('year'),
                'pdf_url': pdf_url,
                'source': 'semantic_scholar_reference',
                's2_id': p.get('paperId'),
                'full_text': None
            })

        time.sleep(0.5)
        return papers

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"  [Citations] Paper {s2_paper_id[:20]} not found in S2")
        elif e.response.status_code == 429:
            print(f"  [Citations] Rate limited. Sleeping 10s...")
            time.sleep(10)
        else:
            print(f"  [Citations] HTTP {e.response.status_code}")
        return []
    except Exception as e:
        print(f"  [Citations] Failed for {s2_paper_id[:20]}: {e}")
        return []


def get_forward_citations(s2_paper_id: str, limit: int = 5) -> List[Dict]:
    """
    Fetch papers that cite this paper (newer work building on this paper).
    Less useful for foundational survey, but useful for 'state of the art' check.
    Not used in main flow but available if needed.
    """
    url = f"{BASE_URL}/{s2_paper_id}/citations"
    params = {
        'fields': PAPER_FIELDS,
        'limit': limit
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        papers = []
        for item in data.get('data', []):
            p = item.get('citingPaper', {})
            if not p.get('paperId') or not p.get('title'):
                continue

            pdf_url = _resolve_pdf_url(p)
            papers.append({
                'paper_id': p.get('paperId', ''),
                'title': p.get('title', ''),
                'authors': [a.get('name', '') for a in p.get('authors', [])],
                'abstract': (p.get('abstract') or '').replace('\n', ' '),
                'year': p.get('year'),
                'pdf_url': pdf_url,
                'source': 'semantic_scholar_citation',
                's2_id': p.get('paperId'),
                'full_text': None
            })

        time.sleep(0.5)
        return papers

    except Exception as e:
        print(f"  [ForwardCite] Failed: {e}")
        return []


def _resolve_pdf_url(paper: dict) -> str | None:
    """Resolve best available PDF URL for a paper."""
    if paper.get('openAccessPdf'):
        url = paper['openAccessPdf'].get('url')
        if url:
            return url
    arxiv_id = paper.get('externalIds', {}).get('ArXiv')
    if arxiv_id:
        return f"https://arxiv.org/pdf/{arxiv_id}"
    return None
