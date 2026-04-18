"""
PDF Reader tool: fetch PDF from URL → extract text → trim to key sections.

Strategy for long papers (typical aerospace paper = 10-20 pages):
  - Read pages 0-2 (abstract, intro, problem statement)
  - Read middle pages (methods section)
  - Read last 2 pages (results, conclusion, limitations)

This gives the LLM what it needs for structured extraction without
blowing the context window. Keeps extraction fast on llama3.2.

Library: pdfplumber (pure Python, handles most arxiv PDFs cleanly)
Fallback: if PDF fails, we use the abstract only.
"""

import io
import time
import requests
from typing import Optional

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    print("WARNING: pdfplumber not installed. Run: pip install pdfplumber")


HEADERS = {
    'User-Agent': 'Mozilla/5.0 (aerospace-research-bot; academic use)'
}


def fetch_and_parse_pdf(pdf_url: str, timeout: int = 30) -> Optional[str]:
    """
    Fetch a PDF from URL and extract text using pdfplumber.
    Returns extracted text string, or None on failure.
    """
    if not pdf_url or not PDFPLUMBER_AVAILABLE:
        return None

    try:
        response = requests.get(pdf_url, headers=HEADERS, timeout=timeout)
        response.raise_for_status()

        # Check we actually got a PDF
        content_type = response.headers.get('Content-Type', '')
        if 'pdf' not in content_type.lower() and not response.content[:4] == b'%PDF':
            print(f"  [PDF] Not a PDF: {content_type} for {pdf_url[:60]}")
            return None

        pdf_bytes = io.BytesIO(response.content)
        return _extract_text(pdf_bytes)

    except requests.exceptions.Timeout:
        print(f"  [PDF] Timeout fetching {pdf_url[:60]}")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"  [PDF] HTTP {e.response.status_code} for {pdf_url[:60]}")
        return None
    except Exception as e:
        print(f"  [PDF] Failed: {e}")
        return None


def _extract_text(pdf_bytes: io.BytesIO) -> Optional[str]:
    """Extract text from a pdfplumber file object."""
    try:
        with pdfplumber.open(pdf_bytes) as pdf:
            total_pages = len(pdf.pages)
            pages_to_read = _select_pages(total_pages)
            sections = []

            for page_num in pages_to_read:
                page = pdf.pages[page_num]
                text = page.extract_text()
                if text and text.strip():
                    sections.append(f"[Page {page_num + 1}]\n{text.strip()}")

            return "\n\n".join(sections) if sections else None

    except Exception as e:
        print(f"  [PDF] pdfplumber extraction failed: {e}")
        return None


def _select_pages(total_pages: int) -> list:
    """
    Select which pages to read for efficient extraction.

    For a 15-page paper this gives us pages: 0,1,2, 7,8, 13,14
    That's intro + methods midpoint + conclusion — enough for structured extraction.
    """
    pages = list(range(min(3, total_pages)))   # First 3 pages (abstract, intro)

    if total_pages > 6:
        mid = total_pages // 2
        pages += [mid, mid + 1]               # Middle (usually methods/results)

    if total_pages > 4:
        tail_start = max(3, total_pages - 2)
        pages += list(range(tail_start, total_pages))  # Last 2 (conclusion)

    return sorted(set(pages))


def extract_key_sections(text: str, max_chars: int = 7000) -> str:
    """
    Trim extracted text to max_chars for LLM context window.
    Truncates from the middle (preserves start and end).
    """
    if not text:
        return ""
    if len(text) <= max_chars:
        return text

    half = max_chars // 2
    return text[:half] + "\n\n[...truncated...]\n\n" + text[-half:]
