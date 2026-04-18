"""
Aerospace Knowledge Base — ChromaDB-backed vector store.

Schema is designed with TWO consumers in mind:
  1. Literature Scout writes ExtractedPaper records here
  2. Feasibility Critic READS from here — it needs constraints + assumptions
     to evaluate whether a method is applicable to the target aerospace problem.

ChromaDB stores:
  - Document (embedded text): concatenated structured fields for semantic search
  - Metadata (flat dict): all structured fields for filtered retrieval
  - ID: paper_id (enables idempotent upsert — run scout twice, no duplicates)

All metadata values must be strings (ChromaDB limitation) — we JSON-encode lists.
"""

import json
from datetime import datetime
from typing import List, Dict, Optional

import chromadb
from chromadb.utils import embedding_functions


class AerospaceKnowledgeBase:
    """
    Vector store for aerospace research papers.
    Uses ChromaDB with sentence-transformer embeddings (runs locally, no API key).
    """

    def __init__(self, persist_path: str = "./aerospace_kb"):
        """
        Args:
            persist_path: Directory where ChromaDB stores its data.
                          Persists across runs — scout can be re-run without
                          re-fetching already-stored papers.
        """
        self.client = chromadb.PersistentClient(path=persist_path)

        # DefaultEmbeddingFunction uses all-MiniLM-L6-v2 (downloads ~80MB on first run)
        # This runs locally — no OpenAI key needed.
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()

        self.papers = self.client.get_or_create_collection(
            name="aerospace_papers",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}   # Cosine similarity for semantic search
        )

    # ── Write ────────────────────────────────────────────────────────────────

    def store_paper(self, paper: dict) -> bool:
        """
        Write an ExtractedPaper to ChromaDB.
        Uses upsert — safe to call multiple times with same paper_id.

        The document text is what gets embedded and searched.
        Metadata fields are returned in query results for downstream use.
        """
        doc_text = self._build_document_text(paper)
        metadata = self._build_metadata(paper)
        paper_id = paper.get('paper_id') or paper.get('title', 'unknown')[:50]

        try:
            self.papers.upsert(
                ids=[paper_id],
                documents=[doc_text],
                metadatas=[metadata]
            )
            return True
        except Exception as e:
            print(f"  [KB] Store failed for '{paper.get('title', '?')[:40]}': {e}")
            return False

    def store_papers_batch(self, papers: List[dict]) -> int:
        """Store multiple papers, return count of successfully stored."""
        stored = 0
        for paper in papers:
            if self.store_paper(paper):
                stored += 1
        return stored

    # ── Read ─────────────────────────────────────────────────────────────────

    def query(self, query_text: str, n_results: int = 5) -> List[Dict]:
        """
        Semantic search over stored papers.
        Returns list of structured paper dicts, ordered by relevance.
        Used by Feasibility Critic to find relevant prior work.
        """
        if self.papers.count() == 0:
            return []

        results = self.papers.query(
            query_texts=[query_text],
            n_results=min(n_results, self.papers.count())
        )
        return self._unpack_results(results)

    def query_by_constraint(self, constraint_keywords: List[str], n_results: int = 5) -> List[Dict]:
        """
        Find papers with specific constraints.
        Useful for feasibility critic: 'find papers that require X'.
        """
        query = f"constraints requirements limitations: {', '.join(constraint_keywords)}"
        return self.query(query, n_results)

    def get_all(self) -> List[Dict]:
        """Return all stored papers (for inspection/debugging)."""
        if self.papers.count() == 0:
            return []
        raw = self.papers.get(include=['documents', 'metadatas'])
        return self._unpack_get_results(raw)

    def count(self) -> int:
        return self.papers.count()

    def clear(self):
        """Delete all papers. Useful for testing."""
        self.client.delete_collection("aerospace_papers")
        self.papers = self.client.get_or_create_collection(
            name="aerospace_papers",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_document_text(self, paper: dict) -> str:
        """
        Build the text that gets embedded. Combines all semantic fields.
        This is what ChromaDB searches over — ordering matters for relevance.
        """
        parts = [
            f"Title: {paper.get('title', '')}",
            f"Problem: {paper.get('problem_addressed', '')}",
            f"Method: {paper.get('proposed_method', '')}",
            f"Results: {paper.get('results_metrics', '')}",
            f"Constraints: {', '.join(paper.get('constraints', []))}",
            f"Assumptions: {', '.join(paper.get('key_assumptions', []))}",
            f"Abstract: {paper.get('abstract', '')[:500]}",
        ]
        return "\n".join(p for p in parts if p.split(': ', 1)[1].strip())

    def _build_metadata(self, paper: dict) -> dict:
        """
        Build ChromaDB metadata dict.
        ALL values must be str/int/float/bool — no nested objects.
        Lists are JSON-encoded.
        """
        return {
            'title': str(paper.get('title', '')),
            'authors': json.dumps(paper.get('authors', [])),
            'year': str(paper.get('year', '')),
            'source': str(paper.get('source', '')),
            'paper_id': str(paper.get('paper_id', '')),
            'problem_addressed': str(paper.get('problem_addressed', '')),
            'proposed_method': str(paper.get('proposed_method', '')),
            'results_metrics': str(paper.get('results_metrics', '')),
            'constraints': json.dumps(paper.get('constraints', [])),
            'key_assumptions': json.dumps(paper.get('key_assumptions', [])),
            'abstract': str(paper.get('abstract', ''))[:1000],  # ChromaDB metadata size limit
            'stored_at': datetime.now().isoformat()
        }

    def _unpack_results(self, results: dict) -> List[Dict]:
        """Convert ChromaDB query response to list of paper dicts."""
        papers = []
        metadatas = results.get('metadatas', [[]])[0]
        distances = results.get('distances', [[]])[0]

        for meta, dist in zip(metadatas, distances):
            paper = self._deserialize_metadata(meta)
            paper['relevance_score'] = round(1 - dist, 3)  # cosine → similarity
            papers.append(paper)

        return papers

    def _unpack_get_results(self, raw: dict) -> List[Dict]:
        """Convert ChromaDB get() response to list of paper dicts."""
        papers = []
        for meta in raw.get('metadatas', []):
            papers.append(self._deserialize_metadata(meta))
        return papers

    def _deserialize_metadata(self, meta: dict) -> dict:
        """Reverse the JSON encoding of list fields."""
        paper = dict(meta)
        for list_field in ['authors', 'constraints', 'key_assumptions']:
            if list_field in paper:
                try:
                    paper[list_field] = json.loads(paper[list_field])
                except (json.JSONDecodeError, TypeError):
                    paper[list_field] = []
        return paper
