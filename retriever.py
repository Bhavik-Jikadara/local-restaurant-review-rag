"""Hybrid retriever: Chroma MMR vector search + BM25 keywords, fused with RRF."""

from __future__ import annotations

import math
import re
from collections import Counter

from langchain_chroma import Chroma
from langchain_core.documents import Document

import config

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "do", "does", "for", "from",
    "has", "have", "how", "i", "in", "is", "it", "of", "on", "or", "that", "the", "there",
    "they", "this", "to", "was", "were", "what", "which", "with", "you", "about", "any",
    "people", "customers", "say", "said",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


class BM25Index:
    """Tiny dependency-free BM25 (Okapi) implementation."""

    def __init__(self, documents: list[Document], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1, self.b = k1, b
        self.tokens = [tokenize(d.page_content) for d in documents]
        self.lengths = [len(t) for t in self.tokens]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        self.term_freqs = [Counter(t) for t in self.tokens]
        doc_freq: Counter[str] = Counter()
        for tf in self.term_freqs:
            doc_freq.update(tf.keys())
        n = len(documents)
        self.idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()
        }

    def search(self, query: str, k: int, restaurant: str | None = None) -> list[Document]:
        terms = tokenize(query)
        scored: list[tuple[float, int]] = []
        for i, doc in enumerate(self.documents):
            if restaurant and doc.metadata.get("restaurant") != restaurant:
                continue
            tf, score = self.term_freqs[i], 0.0
            for term in terms:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                norm = freq + self.k1 * (1 - self.b + self.b * self.lengths[i] / (self.avg_len or 1))
                score += self.idf.get(term, 0.0) * freq * (self.k1 + 1) / norm
            if score > 0:
                scored.append((score, i))
        scored.sort(reverse=True)
        return [self.documents[i] for _, i in scored[:k]]


class HybridRetriever:
    def __init__(self, store: Chroma):
        self.store = store
        self.refresh()

    def refresh(self) -> None:
        data = self.store.get(include=["documents", "metadatas"])
        self.documents = [
            Document(page_content=text, metadata=meta or {}, id=doc_id)
            for doc_id, text, meta in zip(data["ids"], data["documents"] or [], data["metadatas"] or [])
        ]
        self.bm25 = BM25Index(self.documents)

    @property
    def restaurants(self) -> list[str]:
        return sorted({d.metadata.get("restaurant", "") for d in self.documents} - {""})

    def retrieve(
        self,
        query: str,
        restaurant: str | None = None,
        min_rating: float | None = None,
        max_rating: float | None = None,
        k: int = config.TOP_K,
    ) -> list[Document]:
        where = self._where(restaurant, min_rating, max_rating)

        vector_hits = self.store.max_marginal_relevance_search(
            query,
            k=config.KEYWORD_K,
            fetch_k=config.VECTOR_FETCH_K,
            lambda_mult=config.MMR_LAMBDA,
            filter=where,
        )
        keyword_hits = [
            d
            for d in self.bm25.search(query, config.KEYWORD_K * 3, restaurant)
            if self._rating_ok(d, min_rating, max_rating)
        ][: config.KEYWORD_K]

        # Reciprocal Rank Fusion: documents ranked well by either method rise to the top.
        scores: dict[str, float] = {}
        by_id: dict[str, Document] = {}
        for hits in (vector_hits, keyword_hits):
            for rank, doc in enumerate(hits):
                key = doc.id or doc.page_content
                by_id[key] = doc
                scores[key] = scores.get(key, 0.0) + 1.0 / (config.RRF_CONSTANT + rank + 1)
        ranked = sorted(scores, key=scores.get, reverse=True)  # type: ignore[arg-type]
        return [by_id[key] for key in ranked[:k]]

    @staticmethod
    def _rating_ok(doc: Document, low: float | None, high: float | None) -> bool:
        rating = doc.metadata.get("rating", -1)
        if low is not None and rating < low:
            return False
        return not (high is not None and rating > high)

    @staticmethod
    def _where(restaurant: str | None, low: float | None, high: float | None) -> dict | None:
        clauses: list[dict] = []
        if restaurant:
            clauses.append({"restaurant": restaurant})
        if low is not None:
            clauses.append({"rating": {"$gte": low}})
        if high is not None:
            clauses.append({"rating": {"$lte": high}})
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}


# ----------------------------------------------------------------------------
# Lightweight query understanding (no LLM call needed)
# ----------------------------------------------------------------------------
_NEGATIVE = re.compile(r"\b(complain\w*|negative|bad|worst|problems?|issues?|disappoint\w*|poor|dislike\w*)\b", re.I)
_POSITIVE = re.compile(r"\b(praise\w*|positive|best|great|love[sd]?|favou?rite|good things|highlights?)\b", re.I)


def detect_restaurant(question: str, restaurants: list[str]) -> str | None:
    """Return a restaurant if exactly one is named (full name or a distinctive word)."""
    lowered = question.lower()
    full = [r for r in restaurants if r.lower() in lowered]
    if len(full) == 1:
        return full[0]
    if full:
        return None

    generic = {"the", "house", "kitchen", "cafe", "restaurant", "pizzeria", "bar", "grill"}
    matches = set()
    for name in restaurants:
        words = [w for w in re.findall(r"[a-z]+", name.lower()) if w not in generic and len(w) > 3]
        if any(re.search(rf"\b{re.escape(w)}\b", lowered) for w in words):
            matches.add(name)
    return matches.pop() if len(matches) == 1 else None


def detect_rating_filter(question: str) -> tuple[float | None, float | None]:
    """Map sentiment words to rating ranges so 'complaints' prefers low-star reviews."""
    negative, positive = _NEGATIVE.search(question), _POSITIVE.search(question)
    if negative and not positive:
        return None, 3.0
    if positive and not negative:
        return 4.0, None
    return None, None
