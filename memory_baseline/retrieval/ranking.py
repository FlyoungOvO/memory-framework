from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime
from typing import Any

import numpy as np

from memory_baseline.core.utils import timestamp_sort_key


TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "which",
    "who",
    "with",
}
QUERY_KEYWORD_STOPWORDS = STOPWORDS | {
    "about",
    "also",
    "answer",
    "anything",
    "ask",
    "asked",
    "asking",
    "chat",
    "conversation",
    "conversations",
    "current",
    "currently",
    "day",
    "earlier",
    "earliest",
    "first",
    "latest",
    "last",
    "later",
    "memory",
    "mention",
    "mentioned",
    "now",
    "recent",
    "recently",
    "remember",
    "said",
    "say",
    "session",
    "sessions",
    "tell",
    "told",
    "today",
    "user",
    "yesterday",
    "you",
    "your",
}


class BM25Index:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = [_tokenize(text) for text in texts]
        self.doc_lengths = [len(doc) for doc in self.documents]
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        doc_freq: Counter[str] = Counter()
        for doc in self.documents:
            doc_freq.update(set(doc))
        total = len(self.documents)
        self.idf = {term: math.log(1 + (total - freq + 0.5) / (freq + 0.5)) for term, freq in doc_freq.items()}
        self.term_freqs = [Counter(doc) for doc in self.documents]

    def scores(self, query: str) -> np.ndarray:
        query_terms = keyword_query_terms(query)
        scores = np.zeros(len(self.documents), dtype=np.float32)
        if not query_terms or not self.documents or self.avgdl == 0:
            return scores
        for idx, term_freq in enumerate(self.term_freqs):
            doc_len = self.doc_lengths[idx]
            score = 0.0
            for term in query_terms:
                freq = term_freq.get(term, 0)
                if not freq:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += self.idf.get(term, 0.0) * (freq * (self.k1 + 1) / denom)
            scores[idx] = score
        return scores


def rank_indices(
    *,
    query: str,
    dense_scores: np.ndarray,
    raw_turns: list[dict[str, Any]],
    retrieval_texts: list[str],
    candidate_indices: list[int],
    top_k: int,
    retrieval_method: str = "dense",
    question_type: str | None = None,
    question_date: str | None = None,
    temporal_boost: float = 0.0,
    store: Any = None,
) -> tuple[list[int], np.ndarray, np.ndarray | None]:
    candidate_scores = dense_scores[candidate_indices].astype(np.float32, copy=True)
    bm25_scores = None
    if retrieval_method == "hybrid":
        bm25_scores = _bm25_scores(store, retrieval_texts, query)
        candidate_scores = _rrf_scores(candidate_scores, bm25_scores[candidate_indices])
    elif retrieval_method != "dense":
        raise ValueError(f"Unsupported retrieval_method: {retrieval_method}")

    if temporal_boost and question_type in {"temporal-reasoning", "temporal"}:
        candidate_scores = candidate_scores + temporal_boost * _temporal_scores(raw_turns, candidate_indices, question_date)

    top_n = min(top_k, len(candidate_indices))
    ranked_candidate_positions = np.argsort(-candidate_scores)[:top_n]
    indices = [candidate_indices[int(pos)] for pos in ranked_candidate_positions]
    score_by_index = np.zeros(len(raw_turns), dtype=np.float32)
    score_by_index[candidate_indices] = candidate_scores
    return indices, score_by_index, bm25_scores


def _tokenize(text: str) -> list[str]:
    return [
        term
        for match in TOKEN_RE.finditer(text)
        if (term := match.group(0).lower()) not in STOPWORDS
    ]


def keyword_query_terms(text: str) -> list[str]:
    terms: list[str] = []
    for match in TOKEN_RE.finditer(text):
        term = match.group(0).lower()
        if term in QUERY_KEYWORD_STOPWORDS:
            continue
        if len(term) < 3 and not any(char.isdigit() for char in term):
            continue
        terms.append(term)
    return terms


def _bm25_scores(store: Any, retrieval_texts: list[str], query: str) -> np.ndarray:
    if store is not None and getattr(store, "_bm25_index", None) is not None:
        return store._bm25_index.scores(query)
    index = BM25Index(retrieval_texts)
    if store is not None:
        store._bm25_index = index
    return index.scores(query)


def _rrf_scores(scores1: np.ndarray, scores2: np.ndarray, k: int = 60) -> np.ndarray:
    fused = 1.0 / (k + _ranks_desc(scores1))
    positive = scores2 > 0
    if np.any(positive):
        fused[positive] += 1.0 / (k + _ranks_desc(scores2[positive]))
    return fused


def _ranks_desc(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores)
    ranks = np.empty(len(scores), dtype=np.float32)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float32)
    return ranks


def _temporal_scores(raw_turns: list[dict[str, Any]], candidate_indices: list[int], question_date: str | None) -> np.ndarray:
    dates = [_as_datetime(raw_turns[idx].get("session_date")) for idx in candidate_indices]
    question_dt = _as_datetime(question_date)
    valid_dates = [date for date in dates if date is not None and (question_dt is None or date <= question_dt)]
    if not valid_dates:
        return np.zeros(len(candidate_indices), dtype=np.float32)
    min_ts = min(date.timestamp() for date in valid_dates)
    max_ts = max(date.timestamp() for date in valid_dates)
    span = max(max_ts - min_ts, 1.0)
    scores = []
    for date in dates:
        if date is None or (question_dt is not None and date > question_dt):
            scores.append(0.0)
        else:
            scores.append((date.timestamp() - min_ts) / span)
    return np.asarray(scores, dtype=np.float32)


def _as_datetime(value: Any) -> datetime | None:
    key = timestamp_sort_key(str(value) if value is not None else None)
    if key[0] == 0 and isinstance(key[1], datetime):
        return key[1]
    return None
