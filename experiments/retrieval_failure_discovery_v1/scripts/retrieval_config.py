"""Deterministic retrieval configuration helpers for the KCSC Pilot.

This module does not execute retrieval.  It defines the query construction,
passage assembly, tokenizer calls, and rank fusion that must be frozen before
the RQ1 baseline run.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "experiments/retrieval_failure_discovery_v1/retrieval_config_v1.json"

CODE_TYPES = (
    "KRACS",
    "KRCCS",
    "EXCS",
    "KWCS",
    "LHCS",
    "NHCS",
    "SMCS",
    "KDS",
    "KCS",
)
CODE_PATTERN = re.compile(
    rf"(?<![A-Z])(?P<kind>{'|'.join(CODE_TYPES)})\s*[:\-]?\s*"
    r"(?P<a>\d{2})\s*(?P<b>\d{2})\s*(?P<c>\d{2})(?!\d)",
    re.IGNORECASE,
)
TOKEN_PATTERN = re.compile(
    r"[A-Za-z]+(?:-[A-Za-z0-9]+)*|\d+(?:\.\d+)?(?:[A-Za-z가-힣]+)?|[가-힣]+"
)
PARTICLE_SUFFIXES = (
    "으로부터",
    "에서는",
    "에게서",
    "에서",
    "에게",
    "까지",
    "부터",
    "보다",
    "처럼",
    "으로",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "의",
    "와",
    "과",
    "도",
    "만",
    "로",
)
SHORT_STOPWORDS = {
    "그",
    "및",
    "또",
    "등",
    "때",
    "중",
    "내",
    "한",
    "의",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "에",
    "와",
    "과",
    "도",
    "만",
    "로",
}
QUERY_STOPWORDS = {
    "어떻게",
    "무엇인가",
    "무엇인",
    "무엇을",
    "하는가",
    "해야",
    "있는가",
    "인가",
    "되는가",
    "답하라",
    "설명하라",
    "각각",
    "얼마인가",
    "얼마인",
    "어느",
    "어떤",
}


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported retrieval config schema")
    return config


def normalize_unicode_and_space(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return " ".join(normalized.split())


def canonical_code(match: re.Match[str]) -> str:
    kind = match.group("kind").upper()
    digits = "".join(match.group(name) for name in ("a", "b", "c"))
    return f"{kind} {digits}"


def strip_particle(token: str) -> str:
    if not re.fullmatch(r"[가-힣]+", token):
        return token
    for suffix in PARTICLE_SUFFIXES:
        if token.endswith(suffix):
            stem = token[: -len(suffix)]
            if len(stem) >= 2:
                return stem
    return token


def visible_length(text: str) -> int:
    return sum(not character.isspace() for character in text)


def lexical_terms(question: str) -> list[str]:
    """Return fixed, question-only terms for SQLite trigram FTS5."""

    normalized = normalize_unicode_and_space(question)
    normalized = re.sub(r"(?<=\d),(?=\d{3}\b)", "", normalized)
    codes: list[str] = []

    def collect_code(match: re.Match[str]) -> str:
        codes.append(canonical_code(match))
        return " "

    remainder = CODE_PATTERN.sub(collect_code, normalized)
    raw_tokens = [
        token
        for token in TOKEN_PATTERN.findall(remainder)
        if token not in SHORT_STOPWORDS and token not in QUERY_STOPWORDS
    ]
    tokens = [strip_particle(token) for token in raw_tokens]

    candidates = [*codes]
    # Preserve the original surface form as well as a conservative particle-
    # stripped form. This avoids turning a suffix heuristic into information
    # loss while still making "기준은" searchable as "기준" in short phrases.
    candidates.extend(token for token in raw_tokens if visible_length(token) >= 3)
    candidates.extend(token for token in tokens if visible_length(token) >= 3)
    for left, right in zip(tokens, tokens[1:]):
        if visible_length(left) < 3 and visible_length(right) < 3:
            phrase = f"{left} {right}"
            if visible_length(phrase) >= 3:
                candidates.append(phrase)

    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    if not unique:
        raise ValueError("no trigram-compatible lexical term in question")
    return unique


def quote_fts_phrase(term: str) -> str:
    return '"' + term.replace('"', '""') + '"'


def build_fts_query(question: str) -> str:
    return " OR ".join(quote_fts_phrase(term) for term in lexical_terms(question))


def value(row: Mapping[str, Any], key: str, fallback: str = "") -> str:
    raw = row.get(key, fallback)
    return "" if raw is None else str(raw).strip()


def assemble_passage(row: Mapping[str, Any]) -> str:
    body = value(row, "text_search", value(row, "text"))
    section = " ".join(part for part in (value(row, "title"), value(row, "label")) if part)
    return (
        f"[CODE] {value(row, 'code')}\n"
        f"[DOCUMENT] {value(row, 'document_name')}\n"
        f"[SECTION] {section}\n"
        f"[TEXT] {body}"
    )


def tokenize_dense_query(tokenizer: Any, question: str, max_length: int = 128) -> Any:
    return tokenizer(
        normalize_unicode_and_space(question),
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors="pt",
    )


def tokenize_dense_passage(
    tokenizer: Any,
    row: Mapping[str, Any],
    max_length: int = 1024,
) -> Any:
    return tokenizer(
        assemble_passage(row),
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors="pt",
    )


def tokenize_reranker_pair(
    tokenizer: Any,
    question: str,
    row: Mapping[str, Any],
    max_length: int = 1536,
) -> Any:
    return tokenizer(
        normalize_unicode_and_space(question),
        assemble_passage(row),
        truncation="only_second",
        max_length=max_length,
        padding=False,
        return_tensors="pt",
    )


def reciprocal_rank_fusion(
    rankings: Iterable[Sequence[int]],
    k0: int = 60,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if k0 <= 0:
        raise ValueError("RRF k0 must be positive")

    contributions: dict[int, list[float]] = {}
    source_count: dict[int, int] = {}
    for ranking in rankings:
        seen_in_ranking: set[int] = set()
        rank = 0
        for raw_section_id in ranking:
            section_id = int(raw_section_id)
            if section_id in seen_in_ranking:
                continue
            seen_in_ranking.add(section_id)
            rank += 1
            contributions.setdefault(section_id, []).append(1.0 / (k0 + rank))
            source_count[section_id] = source_count.get(section_id, 0) + 1

    ordered = sorted(
        contributions,
        key=lambda section_id: (-math.fsum(contributions[section_id]), section_id),
    )
    if limit is not None:
        ordered = ordered[:limit]
    return [
        {
            "rank": rank,
            "section_id": section_id,
            "rrf_score": math.fsum(contributions[section_id]),
            "source_count": source_count[section_id],
        }
        for rank, section_id in enumerate(ordered, start=1)
    ]
