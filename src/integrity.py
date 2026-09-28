"""수집 응답의 스키마와 요청·응답 식별자를 공통 검증합니다."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

DOCUMENT_FIELDS = ("no", "codeType", "code", "fullCode", "name", "version", "updateDate")
IDENTITY_FIELDS = ("no", "codeType", "code", "fullCode")
EXPECTED_DOCUMENT_KEYS = frozenset((*DOCUMENT_FIELDS, "list"))
EXPECTED_SECTION_KEYS = frozenset(("no", "sort", "title", "level", "label", "contents"))
CURRENT_METADATA_SCHEMA_VERSION = 2
CURRENT_RESPONSE_STATUSES = frozenset(("saved", "empty", "identity_mismatch", "schema_error"))


def sha256_file(path: Path) -> str:
    """파일을 메모리에 전부 올리지 않고 SHA-256을 계산합니다."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def response_identity(document: dict[str, Any] | None) -> dict[str, Any]:
    """메타데이터에 기록할 응답 문서 식별자를 만듭니다."""

    if document is None:
        return {f"response{field[0].upper()}{field[1:]}": None for field in DOCUMENT_FIELDS}
    return {
        f"response{field[0].upper()}{field[1:]}": document.get(field)
        for field in DOCUMENT_FIELDS
    }


def analyze_document_response(item: dict[str, Any], data: Any) -> dict[str, Any]:
    """한 API 응답을 보존 가능 여부와 검색 격리 여부로 분류합니다."""

    anomalies: list[dict[str, Any]] = []
    if not isinstance(data, list):
        return {
            "status": "schema_error",
            "identityStatus": "unknown",
            "documentCount": 0,
            "sectionCount": 0,
            "response": response_identity(None),
            "anomalies": [{"kind": "response_not_array", "severity": "error"}],
        }
    if not data:
        return {
            "status": "empty",
            "identityStatus": "empty",
            "documentCount": 0,
            "sectionCount": 0,
            "response": response_identity(None),
            "anomalies": anomalies,
        }
    if len(data) != 1 or not isinstance(data[0], dict):
        return {
            "status": "schema_error",
            "identityStatus": "unknown",
            "documentCount": len(data),
            "sectionCount": 0,
            "response": response_identity(data[0] if data and isinstance(data[0], dict) else None),
            "anomalies": [{"kind": "unexpected_document_count", "severity": "error", "actual": len(data)}],
        }

    document = data[0]
    missing_document_keys = sorted(EXPECTED_DOCUMENT_KEYS - document.keys())
    sections = document.get("list")
    if missing_document_keys or not isinstance(sections, list):
        anomalies.append({
            "kind": "document_schema_error",
            "severity": "error",
            "missingKeys": missing_document_keys,
            "listType": type(sections).__name__,
        })

    mismatches = {
        field: {"requested": item.get(field), "returned": document.get(field)}
        for field in IDENTITY_FIELDS
        if str(item.get(field)) != str(document.get(field))
    }
    identity_status = "mismatch" if mismatches else "match"
    if mismatches:
        anomalies.append({
            "kind": "identity_mismatch",
            "severity": "quarantine",
            "fields": mismatches,
        })

    section_count = 0
    duplicate_sorts: dict[int, int] = {}
    if isinstance(sections, list):
        sort_counts: dict[int, int] = {}
        for ordinal, section in enumerate(sections):
            section_count += 1
            if not isinstance(section, dict):
                anomalies.append({"kind": "section_not_object", "severity": "error", "ordinal": ordinal})
                continue
            missing_section_keys = sorted(EXPECTED_SECTION_KEYS - section.keys())
            if missing_section_keys:
                anomalies.append({
                    "kind": "section_schema_error",
                    "severity": "error",
                    "ordinal": ordinal,
                    "missingKeys": missing_section_keys,
                })
            api_sort = section.get("sort")
            if isinstance(api_sort, int):
                sort_counts[api_sort] = sort_counts.get(api_sort, 0) + 1
        duplicate_sorts = {key: value for key, value in sort_counts.items() if value > 1}
        if duplicate_sorts:
            anomalies.append({
                "kind": "duplicate_section_sort",
                "severity": "warning",
                "values": duplicate_sorts,
            })

    has_schema_error = any(anomaly["severity"] == "error" for anomaly in anomalies)
    if has_schema_error:
        status = "schema_error"
    elif mismatches:
        status = "identity_mismatch"
    else:
        status = "saved"
    return {
        "status": status,
        "identityStatus": identity_status,
        "documentCount": len(data),
        "sectionCount": section_count,
        "response": response_identity(document),
        "anomalies": anomalies,
    }
