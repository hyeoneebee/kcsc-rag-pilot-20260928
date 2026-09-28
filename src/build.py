"""검증된 스냅샷에서 구조화 파일과 SQLite 검색 DB를 생성합니다."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sqlite3
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO
from zoneinfo import ZoneInfo

from .html_text import html_to_search_text
from .integrity import analyze_document_response, sha256_file

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEOUL = ZoneInfo("Asia/Seoul")


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT, check=False, capture_output=True, text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def write_json_line(file: TextIO, value: dict[str, Any]) -> None:
    file.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    file.write("\n")


def verify_snapshot(snapshot_dir: Path) -> dict[str, Any]:
    manifest_path = snapshot_dir / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("manifest.json이 없는 디렉터리는 빌드 원천으로 사용할 수 없습니다.")
    manifest = load_json(manifest_path)
    for entry in manifest.get("files", []):
        path = snapshot_dir / entry["path"]
        if not path.exists():
            raise ValueError(f"스냅샷 파일 누락: {entry['path']}")
        if path.stat().st_size != entry["size"] or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"스냅샷 파일 손상: {entry['path']}")
    return manifest


def category_model(catalog: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[tuple[str, str], int], dict[int, list[tuple[str, str]]]]:
    nodes: dict[tuple[str, str], dict[str, Any]] = {}
    parents: dict[tuple[str, str], tuple[str, str] | None] = {}
    document_paths: dict[int, list[tuple[str, str]]] = {}
    for document_id, item in enumerate(catalog, start=1):
        path_keys: list[tuple[str, str]] = []
        previous: tuple[str, str] | None = None
        for position, node in enumerate(item.get("listParentCodes") or [], start=1):
            key = (str(node.get("codeType")), str(node.get("fullCode")))
            name = node.get("name")
            if name is None and str(node.get("fullCode")) == str(item.get("fullCode")):
                name = item.get("name")
            if key not in nodes:
                nodes[key] = {
                    "codeType": key[0],
                    "fullCode": key[1],
                    "name": name,
                    "depth": position,
                }
                parents[key] = previous
            elif nodes[key]["name"] is None and name is not None:
                nodes[key]["name"] = name
            if parents[key] != previous:
                raise ValueError(f"카테고리 복수 부모 발견: {key}")
            path_keys.append(key)
            previous = key
        document_paths[document_id] = path_keys
    ordered_keys = sorted(nodes, key=lambda key: (key[0], len(key[1]), key[1]))
    ids = {key: category_id for category_id, key in enumerate(ordered_keys, start=1)}
    rows = []
    for key in ordered_keys:
        parent = parents[key]
        rows.append({
            "categoryId": ids[key],
            **nodes[key],
            "parentCategoryId": ids[parent] if parent else None,
        })
    return rows, ids, document_paths


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE build_info (
    key TEXT PRIMARY KEY NOT NULL,
    value TEXT NOT NULL
) STRICT;
CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY,
    code_type TEXT NOT NULL,
    full_code TEXT NOT NULL,
    name TEXT,
    depth INTEGER NOT NULL,
    parent_category_id INTEGER REFERENCES categories(category_id),
    UNIQUE(code_type, full_code)
) STRICT;
CREATE TABLE documents (
    document_id INTEGER PRIMARY KEY,
    catalog_no INTEGER NOT NULL UNIQUE,
    requested_code_type TEXT NOT NULL,
    requested_code TEXT NOT NULL,
    requested_full_code TEXT NOT NULL,
    catalog_name TEXT NOT NULL,
    version TEXT NOT NULL,
    update_date TEXT NOT NULL,
    response_status TEXT NOT NULL,
    response_no INTEGER,
    response_code_type TEXT,
    response_code TEXT,
    response_full_code TEXT,
    response_name TEXT,
    response_version TEXT,
    response_update_date TEXT,
    raw_file TEXT NOT NULL,
    raw_sha256 TEXT NOT NULL,
    section_count INTEGER NOT NULL CHECK(section_count >= 0),
    is_searchable INTEGER NOT NULL CHECK(is_searchable IN (0, 1)),
    UNIQUE(requested_code_type, requested_code)
) STRICT;
CREATE TABLE document_categories (
    document_id INTEGER NOT NULL REFERENCES documents(document_id),
    position INTEGER NOT NULL,
    category_id INTEGER NOT NULL REFERENCES categories(category_id),
    PRIMARY KEY(document_id, position)
) STRICT;
CREATE TABLE sections (
    section_id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(document_id),
    ordinal INTEGER NOT NULL,
    api_no INTEGER NOT NULL,
    api_sort INTEGER NOT NULL,
    level INTEGER NOT NULL,
    label TEXT,
    title TEXT NOT NULL,
    html_original TEXT NOT NULL,
    text_search TEXT NOT NULL,
    has_table INTEGER NOT NULL CHECK(has_table IN (0, 1)),
    has_image INTEGER NOT NULL CHECK(has_image IN (0, 1)),
    json_pointer TEXT NOT NULL,
    UNIQUE(document_id, ordinal)
) STRICT;
CREATE TABLE searchable_sections (
    section_id INTEGER PRIMARY KEY REFERENCES sections(section_id),
    document_id INTEGER NOT NULL REFERENCES documents(document_id),
    code TEXT NOT NULL,
    document_name TEXT NOT NULL,
    title TEXT NOT NULL,
    label TEXT,
    text_search TEXT NOT NULL
) STRICT;
CREATE TABLE anomalies (
    anomaly_id INTEGER PRIMARY KEY,
    document_id INTEGER REFERENCES documents(document_id),
    kind TEXT NOT NULL,
    severity TEXT NOT NULL,
    details_json TEXT NOT NULL
) STRICT;
CREATE INDEX sections_document_idx ON sections(document_id, ordinal);
CREATE INDEX documents_type_code_idx ON documents(requested_code_type, requested_code);
CREATE INDEX document_categories_category_idx ON document_categories(category_id);
"""


def initialize_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode = OFF")
    connection.execute("PRAGMA synchronous = OFF")
    connection.executescript(SCHEMA)
    return connection


def build(snapshot_dir: Path, output_root: Path, label: str) -> Path:
    manifest = verify_snapshot(snapshot_dir)
    validation = load_json(snapshot_dir / "reports" / "validation-summary.json")
    if not validation.get("validForBuild"):
        raise ValueError("검증 오류가 있는 스냅샷은 구조화할 수 없습니다.")
    final_dir = output_root / label
    temporary_dir = output_root / f".{label}.tmp"
    if final_dir.exists() or temporary_dir.exists():
        raise FileExistsError(f"같은 이름의 구조화 결과가 이미 있습니다: {label}")
    output_root.mkdir(parents=True, exist_ok=True)
    temporary_dir.mkdir()

    catalog = load_json(snapshot_dir / "catalog.json")
    categories, category_ids, document_paths = category_model(catalog)
    database_path = temporary_dir / "kcsc.sqlite3"
    connection = initialize_database(database_path)
    counts: Counter[str] = Counter()
    section_id = 0
    anomaly_id = 0
    started_at = datetime.now(SEOUL)
    try:
        with (
            (temporary_dir / "documents.jsonl").open("w", encoding="utf-8") as documents_file,
            (temporary_dir / "categories.jsonl").open("w", encoding="utf-8") as categories_file,
            (temporary_dir / "document_paths.jsonl").open("w", encoding="utf-8") as paths_file,
            gzip.open(temporary_dir / "sections.jsonl.gz", "wt", encoding="utf-8", compresslevel=6) as sections_file,
            gzip.open(temporary_dir / "quarantined_sections.jsonl.gz", "wt", encoding="utf-8", compresslevel=6) as quarantine_file,
            (temporary_dir / "anomalies.jsonl").open("w", encoding="utf-8") as anomalies_file,
        ):
            connection.execute("BEGIN")
            for row in categories:
                write_json_line(categories_file, row)
                connection.execute(
                    "INSERT INTO categories VALUES (?, ?, ?, ?, ?, ?)",
                    (row["categoryId"], row["codeType"], row["fullCode"], row["name"], row["depth"], row["parentCategoryId"]),
                )

            for document_id, item in enumerate(catalog, start=1):
                code_type = str(item["codeType"])
                code = str(item["code"])
                raw_relative = Path("raw") / code_type / f"{code}.json"
                metadata_relative = Path("metadata") / code_type / f"{code}.json"
                response = load_json(snapshot_dir / raw_relative)
                metadata = load_json(snapshot_dir / metadata_relative)
                analysis = analyze_document_response(item, response)
                status = analysis["status"]
                is_searchable = status == "saved"
                response_values = analysis["response"]
                document_row = {
                    "documentId": document_id,
                    "requestedCodeType": code_type,
                    "requestedCode": code,
                    "requestedFullCode": str(item["fullCode"]),
                    "catalogNo": item["no"],
                    "catalogName": item["name"],
                    "version": str(item["version"]),
                    "updateDate": item["updateDate"],
                    "responseStatus": status,
                    "identityStatus": analysis["identityStatus"],
                    "response": response_values,
                    "rawFile": str(raw_relative),
                    "rawSha256": metadata["sha256"],
                    "sectionCount": analysis["sectionCount"],
                    "isSearchable": is_searchable,
                }
                write_json_line(documents_file, document_row)
                connection.execute(
                    "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        document_id, item["no"], code_type, code, str(item["fullCode"]), item["name"],
                        str(item["version"]), item["updateDate"], status,
                        response_values["responseNo"], response_values["responseCodeType"],
                        response_values["responseCode"], response_values["responseFullCode"],
                        response_values["responseName"], response_values["responseVersion"],
                        response_values["responseUpdateDate"], str(raw_relative), metadata["sha256"],
                        analysis["sectionCount"], int(is_searchable),
                    ),
                )
                for position, category_key in enumerate(document_paths[document_id], start=1):
                    path_row = {
                        "documentId": document_id,
                        "position": position,
                        "categoryId": category_ids[category_key],
                    }
                    write_json_line(paths_file, path_row)
                    connection.execute(
                        "INSERT INTO document_categories VALUES (?, ?, ?)",
                        (document_id, position, category_ids[category_key]),
                    )

                if status == "empty":
                    detail = {"kind": "empty_response", "severity": "warning"}
                    anomaly_id += 1
                    write_json_line(anomalies_file, {"documentId": document_id, **detail})
                    connection.execute(
                        "INSERT INTO anomalies VALUES (?, ?, ?, ?, ?)",
                        (anomaly_id, document_id, detail["kind"], detail["severity"], "{}"),
                    )
                for detail in analysis["anomalies"]:
                    anomaly_id += 1
                    payload = {key: value for key, value in detail.items() if key not in {"kind", "severity"}}
                    write_json_line(anomalies_file, {"documentId": document_id, **detail})
                    connection.execute(
                        "INSERT INTO anomalies VALUES (?, ?, ?, ?, ?)",
                        (anomaly_id, document_id, detail["kind"], detail["severity"], json.dumps(payload, ensure_ascii=False)),
                    )

                if response and isinstance(response[0], dict):
                    for index, section in enumerate(response[0].get("list") or []):
                        section_id += 1
                        html_original = section["contents"]
                        text_search, has_table, has_image = html_to_search_text(html_original)
                        row = {
                            "sectionId": section_id,
                            "documentId": document_id,
                            "ordinal": index + 1,
                            "apiNo": section["no"],
                            "apiSort": section["sort"],
                            "level": section["level"],
                            "label": section["label"],
                            "title": section["title"],
                            "htmlOriginal": html_original,
                            "textSearch": text_search,
                            "hasTable": has_table,
                            "hasImage": has_image,
                            "jsonPointer": f"/0/list/{index}",
                        }
                        write_json_line(sections_file if is_searchable else quarantine_file, row)
                        connection.execute(
                            "INSERT INTO sections VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (
                                section_id, document_id, index + 1, section["no"], section["sort"],
                                section["level"], section["label"], section["title"], html_original,
                                text_search, int(has_table), int(has_image), f"/0/list/{index}",
                            ),
                        )
                        if is_searchable:
                            connection.execute(
                                "INSERT INTO searchable_sections VALUES (?, ?, ?, ?, ?, ?, ?)",
                                (
                                    section_id, document_id, f"{code_type} {code}", item["name"],
                                    section["title"], section["label"], text_search,
                                ),
                            )
                counts[status] += 1
                if document_id % 250 == 0:
                    print(f"구조화 진행: {document_id}/{len(catalog)} 문서")

            connection.execute(
                "CREATE VIRTUAL TABLE sections_fts USING fts5("
                "code, document_name, title, label, text_search, "
                "content='searchable_sections', content_rowid='section_id', tokenize='trigram')"
            )
            connection.execute("INSERT INTO sections_fts(sections_fts) VALUES ('rebuild')")
            build_info = {
                "label": label,
                "builtAt": datetime.now(SEOUL).isoformat(),
                "sourceManifestSha256": sha256_file(snapshot_dir / "manifest.json"),
                "htmlPolicy": "Original HTML is preserved; only derived text is indexed",
                "ftsTokenizer": "trigram",
            }
            connection.executemany(
                "INSERT INTO build_info VALUES (?, ?)",
                [(key, json.dumps(value, ensure_ascii=False)) for key, value in build_info.items()],
            )
            connection.commit()

        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        foreign_key_violations = connection.execute("PRAGMA foreign_key_check").fetchall()
        database_counts = {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("documents", "categories", "document_categories", "sections", "searchable_sections", "anomalies")
        }
        if integrity != "ok" or foreign_key_violations:
            raise ValueError(f"SQLite 검증 실패: integrity={integrity}, foreignKeys={len(foreign_key_violations)}")
        connection.execute("PRAGMA optimize")
        connection.execute("VACUUM")
        connection.close()

        finished_at = datetime.now(SEOUL)
        report = {
            "buildSchemaVersion": 1,
            "label": label,
            "startedAt": started_at.isoformat(),
            "finishedAt": finished_at.isoformat(),
            "durationSeconds": round((finished_at - started_at).total_seconds(), 3),
            "sourceSnapshot": str(snapshot_dir.resolve()),
            "sourceManifestSha256": sha256_file(snapshot_dir / "manifest.json"),
            "builderGitCommit": git_commit(),
            "documentStatusCounts": dict(sorted(counts.items())),
            "databaseCounts": database_counts,
            "sqliteIntegrityCheck": integrity,
            "foreignKeyViolationCount": len(foreign_key_violations),
            "databaseSha256": sha256_file(database_path),
            "databaseBytes": database_path.stat().st_size,
            "ftsTokenizer": "trigram",
            "htmlPolicy": "Original HTML is preserved; only derived text is indexed",
            "quarantinePolicy": "identity_mismatch and schema_error documents are preserved but excluded from search",
            "sourceCatalogCount": manifest["catalogCount"],
        }
        (temporary_dir / "build-report.json").write_bytes(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        )
        temporary_dir.replace(final_dir)
    except Exception:
        connection.close()
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise

    latest = output_root.parent / "kcsc.sqlite3"
    if latest.exists() and not latest.is_symlink():
        raise FileExistsError(f"최신 DB 링크 위치에 일반 파일이 있습니다: {latest}")
    if latest.is_symlink():
        latest.unlink()
    os.symlink(Path("processed") / label / "kcsc.sqlite3", latest)
    return final_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KCSC 구조화 파일과 SQLite 검색 DB 생성")
    parser.add_argument("label", help="원천 스냅샷 이름과 결과 이름")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    snapshot_dir = args.data_dir / "snapshots" / args.label
    try:
        result = build(snapshot_dir, args.data_dir / "processed", args.label)
    except Exception as error:
        print(f"구조화 실패: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"구조화 완료: {result}")
    print(f"검색 DB: {args.data_dir / 'kcsc.sqlite3'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
