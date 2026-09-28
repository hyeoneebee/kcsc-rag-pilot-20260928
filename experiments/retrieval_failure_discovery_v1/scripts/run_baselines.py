#!/usr/bin/env python3
"""Run the four frozen retrieval-only baselines on the KCSC Pilot corpus.

Heavy ML imports are intentionally local to runtime functions so the output and
ranking helpers can be unit-tested in the repository's lightweight environment.
No answer-generation model is called by this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

from retrieval_config import (
    assemble_passage,
    build_fts_query,
    load_config,
    normalize_unicode_and_space,
    reciprocal_rank_fusion,
)


ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = ROOT / "experiments/retrieval_failure_discovery_v1"
DEFAULT_PREFLIGHT = EXPERIMENT_ROOT / "runs/preflight_v1/run_manifest.json"
DEFAULT_QUESTIONS = EXPERIMENT_ROOT / "frozen_inputs/v1/normalized_questions.jsonl"
DEFAULT_CONFIG = EXPERIMENT_ROOT / "retrieval_config_v1.json"
DEFAULT_RUNS_DIR = EXPERIMENT_ROOT / "runs"
ARMS = ("R0-LEX", "R0-DENSE", "R0-HYBRID", "R0-HYBRID-RERANK")
MATERIAL_PASSPORT = {
    "origin_skill": "academic-research-suite / experiment-agent",
    "origin_mode": "run",
    "verification_status": "UNVERIFIED",
    "version_label": "kcsc_pilot_baselines_v1",
}


def now_iso() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def git(args: Sequence[str]) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_questions(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 48 or len({row["question_id"] for row in rows}) != 48:
        raise RuntimeError("frozen question input must contain 48 unique question IDs")
    answerable = [row for row in rows if row["scoring_group"] == "answerable_retrieval"]
    controls = [row for row in rows if row["scoring_group"] != "answerable_retrieval"]
    if len(answerable) != 46 or len(controls) != 2:
        raise RuntimeError("question partition must be 46 answerable plus 2 controls")
    if any(not row["gold_section_ids"] for row in answerable):
        raise RuntimeError("every answerable question must have exact provisional gold")
    if any(row["gold_section_ids"] for row in controls):
        raise RuntimeError("answerability controls must not have retrieval gold")
    return rows


def validate_runner_config(config: Mapping[str, Any]) -> None:
    """Fail visibly if the runner cannot honor a frozen configuration."""

    expected = {
        ("input", "query_fields"): ["question"],
        ("lexical", "query_builder"): "fts_or_v1",
        ("lexical", "operator"): "OR",
        ("lexical", "top_k"): 100,
        ("dense", "mode"): "dense_only",
        ("dense", "pooling"): "cls",
        ("dense", "compute_dtype"): "float16",
        ("dense", "embedding_storage_dtype"): "float32",
        ("dense", "similarity_accumulation_dtype"): "float32",
        ("dense", "similarity"): "exact_inner_product",
        ("dense", "approximate_index"): False,
        ("hybrid", "method"): "reciprocal_rank_fusion",
        ("reranker", "score"): "raw_logit",
    }
    mismatches = []
    for (group, key), value in expected.items():
        if config.get(group, {}).get(key) != value:
            mismatches.append(f"{group}.{key}={config.get(group, {}).get(key)!r}")
    weights = config.get("lexical", {}).get("bm25_column_weights")
    if weights != {"code": 8.0, "document_name": 5.0, "title": 3.0, "label": 2.0, "text_search": 1.0}:
        mismatches.append(f"lexical.bm25_column_weights={weights!r}")
    if mismatches:
        raise RuntimeError(
            "runner does not implement the frozen configuration: " + "; ".join(mismatches)
        )


def database_connection(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def validate_preflight(
    preflight_path: Path, config_path: Path, questions_path: Path
) -> tuple[dict[str, Any], Path]:
    preflight = load_json(preflight_path)
    if preflight.get("status") != "ready_for_baseline_execution":
        raise RuntimeError("preflight manifest is not ready for baseline execution")
    if preflight.get("execution_state", {}).get("baseline_scores_observed"):
        raise RuntimeError("preflight manifest says baseline scores were already observed")
    head = git(("rev-parse", "HEAD"))
    if preflight["repository"]["git_head"] != head:
        raise RuntimeError("preflight manifest does not bind the current Git commit")
    dirty = git(("status", "--porcelain", "--untracked-files=all"))
    if dirty:
        raise RuntimeError("repository must be clean before baseline execution:\n" + dirty)
    if sha256_file(config_path) != preflight["retrieval_config_sha256"]:
        raise RuntimeError("retrieval config hash differs from the preflight manifest")
    if sha256_file(questions_path) != preflight["data"]["normalized_questions_sha256"]:
        raise RuntimeError("question hash differs from the preflight manifest")
    database_path = ROOT / preflight["data"]["database_path"]
    if sha256_file(database_path) != preflight["data"]["database_sha256"]:
        raise RuntimeError("database hash differs from the preflight manifest")
    return preflight, database_path


def lexical_search(
    connection: sqlite3.Connection, question: str, limit: int
) -> tuple[str, list[dict[str, Any]]]:
    query = build_fts_query(question)
    rows = connection.execute(
        """
        SELECT sections_fts.rowid AS section_id,
               bm25(sections_fts, 8.0, 5.0, 3.0, 2.0, 1.0) AS score
        FROM sections_fts
        JOIN searchable_sections ss ON ss.section_id = sections_fts.rowid
        WHERE sections_fts MATCH ?
        ORDER BY score ASC, sections_fts.rowid ASC
        LIMIT ?
        """,
        (query, limit),
    ).fetchall()
    return query, [
        {"rank": rank, "section_id": int(row[0]), "score": float(row[1])}
        for rank, row in enumerate(rows, start=1)
    ]


def stable_top_k(
    scores: Any, section_ids: Any, limit: int
) -> list[dict[str, Any]]:
    """Take a descending float score top-k with section_id ascending ties."""

    import numpy as np

    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    ids = np.asarray(section_ids, dtype=np.int64).reshape(-1)
    if values.shape != ids.shape:
        raise ValueError("scores and section_ids must have the same shape")
    if not np.isfinite(values).all():
        raise ValueError("dense scores contain NaN or infinity")
    take = min(limit, len(values))
    if take == 0:
        return []
    if take == len(values):
        candidates = np.arange(len(values))
    else:
        partition = np.argpartition(-values, take - 1)[:take]
        boundary = values[partition].min()
        better = np.flatnonzero(values > boundary)
        tied = np.flatnonzero(values == boundary)
        tied = tied[np.argsort(ids[tied], kind="stable")]
        candidates = np.concatenate((better, tied[: take - len(better)]))
    ordered = sorted(candidates.tolist(), key=lambda i: (-float(values[i]), int(ids[i])))
    return [
        {"rank": rank, "section_id": int(ids[index]), "score": float(values[index])}
        for rank, index in enumerate(ordered[:take], start=1)
    ]


def rerank_by_score(
    section_ids: Sequence[int], scores: Sequence[float], limit: int
) -> list[dict[str, Any]]:
    if len(section_ids) != len(scores):
        raise ValueError("section IDs and scores must have equal length")
    if any(not math.isfinite(float(score)) for score in scores):
        raise ValueError("reranker scores contain NaN or infinity")
    ordered = sorted(
        zip(section_ids, scores), key=lambda item: (-float(item[1]), int(item[0]))
    )[:limit]
    return [
        {"rank": rank, "section_id": int(section_id), "score": float(score)}
        for rank, (section_id, score) in enumerate(ordered, start=1)
    ]


def document_code_key(code: str) -> tuple[str, str]:
    normalized = normalize_unicode_and_space(code).upper()
    parts = normalized.split(maxsplit=1)
    if len(parts) == 1:
        return "", "".join(character for character in parts[0] if character.isdigit())
    return parts[0], "".join(character for character in parts[1] if character.isdigit())


def source_applicable_at_1(question: Mapping[str, Any], hit: Mapping[str, Any]) -> int:
    hit_key = document_code_key(str(hit.get("code", "")))
    required = {
        (str(row.get("code_type", "")).upper(), "".join(filter(str.isdigit, str(row.get("code", "")))))
        for row in question.get("required_documents", [])
    }
    return int(hit_key in required)


def question_metrics(
    question: Mapping[str, Any], ranked_ids: Sequence[int], cutoffs: Iterable[int]
) -> dict[str, Any]:
    if question["scoring_group"] != "answerable_retrieval":
        return {"scoring_group": question["scoring_group"], "metrics": None}
    gold = {int(value) for value in question["gold_section_ids"]}
    ranks = {int(section_id): rank for rank, section_id in enumerate(ranked_ids, start=1)}
    metrics: dict[str, Any] = {}
    for cutoff in cutoffs:
        found = gold & set(map(int, ranked_ids[:cutoff]))
        first = min((ranks[value] for value in found), default=None)
        metrics[str(cutoff)] = {
            "any_gold_hit": int(bool(found)),
            "required_evidence_recall": len(found) / len(gold),
            "complete_evidence": int(found == gold),
            "mrr": 1.0 / first if first is not None else 0.0,
        }
    return {"scoring_group": question["scoring_group"], "metrics": metrics}


def load_section_ids(connection: sqlite3.Connection) -> Any:
    import numpy as np

    count = int(connection.execute("SELECT count(*) FROM searchable_sections").fetchone()[0])
    ids = np.fromiter(
        (
            int(row[0])
            for row in connection.execute(
                "SELECT section_id FROM searchable_sections ORDER BY section_id"
            )
        ),
        dtype=np.int64,
        count=count,
    )
    return ids


def model_revision(model: Any) -> str | None:
    return getattr(getattr(model, "config", None), "_commit_hash", None)


def load_dense_model(config: Mapping[str, Any], device: Any) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"], revision=config["revision"]
    )
    model = AutoModel.from_pretrained(
        config["model"], revision=config["revision"], dtype=torch.float16
    ).to(device)
    model.eval()
    actual_revision = model_revision(model)
    if actual_revision and actual_revision != config["revision"]:
        raise RuntimeError(f"dense revision mismatch: {actual_revision}")
    return tokenizer, model


def dense_forward(model: Any, inputs: Mapping[str, Any]) -> Any:
    import torch.nn.functional as functional

    cls = model(**inputs, return_dict=True).last_hidden_state[:, 0]
    return functional.normalize(cls.float(), p=2, dim=1)


def dense_cache_key(preflight: Mapping[str, Any], config_hash: str) -> str:
    source = "|".join(
        (
            preflight["data"]["database_sha256"],
            config_hash,
            preflight["retrieval_config"]["dense"]["revision"],
        )
    )
    return hashlib.sha256(source.encode("ascii")).hexdigest()[:20]


def build_or_load_embeddings(
    connection: sqlite3.Connection,
    section_ids: Any,
    tokenizer: Any,
    model: Any,
    device: Any,
    config: Mapping[str, Any],
    cache_dir: Path,
) -> tuple[Any, dict[str, Any]]:
    import numpy as np
    import torch

    cache_dir.mkdir(parents=True, exist_ok=True)
    final_path = cache_dir / "passages.f32.npy"
    partial_path = cache_dir / "passages.partial.f32.npy"
    state_path = cache_dir / "state.json"
    expected = {
        "section_count": int(len(section_ids)),
        "section_ids_sha256": hashlib.sha256(section_ids.tobytes()).hexdigest(),
        "model": config["model"],
        "revision": config["revision"],
        "pooling": config["pooling"],
        "passage_max_tokens": config["passage_max_tokens"],
        "dtype": config["embedding_storage_dtype"],
    }
    if final_path.exists() and state_path.exists():
        state = load_json(state_path)
        if state.get("status") == "complete" and all(
            state.get(key) == value for key, value in expected.items()
        ):
            return np.load(final_path, mmap_mode="r"), state

    hidden_size = int(model.config.hidden_size)
    completed = 0
    if partial_path.exists() and state_path.exists():
        state = load_json(state_path)
        if state.get("status") == "building" and all(
            state.get(key) == value for key, value in expected.items()
        ):
            completed = int(state.get("completed_sections", 0))
            embeddings = np.lib.format.open_memmap(partial_path, mode="r+")
            if embeddings.shape != (len(section_ids), hidden_size):
                raise RuntimeError("partial dense cache shape mismatch")
        else:
            raise RuntimeError("incompatible partial dense cache; preserve and inspect it")
    else:
        embeddings = np.lib.format.open_memmap(
            partial_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(section_ids), hidden_size),
        )

    started = time.monotonic()
    last_report = started
    last_section_id = int(section_ids[completed - 1]) if completed else -1
    cursor = connection.execute(
        """
        SELECT section_id, code, document_name, title, label, text_search
        FROM searchable_sections
        WHERE section_id > ?
        ORDER BY section_id
        """,
        (last_section_id,),
    )
    batch_size = int(config["batch_size"])
    while completed < len(section_ids):
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break
        passages = [assemble_passage(dict(row)) for row in rows]
        encoded = tokenizer(
            passages,
            padding=True,
            truncation=True,
            max_length=int(config["passage_max_tokens"]),
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            vectors = dense_forward(model, encoded).cpu().numpy().astype(np.float32)
        row_ids = [int(row["section_id"]) for row in rows]
        expected_ids = section_ids[completed : completed + len(rows)].tolist()
        if row_ids != expected_ids:
            raise RuntimeError("searchable section order changed during encoding")
        embeddings[completed : completed + len(rows)] = vectors
        completed += len(rows)
        current = time.monotonic()
        if current - last_report >= 30 or completed == len(section_ids):
            embeddings.flush()
            elapsed = max(current - started, 0.001)
            state = {
                **expected,
                "status": "building",
                "embedding_dimension": hidden_size,
                "completed_sections": completed,
                "updated_at": now_iso(),
            }
            atomic_json(state_path, state)
            print(
                f"dense_encode {completed}/{len(section_ids)} "
                f"({completed / len(section_ids):.1%}) "
                f"session_rate={completed / elapsed:.1f}_sections/s",
                flush=True,
            )
            last_report = current
    if completed != len(section_ids):
        raise RuntimeError(f"dense encoding stopped at {completed}/{len(section_ids)}")
    embeddings.flush()
    del embeddings
    os.replace(partial_path, final_path)
    state = {
        **expected,
        "status": "complete",
        "embedding_dimension": hidden_size,
        "completed_sections": completed,
        "completed_at": now_iso(),
        "embedding_file": str(final_path.relative_to(ROOT)),
        "embedding_file_sha256": sha256_file(final_path),
    }
    atomic_json(state_path, state)
    return np.load(final_path, mmap_mode="r"), state


def encode_questions(
    questions: Sequence[Mapping[str, Any]], tokenizer: Any, model: Any, device: Any,
    config: Mapping[str, Any]
) -> Any:
    import numpy as np
    import torch

    output = []
    batch_size = int(config["batch_size"])
    for start in range(0, len(questions), batch_size):
        rows = questions[start : start + batch_size]
        texts = [normalize_unicode_and_space(str(row["question"])) for row in rows]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=int(config["query_max_tokens"]),
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            output.append(dense_forward(model, encoded).cpu().numpy().astype(np.float32))
    return np.concatenate(output, axis=0)


def dense_rankings(
    embeddings: Any,
    question_embeddings: Any,
    section_ids: Any,
    device: Any,
    top_k: int,
    query_batch_size: int,
) -> list[list[dict[str, Any]]]:
    import numpy as np
    import torch

    corpus = torch.from_numpy(np.asarray(embeddings)).to(device=device, dtype=torch.float32)
    rankings: list[list[dict[str, Any]]] = []
    for start in range(0, len(question_embeddings), query_batch_size):
        queries = torch.from_numpy(question_embeddings[start : start + query_batch_size]).to(
            device=device, dtype=torch.float32
        )
        with torch.inference_mode():
            score_batch = torch.matmul(queries, corpus.T).cpu().numpy()
        for scores in score_batch:
            rankings.append(stable_top_k(scores, section_ids, top_k))
        print(f"dense_score {len(rankings)}/{len(question_embeddings)} questions", flush=True)
    del corpus
    torch.cuda.empty_cache()
    return rankings


def fetch_sections(
    connection: sqlite3.Connection, section_ids: Sequence[int]
) -> dict[int, dict[str, Any]]:
    unique = sorted(set(map(int, section_ids)))
    if not unique:
        return {}
    result: dict[int, dict[str, Any]] = {}
    for start in range(0, len(unique), 800):
        block = unique[start : start + 800]
        placeholders = ",".join("?" for _ in block)
        rows = connection.execute(
            f"""
            SELECT ss.section_id, ss.document_id, ss.code, ss.document_name,
                   ss.title, ss.label, ss.text_search, d.version, d.update_date,
                   s.ordinal, s.level, s.has_table, s.has_image
            FROM searchable_sections ss
            JOIN sections s ON s.section_id = ss.section_id
            JOIN documents d ON d.document_id = ss.document_id
            WHERE ss.section_id IN ({placeholders})
            """,
            block,
        ).fetchall()
        for row in rows:
            item = dict(row)
            item["section_id"] = int(item["section_id"])
            item["document_id"] = int(item["document_id"])
            result[item["section_id"]] = item
    if len(result) != len(unique):
        missing = sorted(set(unique) - set(result))
        raise RuntimeError(f"ranked sections missing from corpus: {missing[:10]}")
    return result


def rerank_questions(
    connection: sqlite3.Connection,
    questions: Sequence[Mapping[str, Any]],
    hybrid: Sequence[Sequence[Mapping[str, Any]]],
    config: Mapping[str, Any],
    device: Any,
) -> list[list[dict[str, Any]]]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["revision"])
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model"], revision=config["revision"], dtype=torch.float16
    ).to(device)
    model.eval()
    actual_revision = model_revision(model)
    if actual_revision and actual_revision != config["revision"]:
        raise RuntimeError(f"reranker revision mismatch: {actual_revision}")
    outputs: list[list[dict[str, Any]]] = []
    for number, (question, hybrid_rows) in enumerate(zip(questions, hybrid), start=1):
        candidate_ids = [int(row["section_id"]) for row in hybrid_rows[: config["input_top_k"]]]
        metadata = fetch_sections(connection, candidate_ids)
        passages = [assemble_passage(metadata[section_id]) for section_id in candidate_ids]
        scores: list[float] = []
        for start in range(0, len(candidate_ids), int(config["batch_size"])):
            passage_batch = passages[start : start + int(config["batch_size"])]
            query_batch = [normalize_unicode_and_space(str(question["question"]))] * len(passage_batch)
            encoded = tokenizer(
                query_batch,
                passage_batch,
                padding=True,
                truncation="only_second",
                max_length=int(config["pair_max_tokens"]),
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                logits = model(**encoded, return_dict=True).logits.view(-1).float()
            scores.extend(map(float, logits.cpu().tolist()))
        outputs.append(rerank_by_score(candidate_ids, scores, int(config["output_top_k"])))
        print(f"rerank {number}/{len(questions)} questions", flush=True)
    del model, tokenizer
    torch.cuda.empty_cache()
    return outputs


def enrich_rankings(
    connection: sqlite3.Connection,
    ranking: Sequence[Mapping[str, Any]],
    question: Mapping[str, Any],
    components: Mapping[int, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    metadata = fetch_sections(connection, [int(row["section_id"]) for row in ranking])
    gold = set(map(int, question.get("gold_section_ids", [])))
    output = []
    for row in ranking:
        section_id = int(row["section_id"])
        item = {**row, **metadata[section_id]}
        item["is_exact_provisional_gold"] = section_id in gold
        item["passage_sha256"] = sha256_text(assemble_passage(metadata[section_id]))
        if components and section_id in components:
            item["components"] = dict(components[section_id])
        output.append(item)
    return output


def aggregate_metrics(question_rows: Sequence[Mapping[str, Any]], cutoffs: Sequence[int]) -> dict[str, Any]:
    by_arm: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in question_rows:
        if row["scoring_group"] == "answerable_retrieval":
            by_arm[str(row["arm"])].append(row)
    aggregate: dict[str, Any] = {}
    for arm in ARMS:
        rows = by_arm[arm]
        aggregate[arm] = {"answerable_questions": len(rows)}
        for cutoff in cutoffs:
            values = [row["metrics"][str(cutoff)] for row in rows]
            aggregate[arm][str(cutoff)] = {
                "any_gold_hit_count": sum(value["any_gold_hit"] for value in values),
                "required_evidence_recall_mean": sum(
                    value["required_evidence_recall"] for value in values
                ) / len(values),
                "complete_evidence_count": sum(value["complete_evidence"] for value in values),
                "mrr_mean": sum(value["mrr"] for value in values) / len(values),
            }
        aggregate[arm]["source_applicability_accuracy_at_1"] = sum(
            int(row["source_applicability_at_1"]) for row in rows
        ) / len(rows)
        aggregate[arm]["retrieval_latency_ms_total"] = round(
            sum(float(row["latency_ms"]) for row in rows), 3
        )
    return aggregate


def write_failure_csv(
    path: Path,
    questions: Sequence[Mapping[str, Any]],
    arm_hits: Mapping[str, Sequence[Sequence[Mapping[str, Any]]]],
) -> None:
    fields = [
        "question_id",
        "source_set",
        "analysis_stratum",
        "arm",
        "question",
        "gold_section_ids",
        "found_gold_section_ids_top10",
        "missing_gold_section_ids",
        "failure_shape",
        "top10_section_ids",
        "top10_codes",
        "top10_titles",
        "user_review",
        "review_note",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, question in enumerate(questions):
            if question["scoring_group"] != "answerable_retrieval":
                continue
            gold = set(map(int, question.get("gold_section_ids", [])))
            for arm in ARMS:
                top10 = arm_hits[arm][index][:10]
                top10_ids = [int(hit["section_id"]) for hit in top10]
                found = sorted(gold & set(top10_ids))
                missing = sorted(gold - set(top10_ids))
                if not missing:
                    continue
                writer.writerow({
                    "question_id": question["question_id"],
                    "source_set": question.get("source_set", ""),
                    "analysis_stratum": question.get("analysis_stratum", ""),
                    "arm": arm,
                    "question": question["question"],
                    "gold_section_ids": ";".join(map(str, sorted(gold))),
                    "found_gold_section_ids_top10": ";".join(map(str, found)),
                    "missing_gold_section_ids": ";".join(map(str, missing)),
                    "failure_shape": "partial_gold" if found else "no_gold",
                    "top10_section_ids": ";".join(map(str, top10_ids)),
                    "top10_codes": " | ".join(str(hit["code"]) for hit in top10),
                    "top10_titles": " | ".join(
                        f"{hit['title']} {hit.get('label') or ''}".strip() for hit in top10
                    ),
                    "user_review": "",
                    "review_note": "",
                })


def run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import torch

    config = load_config(args.config)
    validate_runner_config(config)
    preflight, database_path = validate_preflight(args.preflight, args.config, args.questions)
    questions = load_questions(args.questions)
    run_dir = args.runs_dir / args.run_id
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    shutil.copy2(args.questions, run_dir / "normalized_questions.jsonl")

    dense_config = dict(config["dense"])
    dense_config["batch_size"] = int(preflight["pre_run_gates"]["resource_dry_run"]["dense_batch_size"])
    reranker_config = dict(config["reranker"])
    reranker_config["batch_size"] = int(preflight["pre_run_gates"]["resource_dry_run"]["reranker_batch_size"])
    started_at = now_iso()
    command = " ".join(map(str, [sys.executable, Path(__file__).relative_to(ROOT), "--run-id", args.run_id]))
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": args.run_id,
        "status": "running",
        "material_passport": {**MATERIAL_PASSPORT, "origin_date": started_at[:10]},
        "started_at": started_at,
        "command": command,
        "working_directory": str(ROOT),
        "preflight_manifest_path": str(args.preflight.relative_to(ROOT)),
        "preflight_manifest_sha256": sha256_file(args.preflight),
        "git_head": git(("rev-parse", "HEAD")),
        "database_sha256": sha256_file(database_path),
        "questions_sha256": sha256_file(args.questions),
        "retrieval_config_sha256": sha256_file(args.config),
        "score_semantics": "exact provisional gold only; pooled equivalent evidence pending human review",
    }
    atomic_json(run_dir / "manifest.json", manifest)

    connection: sqlite3.Connection | None = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required by the frozen baseline configuration")
        torch.manual_seed(int(config["seed"]))
        torch.cuda.manual_seed_all(int(config["seed"]))
        torch.backends.cuda.matmul.allow_tf32 = False
        device = torch.device("cuda:0")
        connection = database_connection(database_path)
        section_ids = load_section_ids(connection)
        if len(section_ids) != 557027:
            raise RuntimeError(f"unexpected searchable section count: {len(section_ids)}")

        lexical_rankings: list[list[dict[str, Any]]] = []
        lexical_queries: list[str] = []
        lexical_latencies: list[float] = []
        for number, question in enumerate(questions, start=1):
            started = time.perf_counter()
            query, rows = lexical_search(connection, question["question"], config["lexical"]["top_k"])
            lexical_latencies.append((time.perf_counter() - started) * 1000)
            lexical_queries.append(query)
            lexical_rankings.append(rows)
            print(f"lexical {number}/{len(questions)} questions", flush=True)

        dense_tokenizer, dense_model = load_dense_model(dense_config, device)
        cache_key = dense_cache_key(preflight, sha256_file(args.config))
        embeddings, cache_state = build_or_load_embeddings(
            connection,
            section_ids,
            dense_tokenizer,
            dense_model,
            device,
            dense_config,
            args.runs_dir / "cache" / cache_key,
        )
        question_embeddings = encode_questions(
            questions, dense_tokenizer, dense_model, device, dense_config
        )
        del dense_model, dense_tokenizer
        torch.cuda.empty_cache()
        dense_started = time.perf_counter()
        dense_ranked = dense_rankings(
            embeddings,
            question_embeddings,
            section_ids,
            device,
            int(dense_config["top_k"]),
            int(dense_config["batch_size"]),
        )
        dense_total_ms = (time.perf_counter() - dense_started) * 1000
        dense_latencies = [dense_total_ms / len(questions)] * len(questions)

        hybrid_rankings: list[list[dict[str, Any]]] = []
        hybrid_latencies: list[float] = []
        hybrid_components: list[dict[int, dict[str, Any]]] = []
        for lexical_rows, dense_rows in zip(lexical_rankings, dense_ranked):
            started = time.perf_counter()
            fused = reciprocal_rank_fusion(
                [
                    [row["section_id"] for row in lexical_rows],
                    [row["section_id"] for row in dense_rows],
                ],
                k0=int(config["hybrid"]["rrf_k0"]),
                limit=int(config["hybrid"]["top_k"]),
            )
            hybrid_latencies.append((time.perf_counter() - started) * 1000)
            hybrid_rankings.append([
                {"rank": row["rank"], "section_id": row["section_id"], "score": row["rrf_score"]}
                for row in fused
            ])
            lexical_map = {row["section_id"]: row for row in lexical_rows}
            dense_map = {row["section_id"]: row for row in dense_rows}
            hybrid_components.append({
                int(row["section_id"]): {
                    "source_count": row["source_count"],
                    "lexical_rank": lexical_map.get(row["section_id"], {}).get("rank"),
                    "lexical_score": lexical_map.get(row["section_id"], {}).get("score"),
                    "dense_rank": dense_map.get(row["section_id"], {}).get("rank"),
                    "dense_score": dense_map.get(row["section_id"], {}).get("score"),
                }
                for row in fused
            })

        rerank_started = time.perf_counter()
        reranked = rerank_questions(connection, questions, hybrid_rankings, reranker_config, device)
        rerank_total_ms = (time.perf_counter() - rerank_started) * 1000
        rerank_latencies = [rerank_total_ms / len(questions)] * len(questions)

        raw_by_arm = {
            "R0-LEX": lexical_rankings,
            "R0-DENSE": dense_ranked,
            "R0-HYBRID": hybrid_rankings,
            "R0-HYBRID-RERANK": reranked,
        }
        latency_by_arm = {
            "R0-LEX": lexical_latencies,
            "R0-DENSE": dense_latencies,
            "R0-HYBRID": [a + b + c for a, b, c in zip(lexical_latencies, dense_latencies, hybrid_latencies)],
            "R0-HYBRID-RERANK": [
                a + b + c + d for a, b, c, d in zip(
                    lexical_latencies, dense_latencies, hybrid_latencies, rerank_latencies
                )
            ],
        }
        enriched_by_arm: dict[str, list[list[dict[str, Any]]]] = {arm: [] for arm in ARMS}
        question_results: list[dict[str, Any]] = []
        topk_path = run_dir / "retrieval_topk.jsonl"
        with topk_path.open("w", encoding="utf-8") as handle:
            for index, question in enumerate(questions):
                for arm in ARMS:
                    components = hybrid_components[index] if arm in {"R0-HYBRID", "R0-HYBRID-RERANK"} else None
                    enriched = enrich_rankings(
                        connection, raw_by_arm[arm][index], question, components
                    )
                    enriched_by_arm[arm].append(enriched)
                    ranked_ids = [hit["section_id"] for hit in enriched]
                    measured = question_metrics(question, ranked_ids, config["evaluation"]["cutoffs"])
                    result = {
                        "question_id": question["question_id"],
                        "arm": arm,
                        "scoring_group": measured["scoring_group"],
                        "metrics": measured["metrics"],
                        "source_applicability_at_1": (
                            source_applicable_at_1(question, enriched[0])
                            if enriched and measured["metrics"] is not None else None
                        ),
                        "latency_ms": round(latency_by_arm[arm][index], 3),
                    }
                    question_results.append(result)
                    record = {
                        "run_id": args.run_id,
                        "question_id": question["question_id"],
                        "question": question["question"],
                        "arm": arm,
                        "fts_query": lexical_queries[index] if arm in {"R0-LEX", "R0-HYBRID", "R0-HYBRID-RERANK"} else None,
                        "hits": enriched,
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        metrics = {
            "schema_version": 1,
            "run_id": args.run_id,
            "evaluation_label": "provisional_exact_gold_before_pooling",
            "answerable_question_count": 46,
            "control_question_count": 2,
            "cutoffs": config["evaluation"]["cutoffs"],
            "primary_metric": config["evaluation"]["primary_metric"],
            "aggregate": aggregate_metrics(question_results, config["evaluation"]["cutoffs"]),
            "question_results": question_results,
        }
        atomic_json(run_dir / "retrieval_metrics.json", metrics)
        write_failure_csv(run_dir / "failure_list.csv", questions, enriched_by_arm)

        report_lines = [
            f"# {args.run_id}", "", "## Material Passport", "",
            "- Origin Skill: academic-research-suite / experiment-agent",
            "- Origin Mode: run", f"- Origin Date: {started_at[:10]}",
            "- Verification Status: UNVERIFIED", "",
            "## Exact provisional-gold retrieval", "",
            "| Arm | Complete@10 | Recall@10 | Source@1 |", "|---|---:|---:|---:|",
        ]
        for arm in ARMS:
            item = metrics["aggregate"][arm]
            report_lines.append(
                f"| {arm} | {item['10']['complete_evidence_count']}/46 | "
                f"{item['10']['required_evidence_recall_mean']:.3f} | "
                f"{item['source_applicability_accuracy_at_1']:.3f} |"
            )
        report_lines.extend([
            "", "> These scores use exact provisional gold only. The user will review the plain failure list separately.", ""
        ])
        (run_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

        manifest.update(
            {
                "status": "complete",
                "finished_at": now_iso(),
                "material_passport": {**manifest["material_passport"], "verification_status": "ANALYZED"},
                "question_count": len(questions),
                "ranking_record_count": len(questions) * len(ARMS),
                "dense_cache": cache_state,
                "outputs": {
                    name: {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
                    for name, path in {
                        "questions": run_dir / "normalized_questions.jsonl",
                        "rankings": topk_path,
                        "metrics": run_dir / "retrieval_metrics.json",
                        "failure_list": run_dir / "failure_list.csv",
                        "report": run_dir / "report.md",
                    }.items()
                },
            }
        )
        atomic_json(run_dir / "manifest.json", manifest)
        print(json.dumps(metrics["aggregate"], ensure_ascii=False, indent=2), flush=True)
        return metrics
    except BaseException as error:
        manifest.update({"status": "failed", "finished_at": now_iso(), "error": repr(error)})
        atomic_json(run_dir / "manifest.json", manifest)
        raise
    finally:
        if connection is not None:
            connection.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
