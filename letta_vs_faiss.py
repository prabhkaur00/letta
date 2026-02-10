#!/usr/bin/env python
"""
Benchmark Letta's archival memory search against a Faiss ground-truth index.

This script mirrors the orchestration of M3_memory2/test_perf/diskann_vs_faiss_suite.py:
  * hydrate a fresh archival memory index from a Faiss snapshot,
  * replay ratio_throughput-style request schedules (multiple modes),
  * mirror every insert back into Faiss so recall reflects the evolving corpus,
  * measure average Letta latency, Faiss latency, and recall, and
  * emit JSON summaries per mode for downstream analysis.

The dataset loaders are shared with the M3 repo. By default the script searches for
../M3_memory2/test_perf/ratio_throughput.py (or a custom --m3-repo path). When that
module is unavailable you can provide --dataset-file pointing at a newline-delimited
text file to replay instead.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

import faiss  # type: ignore
import numpy as np

from letta.config import LettaConfig
from letta.schemas.agent import AgentState, CreateAgent
from letta.schemas.block import CreateBlock
from letta.schemas.passage import Passage as PydanticPassage
from letta.schemas.user import User
from letta.server.server import SyncServer
from letta.settings import DatabaseChoice, settings as letta_settings

try:
    from tqdm.auto import tqdm
except ImportError as exc:  # pragma: no cover - dependency requirement
    raise RuntimeError("tqdm is required for letta_vs_faiss.py. Install it with `pip install tqdm`.") from exc

if TYPE_CHECKING:
    from AgentMemory.interface import MemoryManagement
    from AgentMemory.types import MemoryItem

SUPPORTED_MODES = [
    "item_search_insert",
    "step_search_then_update",
    "head_search_tail_insert",
    "search_only",
]

DEFAULT_DATASETS = ["gsm8k", "agentgym", "prm800k", "ultrachat", "ultrafeedback", "xlam_function_calling"]


def _sanitize_text(text: str) -> str:
    """Remove NULs and other non-printable ASCII control characters except common whitespace."""
    return "".join(ch for ch in text if (ord(ch) >= 32 or ch in "\n\r\t"))


def _assert_postgres_backend() -> None:
    """
    Ensure the server is configured to use Postgres/pgvector before hydrating embeddings.
    """
    if letta_settings.database_engine is DatabaseChoice.POSTGRES:
        return
    config_path = Path.home() / ".letta" / "config"
    raise RuntimeError(
        "letta_vs_faiss.py detected that Letta is configured to use SQLite (settings.database_engine=sqlite). "
        "This benchmark requires a Postgres backend with pgvector; delete or update "
        f"{config_path} and/or export LETTA_DATABASE_ENGINE=postgres / LETTA_PG_URI before rerunning."
    )


@dataclass
class SearchBatch:
    start: int
    end: int


@dataclass
class InsertBatch:
    start: int
    end: int
    doc_ids: List[int]


def _candidate_m3_roots(extra: Optional[str]) -> List[Path]:
    script_dir = Path(__file__).resolve().parent
    roots = []
    if extra:
        roots.append(Path(extra).expanduser().resolve())
    roots.extend(
        [
            script_dir / "M3_memory2",
            script_dir / "M3_memory",
            script_dir.parent / "M3_memory2",
            script_dir.parent / "M3_memory",
            Path("/workspace/M3_memory2"),
            Path("/workspace/M3_memory"),
        ]
    )
    return roots


def _load_dataset_helpers(extra_repo: Optional[str]):
    """
    Locate test_perf/ratio_throughput.py in a sibling M3 repository and import the dataset utilities.
    Returns (DATASET_LOADERS, flatten_dataset) when successful.
    """
    for root in _candidate_m3_roots(extra_repo):
        ratio_file = root / "test_perf" / "ratio_throughput.py"
        if ratio_file.is_file():
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            try:
                from test_perf.ratio_throughput import DATASET_LOADERS, flatten_dataset  # type: ignore
            except ImportError as exc:  # pragma: no cover - informative failure
                raise RuntimeError(f"Found {ratio_file} but failed to import dataset helpers: {exc}") from exc
            return DATASET_LOADERS, flatten_dataset
    raise RuntimeError(
        "Unable to locate test_perf/ratio_throughput.py. Provide --m3-repo pointing to the root of M3_memory2 "
        "or use --dataset-file to supply newline-delimited texts."
    )


def chunk_searches(total: int, batch_size: int) -> List[SearchBatch]:
    if batch_size <= 0:
        raise ValueError("search_batch must be > 0")
    batches: List[SearchBatch] = []
    for start in range(0, total, batch_size):
        end = min(total, start + batch_size)
        batches.append(SearchBatch(start=start, end=end))
    return batches


def chunk_inserts(total: int, batch_size: int, base_offset: int) -> List[InsertBatch]:
    if batch_size <= 0:
        raise ValueError("insert_batch must be > 0")
    batches: List[InsertBatch] = []
    for start in range(0, total, batch_size):
        end = min(total, start + batch_size)
        doc_ids = list(range(base_offset + start, base_offset + end))
        batches.append(InsertBatch(start=start, end=end, doc_ids=doc_ids))
    return batches


def compute_recall(predicted: Sequence[int], ground_truth: Sequence[int], top_k: int) -> float:
    if not ground_truth:
        return 0.0
    denom = min(top_k, len(ground_truth))
    gt = set(doc for doc in ground_truth if doc >= 0)
    if not gt:
        return 0.0
    overlap = sum(1 for doc in predicted if doc in gt)
    return overlap / max(1, denom)


def clone_faiss_index(serialized: bytes) -> faiss.Index:
    buffer = np.frombuffer(serialized, dtype=np.uint8)
    return faiss.deserialize_index(buffer)


def file_fingerprint(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def init_hf_encoder(model_name: str, batch_size: int) -> "MemoryManagement":
    from AgentMemory.interface import MemoryManagement

    return MemoryManagement(backend="placeholder", model_name=model_name, batch_size=batch_size)


def encode_query_vectors_local(mm: "MemoryManagement", items: List["MemoryItem"]) -> np.ndarray:
    if mm.encoder is None:
        raise RuntimeError("MemoryManagement encoder is not initialized.")
    vecs = mm.encoder.encode_queries(items)
    return np.ascontiguousarray(vecs, dtype=np.float32)


def encode_insert_vectors_local(mm: "MemoryManagement", items: List["MemoryItem"]) -> np.ndarray:
    if mm.encoder is None:
        raise RuntimeError("MemoryManagement encoder is not initialized.")
    mat = mm.encoder.encode_items(items)
    return np.ascontiguousarray(mat, dtype=np.float32)


def make_query_items(texts: Sequence[str]) -> List["MemoryItem"]:
    from AgentMemory.types import MemoryItem

    return [MemoryItem(id=f"q-{idx}", data=text) for idx, text in enumerate(texts)]


def make_insert_items(texts: Sequence[str], base_offset: int) -> List["MemoryItem"]:
    from AgentMemory.types import MemoryItem

    items: List[MemoryItem] = []
    for idx, text in enumerate(texts):
        items.append(MemoryItem(id=str(base_offset + idx), data=text))
    return items


async def create_server() -> SyncServer:
    config = LettaConfig.load()
    config.save()
    server = SyncServer()
    await server.init_async()
    return server


async def create_benchmark_agent(
    server: SyncServer,
    actor: User,
    dataset: str,
    *,
    agent_model: str,
    embedding_model: str,
) -> AgentState:
    agent_create = CreateAgent(
        name=f"letta-vs-faiss-{dataset}",
        memory_blocks=[
            CreateBlock(label="persona", value=""),
            CreateBlock(label="human", value=""),
        ],
        include_base_tools=False,
        include_multi_agent_tools=False,
        include_default_source=False,
        model=agent_model,
        embedding=embedding_model,
    )
    return await server.create_agent_async(agent_create, actor=actor)


async def hydrate_agent_from_faiss(
    *,
    server: SyncServer,
    actor: User,
    agent_state: AgentState,
    faiss_index: faiss.Index,
    texts: Sequence[str],
    hydrate_batch: int,
    vector_count: int,
) -> None:
    archive = await server.archive_manager.get_or_create_default_archive_for_agent_async(agent_state=agent_state, actor=actor)
    total = vector_count
    if total > len(texts):
        raise RuntimeError(
            f"Dataset only contains {len(texts)} texts, but Faiss index expects {total}. Increase --limit or dataset size."
        )
    print(f"[hydrate] Inserting {total} baseline passages into Letta (batch={hydrate_batch})...")
    start = time.perf_counter()
    for batch_start in range(0, total, hydrate_batch):
        batch_end = min(total, batch_start + hydrate_batch)
        passages: List[PydanticPassage] = []
        for doc_id in range(batch_start, batch_end):
            vector = faiss_index.reconstruct(doc_id)
            passage = PydanticPassage(
                text=_sanitize_text(texts[doc_id]),
                embedding=vector.tolist(),
                embedding_config=agent_state.embedding_config,
                organization_id=actor.organization_id,
                archive_id=archive.id,
                metadata={"faiss_id": doc_id},
            )
            passages.append(passage)
        await server.passage_manager.create_many_archival_passages_async(passages, actor=actor)
    elapsed = time.perf_counter() - start
    print(f"[hydrate] Seeded archival memory with {total} passages in {elapsed:.2f}s")


async def run_mode(
    *,
    mode: str,
    server: SyncServer,
    actor: User,
    agent_state: AgentState,
    archive_id: str,
    search_batches: List[SearchBatch],
    insert_batches: List[InsertBatch],
    query_texts: Sequence[str],
    query_vecs: np.ndarray,
    insert_vectors: np.ndarray,
    faiss_serialized: bytes,
    top_k: int,
    ops_per_run: int,
) -> Dict[str, float]:
    stats = {
        "processed": 0,
        "total_queries": len(query_texts),
        "letta_time": 0.0,
        "letta_queries": 0,
        "faiss_time": 0.0,
        "faiss_queries": 0,
        "recall_total": 0.0,
        "recall_count": 0,
        "inserted": 0,
        "insert_time": 0.0,
        "search_batches": 0,
        "insert_batches": 0,
    }

    faiss_rt = clone_faiss_index(faiss_serialized)
    ops_in_run = 0
    run_idx = 1
    search_pbar = (
        tqdm(total=len(search_batches), desc=f"{mode}:search", leave=False, unit="batch") if search_batches else None
    )
    insert_pbar = (
        tqdm(total=len(insert_batches), desc=f"{mode}:insert", leave=False, unit="batch")
        if insert_batches and mode != "search_only"
        else None
    )

    def maybe_log() -> None:
        nonlocal ops_in_run, run_idx
        if ops_per_run <= 0:
            return
        if ops_in_run >= ops_per_run:
            print(
                f"[run {run_idx:03d}] mode={mode} search_batches={stats['search_batches']} "
                f"insert_batches={stats['insert_batches']} processed={stats['processed']}"
            )
            run_idx += 1
            ops_in_run = 0

    async def perform_search(batch: SearchBatch) -> None:
        nonlocal ops_in_run
        if batch.end <= batch.start:
            return
        batch_vecs = np.ascontiguousarray(query_vecs[batch.start : batch.end], dtype=np.float32)
        faiss_start = time.perf_counter()
        _, faiss_ids = faiss_rt.search(batch_vecs, top_k)
        faiss_elapsed = time.perf_counter() - faiss_start
        stats["faiss_time"] += faiss_elapsed
        stats["faiss_queries"] += (batch.end - batch.start)

        for local_idx, text in enumerate(query_texts[batch.start : batch.end]):
            letta_start = time.perf_counter()
            query_vector = batch_vecs[local_idx].tolist()
            tuples = await server.agent_manager.query_agent_passages_async(
                actor=actor,
                agent_id=agent_state.id,
                query_text=text,
                limit=top_k,
                embed_query=True,
                embedding_config=agent_state.embedding_config,
                query_embedding=query_vector,
            )
            letta_elapsed = time.perf_counter() - letta_start
            stats["letta_time"] += letta_elapsed
            stats["letta_queries"] += 1
            stats["processed"] += 1

            hits: List[int] = []
            for passage, _, _ in tuples[:top_k]:
                faiss_id = None
                metadata = getattr(passage, "metadata", None)
                if isinstance(metadata, dict):
                    faiss_id = metadata.get("faiss_id")
                if faiss_id is None:
                    try:
                        faiss_id = int(passage.id)
                    except (ValueError, AttributeError, TypeError):
                        faiss_id = None
                if faiss_id is None:
                    continue
                hits.append(int(faiss_id))
            gt = [int(doc) for doc in faiss_ids[local_idx] if int(doc) >= 0]
            stats["recall_total"] += compute_recall(hits, gt, top_k)
            stats["recall_count"] += 1

        stats["search_batches"] += 1
        if search_pbar:
            search_pbar.update(1)
            avg_letta_ms = (stats["letta_time"] / max(1, stats["letta_queries"])) * 1000.0
            search_pbar.set_postfix({"letta_ms": f"{avg_letta_ms:.2f}"}, refresh=False)
        ops_in_run += 1
        maybe_log()

    async def perform_insert(batch: InsertBatch) -> None:
        nonlocal ops_in_run
        if batch.end <= batch.start:
            return
        chunk_vecs = np.ascontiguousarray(insert_vectors[batch.start : batch.end], dtype=np.float32)
        chunk_texts = [_sanitize_text(text) for text in query_texts[batch.start : batch.end]]

        passages: List[PydanticPassage] = []
        for offset, doc_id in enumerate(batch.doc_ids):
            passage = PydanticPassage(
                text=chunk_texts[offset],
                embedding=chunk_vecs[offset].tolist(),
                embedding_config=agent_state.embedding_config,
                organization_id=actor.organization_id,
                archive_id=archive_id,
                metadata={"faiss_id": doc_id},
            )
            passages.append(passage)

        insert_start = time.perf_counter()
        await server.passage_manager.create_many_archival_passages_async(passages, actor=actor)
        stats["insert_time"] += time.perf_counter() - insert_start
        stats["inserted"] += len(passages)
        stats["insert_batches"] += 1
        batch_insert_ms = (time.perf_counter() - insert_start) * 1000.0
        if insert_pbar:
            insert_pbar.update(1)
            insert_pbar.set_postfix({"batch_insert_ms": f"{batch_insert_ms:.2f}", "inserted": stats["inserted"]}, refresh=False)

        # Index uses a direct map with sequential ids, so plain .add() appends vectors
        faiss_rt.add(chunk_vecs)

        ops_in_run += 1
        maybe_log()

    try:
        if mode == "item_search_insert":
            for idx in range(max(len(search_batches), len(insert_batches))):
                if idx < len(search_batches):
                    await perform_search(search_batches[idx])
                if idx < len(insert_batches):
                    await perform_insert(insert_batches[idx])
        elif mode == "step_search_then_update":
            for batch in search_batches:
                await perform_search(batch)
            for batch in insert_batches:
                await perform_insert(batch)
        elif mode == "head_search_tail_insert":
            if search_batches:
                await perform_search(search_batches[0])
            for batch in insert_batches:
                await perform_insert(batch)
        elif mode == "search_only":
            for batch in search_batches:
                await perform_search(batch)
        else:  # pragma: no cover - validation protects this path
            raise ValueError(f"Unsupported mode: {mode}")
    finally:
        if search_pbar:
            search_pbar.close()
        if insert_pbar:
            insert_pbar.close()

    avg_letta_ms = (stats["letta_time"] / max(1, stats["letta_queries"])) * 1000.0
    avg_faiss_ms = (stats["faiss_time"] / max(1, stats["faiss_queries"])) * 1000.0
    avg_recall = stats["recall_total"] / max(1, stats["recall_count"])

    return {
        "mode": mode,
        "processed": stats["processed"],
        "total_queries": stats["total_queries"],
        "avg_letta_ms": avg_letta_ms,
        "avg_faiss_ms": avg_faiss_ms,
        "avg_recall": avg_recall,
        "inserted": stats["inserted"],
        "insert_time": stats["insert_time"],
        "search_batches": stats["search_batches"],
        "insert_batches": stats["insert_batches"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Letta archival memory against Faiss ground truth.")
    parser.add_argument("--dataset", type=str, help="Dataset name (requires access to M3_memory2/test_perf).")
    parser.add_argument("--dataset-file", type=str, help="Path to newline-delimited text file used when --dataset is not provided.")
    parser.add_argument("--m3-repo", type=str, default=None, help="Optional path to the M3 repository root.")
    parser.add_argument("--split", default=None, help="Dataset split override passed to the loader.")
    parser.add_argument("--limit", type=int, default=0, help="Limit dataset items when loading (<=0 means all).")
    parser.add_argument("--query-limit", type=int, default=10000, help="Maximum number of queries to evaluate (<=0 means all loaded items).")
    parser.add_argument("--mode", default="all", help="Comma-separated list of request modes (all = run every supported mode).")
    parser.add_argument("--search-batch", type=int, default=256, help="Queries per search batch.")
    parser.add_argument("--insert-batch", type=int, default=256, help="Payloads per insert batch.")
    parser.add_argument("--hydrate-batch", type=int, default=2048, help="Batch size for initial Faiss -> Letta hydration.")
    parser.add_argument("--ops-per-run", type=int, default=0, help="Emit a progress log every N operations (0 disables chunked logging).")
    parser.add_argument("--top-k", type=int, default=5, help="Search top-k.")
    parser.add_argument("--faiss-index", type=str, required=True, help="Path to the Faiss .index file (used for hydration + ground truth).")
    parser.add_argument("--log-file", type=str, default=None, help="Destination file for JSON summaries.")
    parser.add_argument("--agent-model", type=str, default="letta/letta-free", help="LLM handle used when creating benchmark agents.")
    parser.add_argument("--embedding-model", type=str, default="letta/letta-free", help="Embedding handle used when creating benchmark agents.")
    parser.add_argument("--hf-model", type=str, default="intfloat/e5-large-v2", help="Hugging Face encoder to embed benchmarks.")
    parser.add_argument("--hf-batch-size", type=int, default=64, help="Batch size for the Hugging Face encoder.")
    return parser.parse_args()


def _load_texts_from_file(path: Path, limit: Optional[int]) -> List[str]:
    content = path.read_text(encoding="utf-8")
    texts = [line.strip() for line in content.splitlines() if line.strip()]
    if limit and limit > 0:
        return texts[:limit]
    return texts


def _resolve_modes(raw: str) -> List[str]:
    if not raw or raw.strip().lower() == "all":
        return list(SUPPORTED_MODES)
    candidates = []
    seen = set()
    for part in raw.split(","):
        key = part.strip()
        if not key:
            continue
        if key not in SUPPORTED_MODES:
            raise ValueError(f"Unsupported mode: {key} (supported: {', '.join(SUPPORTED_MODES)})")
        if key not in seen:
            candidates.append(key)
            seen.add(key)
    if not candidates:
        raise ValueError("No valid modes were provided.")
    return candidates


async def main_async() -> None:
    args = parse_args()
    _assert_postgres_backend()
    if not args.dataset and not args.dataset_file:
        raise ValueError("Provide either --dataset or --dataset-file.")

    faiss_path = Path(args.faiss_index).expanduser()
    if not faiss_path.is_file():
        raise FileNotFoundError(f"Faiss index not found: {faiss_path}")

    if args.dataset:
        loaders, flatten_dataset = _load_dataset_helpers(args.m3_repo)
        if args.dataset not in loaders:
            available = ", ".join(sorted(loaders.keys()) or DEFAULT_DATASETS)
            raise ValueError(f"Dataset '{args.dataset}' not available. Known datasets: {available}")
        texts = flatten_dataset(
            args.dataset,
            args.split,
            args.limit if args.limit and args.limit > 0 else None,
        )
    else:
        dataset_file = Path(args.dataset_file).expanduser()
        if not dataset_file.is_file():
            raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
        texts = _load_texts_from_file(dataset_file, args.limit if args.limit and args.limit > 0 else None)
        args.dataset = dataset_file.stem

    if not texts:
        raise RuntimeError("No dataset texts were loaded.")
    print(f"[dataset] Loaded {len(texts)} items for dataset='{args.dataset}'")

    faiss_load_start = time.perf_counter()
    faiss_index = faiss.read_index(str(faiss_path))
    print(f"[timer] Faiss index loaded in {time.perf_counter() - faiss_load_start:.2f}s (ntotal={faiss_index.ntotal})")

    requested_limit = args.limit if args.limit and args.limit > 0 else None
    expected_size = faiss_index.ntotal
    if requested_limit is not None:
        expected_size = min(expected_size, requested_limit)
        if expected_size < faiss_index.ntotal:
            print(f"[faiss] Using first {expected_size} vectors (limit={requested_limit}, ntotal={faiss_index.ntotal}).")
    if expected_size <= 0:
        raise RuntimeError("Faiss index contains no vectors.")

    hydrate_texts = texts
    if len(hydrate_texts) < expected_size:
        needed = expected_size - len(hydrate_texts)
        print(f"[dataset] Only {len(hydrate_texts)} texts available; synthesizing {needed} placeholders.")
        for idx in range(needed):
            hydrate_texts.append(f"faiss-doc-{len(hydrate_texts) + idx}")
    hydrate_texts = hydrate_texts[:expected_size]
    faiss_serialized = faiss.serialize_index(faiss_index)

    search_limit = args.query_limit if args.query_limit and args.query_limit > 0 else expected_size
    search_limit = min(search_limit, expected_size)
    query_texts = hydrate_texts[:search_limit]
    print(f"[dataset] Using {len(query_texts)} texts for queries/inserts (top_k={args.top_k})")

    print(f"[encoder] Initializing Hugging Face encoder '{args.hf_model}' (batch_size={args.hf_batch_size})")
    hf_mm = init_hf_encoder(args.hf_model, args.hf_batch_size)
    search_items = make_query_items(query_texts)
    insert_items = make_insert_items(query_texts, base_offset=expected_size)
    print("[encoder] Encoding queries once for reuse across modes...")
    query_vecs = encode_query_vectors_local(hf_mm, search_items)
    print("[encoder] Encoding insert payloads once for Faiss mirroring...")
    insert_vectors = encode_insert_vectors_local(hf_mm, insert_items)

    if args.search_batch != 256 or args.insert_batch != 256:
        raise ValueError("For benchmarking consistency, use --search-batch 256 and --insert-batch 256.")
    search_cap = min(256, len(query_texts))
    insert_cap = min(256, len(query_texts))
    search_batches = [SearchBatch(start=0, end=search_cap)]
    insert_batches = [
        InsertBatch(start=0, end=insert_cap, doc_ids=list(range(expected_size, expected_size + insert_cap)))
    ]

    modes = _resolve_modes(args.mode)

    server = await create_server()
    actor = await server.user_manager.get_actor_or_default_async()

    summaries: List[Dict[str, object]] = []

    for mode in modes:
        agent_state = await create_benchmark_agent(
            server,
            actor,
            args.dataset,
            agent_model=args.agent_model,
            embedding_model=args.embedding_model,
        )
        try:
            await hydrate_agent_from_faiss(
                server=server,
                actor=actor,
                agent_state=agent_state,
                faiss_index=faiss_index,
                texts=hydrate_texts,
                hydrate_batch=args.hydrate_batch,
                vector_count=expected_size,
            )

            archive = await server.archive_manager.get_or_create_default_archive_for_agent_async(agent_state=agent_state, actor=actor)

            metrics = await run_mode(
                mode=mode,
                server=server,
                actor=actor,
                agent_state=agent_state,
                archive_id=archive.id,
                search_batches=search_batches,
                insert_batches=insert_batches,
                query_texts=query_texts,
                query_vecs=query_vecs,
                insert_vectors=insert_vectors,
                faiss_serialized=faiss_serialized,
                top_k=args.top_k,
                ops_per_run=args.ops_per_run,
            )

            print(
                f"[metrics] dataset={args.dataset} mode={mode} recall@{args.top_k}={metrics['avg_recall']:.4f} "
                f"letta={metrics['avg_letta_ms']:.3f} ms/query faiss={metrics['avg_faiss_ms']:.3f} ms/query "
                f"inserted={int(metrics['inserted'])}"
            )

            summary = {
                "dataset": args.dataset,
                "mode": mode,
                "total_queries": metrics["total_queries"],
                "processed_queries": metrics["processed"],
                "top_k": args.top_k,
                "search_batch": args.search_batch,
                "insert_batch": args.insert_batch,
                "hydrate_batch": args.hydrate_batch,
                "agent_model": args.agent_model,
                "embedding_model": args.embedding_model,
                "faiss_index": str(faiss_path),
                "faiss_index_fingerprint": file_fingerprint(faiss_path),
                "faiss_serialized_bytes": len(faiss_serialized),
                "query_limit": args.query_limit,
                "average_letta_ms": metrics["avg_letta_ms"],
                "average_faiss_ms": metrics["avg_faiss_ms"],
                "average_recall": metrics["avg_recall"],
                "inserted": metrics["inserted"],
                "insert_time": metrics["insert_time"],
                "search_batches": metrics["search_batches"],
                "insert_batches": metrics["insert_batches"],
            }
            summaries.append(summary)

            if args.log_file:
                log_path = Path(args.log_file)
                if len(modes) > 1:
                    stem = log_path.stem
                    mode_path = log_path.with_name(f"{stem}_{mode}{log_path.suffix or '.json'}")
                else:
                    mode_path = log_path
                mode_path.parent.mkdir(parents=True, exist_ok=True)
                mode_path.write_text(json.dumps(summary, indent=2))
                print(f"[log] Summary for mode={mode} written to {mode_path}")
        finally:
            await server.agent_manager.delete_agent_async(agent_id=agent_state.id, actor=actor)

    if summaries:
        best = max(summaries, key=lambda row: (row["average_recall"], -row["average_letta_ms"]))
        print(
            f"[best] dataset={best['dataset']} mode={best['mode']} "
            f"(recall={best['average_recall']:.4f}, letta_ms={best['average_letta_ms']:.2f})"
        )


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
