#!/usr/bin/env python
"""
Simulate a single-agent workflow that:
  * hydrates archival memory from a Faiss index (like letta_vs_faiss.py),
  * replays search/insert schedules (modes), and
  * optionally invokes the agent LLM with searched context when relevant.

This is intentionally similar to letta_vs_faiss.py but focuses on a realistic
agent flow (search -> optional LLM -> optional insert).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import datetime
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import faiss  # type: ignore
import numpy as np

from letta.schemas.agent import AgentState
from letta.schemas.passage import Passage as PydanticPassage
from letta.schemas.user import User
from letta.server.server import SyncServer

from letta_llm_agent_helper import (
    HFEncoder,
    InsertBatch,
    SearchBatch,
    SimpleHTTPChatModel,
    _sanitize_text,
    tqdm,
    _assert_postgres_backend,
    _load_dataset_helpers,
    build_context,
    build_llm_messages,
    build_llm,
    chunk_inserts,
    chunk_searches,
    create_benchmark_agent,
    create_server,
    encode_query_vectors_local,
    file_fingerprint,
    hydrate_agent_from_faiss,
    init_hf_encoder,
    make_query_items,
    maybe_invoke_llm,
)

SUPPORTED_MODES = [
    "one_search_one_insert",
    "step_search_then_insert",
    "search_then_step_insert",
    "search_only",
]

MEMORY_MODES = {
    "one_search_one_insert": "Search each query and insert each query.",
    "step_search_then_insert": "Search each query; insert once at end.",
    "search_then_step_insert": "Search first query only; insert each query.",
    "search_only": "Search each query; never insert.",
}

DEFAULT_DATASETS = ["gsm8k", "agentgym", "prm800k", "ultrachat", "ultrafeedback", "xlam_function_calling"]


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
    insert_doc_ids: Sequence[int],
    encoder: HFEncoder,
    top_k: int,
    ops_per_run: int,
    llm_client,
    llm_enabled: bool,
    llm_every: int,
    llm_max_calls: int,
    csv_writer,
    e2e_fp,
    e2e_limit: int,
) -> Dict[str, float]:
    stats = {
        "processed": 0,
        "total_queries": len(query_texts),
        "letta_time": 0.0,
        "letta_queries": 0,
        "inserted": 0,
        "insert_time": 0.0,
        "search_batches": 0,
        "insert_batches": 0,
        "llm_calls": 0,
        "llm_skipped": 0,
        "llm_errors": 0,
    }
    outputs: List[Optional[str]] = [None] * len(query_texts)
    record_rows = [
        {"search_ms": None, "insert_ms": None, "insert_tokens": None, "touched": False}
        for _ in range(len(query_texts))
    ]
    e2e_records: Dict[int, Dict[str, object]] = {}
    e2e_logged: set[int] = set()

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

    def count_message_tokens(messages) -> int:
        if not messages:
            return 0
        if hasattr(llm_client, "_message_payload"):
            payload = llm_client._message_payload(messages)
            return sum(encoder.count_tokens(entry.get("content", "") or "") for entry in payload)
        total = 0
        for msg in messages:
            for part in msg.content or []:
                text = getattr(part, "text", "") or ""
                total += encoder.count_tokens(text)
        return total

    def maybe_write_e2e(query_index: int) -> None:
        if not e2e_fp or e2e_limit <= 0:
            return
        if query_index in e2e_logged or len(e2e_logged) >= e2e_limit:
            return
        record = e2e_records.get(query_index)
        if not record:
            return
        record["query_index"] = query_index
        e2e_fp.write(json.dumps(record) + "\n")
        e2e_logged.add(query_index)

    async def invoke_llm_for_query(query_index: int, text: str, passages) -> None:
        llm_text = None
        if llm_enabled and (llm_max_calls <= 0 or stats["llm_calls"] < llm_max_calls):
            try:
                context = build_context(passages) if passages else ""
                messages = build_llm_messages(text, context) if context else []
                e2e_records.setdefault(query_index, {})
                e2e_records[query_index].update(
                    {
                        "query_chars": len(text or ""),
                        "context_chars": len(context or ""),
                        "llm_input_tokens": count_message_tokens(messages),
                        "llm_skipped": False,
                    }
                )
                llm_text = await maybe_invoke_llm(
                    llm_client=llm_client,
                    agent_state=agent_state,
                    query_text=text,
                    passages=passages,
                )
                if llm_text is not None:
                    stats["llm_calls"] += 1
                else:
                    stats["llm_skipped"] += 1
                    e2e_records[query_index]["llm_skipped"] = True
            except Exception:
                stats["llm_errors"] += 1
        else:
            stats["llm_skipped"] += 1
            e2e_records.setdefault(query_index, {})
            e2e_records[query_index]["llm_skipped"] = True
        if llm_text:
            outputs[query_index] = llm_text
            e2e_records.setdefault(query_index, {})
            e2e_records[query_index].update(
                {
                    "llm_output_chars": len(llm_text),
                    "llm_output_tokens": encoder.count_tokens(llm_text),
                }
            )
        if not llm_text:
            maybe_write_e2e(query_index)

    async def perform_search(batch: SearchBatch, *, do_llm: bool) -> Optional[Sequence]:
        nonlocal ops_in_run
        if batch.end <= batch.start:
            return None
        batch_vecs = np.ascontiguousarray(query_vecs[batch.start : batch.end], dtype=np.float32)
        last_tuples = None

        for local_idx, text in enumerate(query_texts[batch.start : batch.end]):
            query_index = batch.start + local_idx
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
            last_tuples = tuples
            record_rows[query_index]["search_ms"] = letta_elapsed * 1000.0
            record_rows[query_index]["touched"] = True

            if tuples:
                faiss_ids = []
                text_lens = []
                for passage, _score, _meta in tuples:
                    meta = getattr(passage, "metadata", {}) or {}
                    faiss_ids.append(meta.get("faiss_id"))
                    text_lens.append(len(getattr(passage, "text", "") or ""))
                e2e_records.setdefault(query_index, {})
                e2e_records[query_index].update(
                    {
                        "retrieved_count": len(tuples),
                        "retrieved_faiss_ids": faiss_ids,
                        "retrieved_text_chars_total": sum(text_lens),
                        "retrieved_text_chars_per_item": text_lens,
                    }
                )

            if do_llm and (stats["processed"] % max(1, llm_every) == 0):
                await invoke_llm_for_query(query_index, text, tuples)

            if not do_llm:
                maybe_write_e2e(query_index)

        stats["search_batches"] += 1
        if search_pbar:
            search_pbar.update(1)
            avg_letta_ms = (stats["letta_time"] / max(1, stats["letta_queries"])) * 1000.0
            search_pbar.set_postfix(
                {"letta_ms": f"{avg_letta_ms:.2f}", "at": f"{batch.end}/{len(query_texts)}"}, refresh=False
            )
        ops_in_run += 1
        maybe_log()
        return last_tuples

    async def perform_insert(batch: InsertBatch) -> None:
        nonlocal ops_in_run
        if batch.end <= batch.start:
            return
        indices = list(range(batch.start, batch.end))
        chunk_texts = []
        chunk_doc_ids = []
        chunk_query_indices = []
        chunk_token_counts = []
        for offset, idx in enumerate(indices):
            text = outputs[idx]
            if not text:
                continue
            chunk_texts.append(_sanitize_text(text))
            chunk_doc_ids.append(batch.doc_ids[offset])
            chunk_query_indices.append(idx)
            chunk_token_counts.append(encoder.count_tokens(text))
        if not chunk_texts:
            return
        chunk_vecs = encoder.encode_texts(chunk_texts)

        passages: List[PydanticPassage] = []
        for offset, doc_id in enumerate(chunk_doc_ids):
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
        insert_elapsed = time.perf_counter() - insert_start
        stats["insert_time"] += insert_elapsed
        stats["inserted"] += len(passages)
        stats["insert_batches"] += 1
        batch_insert_ms = insert_elapsed * 1000.0
        if insert_pbar:
            insert_pbar.update(1)
            insert_pbar.set_postfix(
                {"batch_insert_ms": f"{batch_insert_ms:.2f}", "inserted": stats["inserted"], "at": f"{batch.end}/{len(query_texts)}"},
                refresh=False,
            )

        if passages:
            per_item_ms = (insert_elapsed * 1000.0) / max(1, len(passages))
            for offset in range(len(passages)):
                query_index = chunk_query_indices[offset]
                record_rows[query_index]["insert_ms"] = per_item_ms
                record_rows[query_index]["insert_tokens"] = chunk_token_counts[offset]
                record_rows[query_index]["touched"] = True
                e2e_records.setdefault(query_index, {})
                e2e_records[query_index]["insert_tokens"] = chunk_token_counts[offset]
                maybe_write_e2e(query_index)

        ops_in_run += 1
        maybe_log()

    try:
        # perform_search gets passages, runs LLM, stores outputs[idx]
        # perform_insert embeds/inserts that output immediately.
        # Batching: none (1 item per loop).
        if mode == "one_search_one_insert":
            async def run_query(search_index: int) -> None:
                await perform_search(SearchBatch(start=search_index, end=search_index + 1), do_llm=True)
                await perform_insert(
                    InsertBatch(start=search_index, end=search_index + 1, doc_ids=[insert_doc_ids[search_index]])
                )

            await asyncio.gather(*(run_query(idx) for idx in range(len(query_texts))))
        # Batching: search and insert are batched via search_batches and insert_batches
        elif mode == "step_search_then_insert":
            for batch in search_batches:
                await perform_search(batch, do_llm=True)
            for batch in insert_batches:
                await perform_insert(batch)
        # first perform_search only for idx 0, cache passages
        # LLM uses cached passages for all items to fill outputs[...], then perform_insert inserts each batch.
        # Batching: only inserts are batched. Search is single-item.
        elif mode == "search_then_step_insert":
            cached_passages = None
            if query_texts:
                cached_passages = await perform_search(SearchBatch(start=0, end=1), do_llm=False)
            if cached_passages is not None:
                await asyncio.gather(
                    *(invoke_llm_for_query(idx, text, cached_passages) for idx, text in enumerate(query_texts))
                )
            for batch in insert_batches:
                await perform_insert(batch)
        # no llm calls, searches batched
        elif mode == "search_only":
            for batch in search_batches:
                await perform_search(batch, do_llm=False)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
    finally:
        if search_pbar:
            search_pbar.close()
        if insert_pbar:
            insert_pbar.close()

    avg_letta_ms = (stats["letta_time"] / max(1, stats["letta_queries"])) * 1000.0
    if csv_writer:
        for query_index, record in enumerate(record_rows):
            if not record["touched"]:
                continue
            search_ms = record["search_ms"] or 0.0
            insert_ms = record["insert_ms"] or 0.0
            insert_tokens = record["insert_tokens"] or 0
            total_ms = search_ms + insert_ms
            csv_writer.writerow(
                {
                    "query_index": query_index,
                    "search_ms": f"{search_ms:.4f}",
                    "insert_ms": f"{insert_ms:.4f}",
                    "insert_tokens": str(int(insert_tokens)),
                    "total_ms": f"{total_ms:.4f}",
                }
            )
    return {
        "mode": mode,
        "processed": stats["processed"],
        "total_queries": stats["total_queries"],
        "avg_letta_ms": avg_letta_ms,
        "inserted": stats["inserted"],
        "insert_time": stats["insert_time"],
        "search_batches": stats["search_batches"],
        "insert_batches": stats["insert_batches"],
        "llm_calls": stats["llm_calls"],
        "llm_skipped": stats["llm_skipped"],
        "llm_errors": stats["llm_errors"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate a single-agent workflow over a Faiss-backed memory index.")
    parser.add_argument("--dataset", type=str, help="Dataset name (requires access to M3_memory2/test_perf).")
    parser.add_argument("--dataset-file", type=str, help="Path to newline-delimited text file used when --dataset is not provided.")
    parser.add_argument("--m3-repo", type=str, default=None, help="Optional path to the M3 repository root.")
    parser.add_argument("--split", default=None, help="Dataset split override passed to the loader.")
    parser.add_argument("--limit", type=int, default=0, help="Limit dataset items when loading (<=0 means all).")
    parser.add_argument("--faiss-limit", type=int, default=0, help="Limit number of Faiss vectors hydrated (<=0 means all).")
    parser.add_argument("--query-limit", type=int, default=10000, help="Maximum number of queries to evaluate (<=0 means all loaded items).")
    parser.add_argument(
        "--mode",
        default="all",
        help="Comma-separated list of request modes (all = run every supported mode). "
        f"Supported: {', '.join(SUPPORTED_MODES)}.",
    )
    parser.add_argument("--search-batch", type=int, default=256, help="Queries per search batch.")
    parser.add_argument("--insert-batch", type=int, default=256, help="Payloads per insert batch.")
    parser.add_argument("--hydrate-batch", type=int, default=8192, help="Batch size for initial Faiss -> Letta hydration.")
    parser.add_argument("--ops-per-run", type=int, default=0, help="Emit a progress log every N operations (0 disables chunked logging).")
    parser.add_argument("--top-k", type=int, default=5, help="Search top-k.")
    parser.add_argument("--faiss-index", type=str, required=True, help="Path to the Faiss .index file (used for hydration).")
    parser.add_argument("--log-file", type=str, default=None, help="Destination file for JSON summaries.")
    parser.add_argument("--agent-model", type=str, default="letta/letta-free", help="LLM handle used when creating benchmark agents.")
    parser.add_argument("--embedding-model", type=str, default="letta/letta-free", help="Embedding handle used when creating benchmark agents.")
    parser.add_argument("--hf-model", type=str, default="intfloat/e5-large-v2", help="Hugging Face encoder to embed benchmarks.")
    parser.add_argument("--hf-batch-size", type=int, default=64, help="Batch size for the Hugging Face encoder.")
    parser.add_argument("--llm-url", type=str, default=None, help="HTTP URL for vLLM (or compatible) chat endpoint.")
    parser.add_argument("--llm-timeout", type=float, default=120.0, help="LLM HTTP timeout in seconds.")
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
    hf_root = Path("/data/hf")
    (hf_root / "hub").mkdir(parents=True, exist_ok=True)
    (hf_root / "datasets").mkdir(parents=True, exist_ok=True)
    (hf_root / "transformers").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_root))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_root / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_root / "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_root / "transformers"))
    print(f"[env] HF cache root set to {hf_root}")
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

    ## hydrate from faiss
    requested_limit = args.limit if args.limit and args.limit > 0 else None
    expected_size = faiss_index.ntotal
    if args.faiss_limit and args.faiss_limit > 0:
        expected_size = min(expected_size, args.faiss_limit)
        if expected_size < faiss_index.ntotal:
            print(f"[faiss] Using first {expected_size} vectors (faiss_limit={args.faiss_limit}, ntotal={faiss_index.ntotal}).")
    if requested_limit is not None:
        expected_size = min(expected_size, requested_limit)
        if expected_size < faiss_index.ntotal:
            print(f"[faiss] Using first {expected_size} vectors (limit={requested_limit}, ntotal={faiss_index.ntotal}).")
    if expected_size <= 0:
        raise RuntimeError("Faiss index contains no vectors.")

    hydrate_texts = [f"faiss-doc-{idx}" for idx in range(expected_size)]
    faiss_serialized = faiss.serialize_index(faiss_index)

    search_limit = args.query_limit if args.query_limit and args.query_limit > 0 else expected_size
    search_limit = min(search_limit, expected_size)
    query_texts = texts[:search_limit]
    print(f"[dataset] Using {len(query_texts)} texts for queries/inserts (top_k={args.top_k})")

    print(f"[encoder] Initializing Hugging Face encoder '{args.hf_model}' (batch_size={args.hf_batch_size})")
    hf_encoder = init_hf_encoder(args.hf_model, args.hf_batch_size)
    search_items = make_query_items(query_texts)
    print("[encoder] Encoding queries once for reuse across modes...")
    query_vecs = encode_query_vectors_local(hf_encoder, search_items)

    if args.search_batch <= 0 or args.insert_batch <= 0:
        raise ValueError("search_batch and insert_batch must be > 0.")
    search_batches = [
        SearchBatch(start=start, end=min(start + args.search_batch, len(query_texts)))
        for start in range(0, len(query_texts), args.search_batch)
    ]
    insert_doc_ids = list(range(expected_size, expected_size + len(query_texts)))
    insert_batches = [
        InsertBatch(
            start=start,
            end=min(start + args.insert_batch, len(query_texts)),
            doc_ids=insert_doc_ids[start : min(start + args.insert_batch, len(query_texts))],
        )
        for start in range(0, len(query_texts), args.insert_batch)
    ]

    modes = _resolve_modes(args.mode)

    server = await create_server()
    actor = await server.user_manager.get_actor_or_default_async()

    summaries: List[Dict[str, object]] = []

    timestamp = datetime.datetime.now().strftime("%m%d%H%M")
    safe_dataset = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(args.dataset))
    csv_path = Path(
        f"benchmarks/letta_llm_agent_workflow/{safe_dataset}_{args.mode}_{timestamp}.csv"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_fp = csv_path.open("a", encoding="utf-8", newline="")
    csv_writer = csv.DictWriter(
        csv_fp,
        fieldnames=["query_index", "search_ms", "insert_ms", "insert_tokens", "total_ms"],
    )
    if csv_fp.tell() == 0:
        csv_fp.write(f"# top_k={args.top_k}, faiss_init_vectors={expected_size}\n")
        csv_writer.writeheader()
    e2e_path = csv_path.with_name(f"{csv_path.stem}_e2e.jsonl")
    e2e_fp = e2e_path.open("a", encoding="utf-8")

    async def run_modes_with_agent(modes_to_run: List[str]) -> None:
        agent_state = await create_benchmark_agent(
            server,
            actor,
            args.dataset,
            agent_model=args.agent_model,
            embedding_model=args.embedding_model,
        )
        llm_client = build_llm(args)
        llm_enabled = True
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

            archive = await server.archive_manager.get_or_create_default_archive_for_agent_async(
                agent_state=agent_state, actor=actor
            )

            for mode in modes_to_run:
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
                    insert_doc_ids=insert_doc_ids,
                    encoder=hf_encoder,
                    top_k=args.top_k,
                    ops_per_run=args.ops_per_run,
                    llm_client=llm_client,
                    llm_enabled=llm_enabled,
                    llm_every=1,
                    llm_max_calls=0,
                    csv_writer=csv_writer,
                    e2e_fp=e2e_fp,
                    e2e_limit=3,
                )

                print(
                    f"[metrics] dataset={args.dataset} mode={mode} "
                    f"letta={metrics['avg_letta_ms']:.3f} ms/query "
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
                    "query_limit": args.query_limit,
                    "average_letta_ms": metrics["avg_letta_ms"],
                    "inserted": metrics["inserted"],
                    "insert_time": metrics["insert_time"],
                    "search_batches": metrics["search_batches"],
                    "insert_batches": metrics["insert_batches"],
                    "llm_calls": metrics["llm_calls"],
                    "llm_skipped": metrics["llm_skipped"],
                    "llm_errors": metrics["llm_errors"],
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
            await server.agent_manager.delete_agent_async(agent_state.id, actor=actor)

    captured_exception: Optional[BaseException] = None
    try:
        if "search_only" in modes and len(modes) > 1:
            modes_without_search = [m for m in modes if m != "search_only"]
            first_mode = modes_without_search[0]
            await run_modes_with_agent(["search_only", first_mode])
            for mode in modes_without_search[1:]:
                await run_modes_with_agent([mode])
        else:
            for mode in modes:
                await run_modes_with_agent([mode])
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        captured_exception = exc
    finally:
        csv_fp.flush()
        e2e_fp.flush()
        csv_fp.close()
        e2e_fp.close()
        if not args.log_file:
            print(json.dumps(summaries, indent=2))

    if captured_exception is not None:
        raise captured_exception


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
