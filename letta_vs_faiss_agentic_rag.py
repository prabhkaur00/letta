#!/usr/bin/env python3
"""
End-to-end single-agent RAG benchmark using Letta archival memory.

Stages per query:
  1) search archival memory (server.agent_manager.query_agent_passages_async)
  2) prompt augmentation with retrieved context
  3) LLM call (optional)
  4) insert response into archival memory (server.passage_manager.create_many_archival_passages_async)
  5) per-stage + end-to-end latency logging

Hydrates the agent's archival memory from a Faiss index, mirroring letta_vs_faiss.py.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import sys
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

import faiss  # type: ignore
import numpy as np

from letta.config import LettaConfig
from letta.llm_api.llm_client import LLMClient
from letta.schemas.agent import AgentState, CreateAgent
from letta.schemas.block import CreateBlock
from letta.schemas.enums import MessageRole
from letta.schemas.letta_message_content import TextContent
from letta.schemas.message import Message
from letta.schemas.passage import Passage as PydanticPassage
from letta.schemas.user import User
from letta.server.server import SyncServer
from letta.settings import DatabaseChoice, settings as letta_settings

try:
    from tqdm.auto import tqdm
except ImportError as exc:  # pragma: no cover - dependency requirement
    raise RuntimeError("tqdm is required for letta_vs_faiss_agentic_rag.py. Install it with `pip install tqdm`.") from exc

if TYPE_CHECKING:
    from AgentMemory.interface import MemoryManagement
    from AgentMemory.types import MemoryItem

MEMORY_MODES = {
    "one_search_one_insert": "Search each query and insert each query.",
    "step_search_then_insert": "Search each query; insert once at end.",
    "search_then_step_insert": "Search first query only; insert each query.",
    "search_only": "Search each query; never insert.",
}

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
        "letta_vs_faiss_agentic_rag.py detected that Letta is configured to use SQLite (settings.database_engine=sqlite). "
        "This benchmark requires a Postgres backend with pgvector; delete or update "
        f"{config_path} and/or export LETTA_DATABASE_ENGINE=postgres / LETTA_PG_URI before rerunning."
    )


@dataclass
class Timings:
    search_ms: float
    insert_ms: float
    total_ms: float


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


def format_context(hits: Sequence[Tuple[PydanticPassage, float, dict]]) -> str:
    lines: List[str] = []
    for idx, (passage, score, _meta) in enumerate(hits):
        text = _sanitize_text(getattr(passage, "text", "") or "")
        if not text:
            continue
        line = f"[{idx + 1}] {text} (score={score:.4f})"
        lines.append(line)
    return "\n".join(lines) if lines else "(no memory)"


def build_llm_messages(query: str, context: str) -> List[Message]:
    system_text = (
        "You are a helpful assistant. Use the provided context if it helps answer the question. "
        "If the context is insufficient, say you do not know."
    )
    user_text = f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer concisely."
    return [
        Message(role=MessageRole.system, content=[TextContent(text=system_text)]),
        Message(role=MessageRole.user, content=[TextContent(text=user_text)]),
    ]


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
        name=f"letta-agentic-rag-{dataset}",
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
                metadata={"faiss_id": doc_id, "seed": True},
            )
            passages.append(passage)
        await server.passage_manager.create_many_archival_passages_async(passages, actor=actor)
    elapsed = time.perf_counter() - start
    print(f"[hydrate] Seeded archival memory with {total} passages in {elapsed:.2f}s")


async def call_llm(
    *,
    llm_client,
    agent_state: AgentState,
    query: str,
    context: str,
) -> Optional[str]:
    if llm_client is None:
        return None
    messages = build_llm_messages(query, context)
    request_data = llm_client.build_request_data(
        agent_type=agent_state.agent_type,
        messages=messages,
        llm_config=agent_state.llm_config,
        tools=[],
        force_tool_call=None,
    )
    response = await llm_client.send_llm_request_async(
        request_data=request_data,
        messages=messages,
        llm_config=agent_state.llm_config,
    )
    content = None
    try:
        content = response.choices[0].message.content
    except Exception:
        content = None
    return content


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agentic RAG benchmark using Letta archival memory.")
    parser.add_argument("--dataset", type=str, help="Dataset name (requires access to M3_memory2/test_perf).")
    parser.add_argument("--dataset-file", type=str, help="Path to newline-delimited text file used when --dataset is not provided.")
    parser.add_argument("--m3-repo", type=str, default=None, help="Optional path to the M3 repository root.")
    parser.add_argument("--split", default=None, help="Dataset split override passed to the loader.")
    parser.add_argument("--limit", type=int, default=0, help="Limit dataset items when loading (<=0 means all).")
    parser.add_argument("--query-limit", type=int, default=10000, help="Maximum number of queries to evaluate (<=0 means all loaded items).")
    parser.add_argument("--memory-mode", choices=sorted(MEMORY_MODES.keys()), default="one_search_one_insert")
    parser.add_argument("--search-batch", type=int, default=256, help="Queries per search batch.")
    parser.add_argument("--insert-batch", type=int, default=256, help="Payloads per insert batch.")
    parser.add_argument("--hydrate-batch", type=int, default=2048, help="Batch size for initial Faiss -> Letta hydration.")
    parser.add_argument("--top-k", type=int, default=3, help="Search top-k.")
    parser.add_argument("--faiss-index", type=str, required=True, help="Path to the Faiss .index file (used for hydration).")
    parser.add_argument("--log-jsonl", type=str, default=None, help="Optional JSONL log file path.")
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
    print("[encoder] Encoding insert payloads once for insert mirroring...")
    insert_vectors = encode_insert_vectors_local(hf_mm, insert_items)

    if args.search_batch != 256 or args.insert_batch != 256:
        raise ValueError("For benchmarking consistency, use --search-batch 256 and --insert-batch 256.")
    search_cap = min(256, len(query_texts))
    insert_cap = min(256, len(query_texts))
    search_batches = [SearchBatch(start=0, end=search_cap)]
    insert_batches = [
        InsertBatch(start=0, end=insert_cap, doc_ids=list(range(expected_size, expected_size + insert_cap)))
    ]

    server = await create_server()
    actor = await server.user_manager.get_actor_or_default_async()
    agent_state = await create_benchmark_agent(
        server,
        actor,
        args.dataset,
        agent_model=args.agent_model,
        embedding_model=args.embedding_model,
    )
    llm_client = LLMClient.create(
        provider_type=agent_state.llm_config.model_endpoint_type,
        actor=actor,
    )
    if llm_client is None:
        raise RuntimeError("LLM client not available; cannot run with LLM enabled.")

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

        log_fp = None
        if args.log_jsonl:
            log_path = Path(args.log_jsonl)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_fp = log_path.open("a", encoding="utf-8")
            log_fp.write(json.dumps({"event": "start", "ts": datetime.now().isoformat(), "args": vars(args)}) + "\n")
            log_fp.flush()

        timing_records: List[Timings] = []
        pending_inserts: List[PydanticPassage] = []
        cached_context: Optional[str] = None

        iterator: Iterable[Tuple[int, str]] = enumerate(query_texts)
        iterator = tqdm(iterator, total=len(query_texts), desc="queries", unit="q")

        for idx, query in iterator:
            t0 = time.perf_counter()
            do_search = args.memory_mode in (
                "one_search_one_insert",
                "step_search_then_insert",
                "search_only",
            ) or (args.memory_mode == "search_then_step_insert" and idx == 0)

            hits: List[Tuple[PydanticPassage, float, dict]] = []
            t_search0 = time.perf_counter()
            if do_search:
                batch_vec = np.ascontiguousarray(query_vecs[idx : idx + 1], dtype=np.float32)
                query_vector = batch_vec[0].tolist()
                hits = await server.agent_manager.query_agent_passages_async(
                    actor=actor,
                    agent_id=agent_state.id,
                    query_text=query,
                    limit=args.top_k,
                    embed_query=True,
                    embedding_config=agent_state.embedding_config,
                    query_embedding=query_vector,
                )
                cached_context = format_context(hits)
            t_search1 = time.perf_counter()

            context = cached_context if cached_context is not None else "(no memory)"
            response_text: Optional[str] = None
            if len(hits) >= 1:
                try:
                    response_text = await call_llm(
                        llm_client=llm_client,
                        agent_state=agent_state,
                        query=query,
                        context=context,
                    )
                except Exception:
                    response_text = None

            t_insert0 = time.perf_counter()
            if args.memory_mode in ("one_search_one_insert", "search_then_step_insert"):
                if response_text is None:
                    response_text = ""
                passage = PydanticPassage(
                    text=_sanitize_text(response_text),
                    embedding=insert_vectors[idx].tolist(),
                    embedding_config=agent_state.embedding_config,
                    organization_id=actor.organization_id,
                    archive_id=archive.id,
                    metadata={"query": query, "type": "rag_response"},
                )
                await server.passage_manager.create_many_archival_passages_async([passage], actor=actor)
            elif args.memory_mode == "step_search_then_insert":
                if response_text is None:
                    response_text = ""
                passage = PydanticPassage(
                    text=_sanitize_text(response_text),
                    embedding=insert_vectors[idx].tolist(),
                    embedding_config=agent_state.embedding_config,
                    organization_id=actor.organization_id,
                    archive_id=archive.id,
                    metadata={"query": query, "type": "rag_response"},
                )
                pending_inserts.append(passage)
            t_insert1 = time.perf_counter()
            t1 = time.perf_counter()

            timings = Timings(
                search_ms=(t_search1 - t_search0) * 1000.0,
                insert_ms=(t_insert1 - t_insert0) * 1000.0,
                total_ms=(t1 - t0) * 1000.0,
            )
            timing_records.append(timings)

            if log_fp:
                record = {
                    "query": query,
                    "response": response_text,
                    "retrieved_ids": [getattr(p, "id", None) for p, _s, _m in hits],
                    "timings_ms": asdict(timings),
                    "latency_ms": timings.total_ms,
                }
                log_fp.write(json.dumps(record) + "\n")
                log_fp.flush()

        if args.memory_mode == "step_search_then_insert" and pending_inserts:
            await server.passage_manager.create_many_archival_passages_async(pending_inserts, actor=actor)

        if timing_records and log_fp:
            def avg(field: str) -> float:
                return statistics.fmean(getattr(t, field) for t in timing_records)

            summary = {
                "count": len(timing_records),
                "avg_search_ms": avg("search_ms"),
                "avg_insert_ms": avg("insert_ms"),
                "avg_total_ms": avg("total_ms"),
                "faiss_index": str(faiss_path),
                "faiss_index_fingerprint": file_fingerprint(faiss_path),
                "faiss_serialized_bytes": len(faiss_serialized),
            }
            log_fp.write(json.dumps({"timing_summary": summary}) + "\n")
            log_fp.flush()
    finally:
        await server.agent_manager.delete_agent_async(agent_state.id, actor=actor)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
