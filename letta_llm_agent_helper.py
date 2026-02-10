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
from typing import Dict, List, Optional, Sequence, Tuple

import faiss  # type: ignore
import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from letta.config import LettaConfig
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
    raise RuntimeError("tqdm is required for letta_llm_agent_workflow.py. Install it with `pip install tqdm`.") from exc


@dataclass
class MemoryItem:
    id: str
    data: str


@dataclass
class SearchBatch:
    start: int
    end: int


@dataclass
class InsertBatch:
    start: int
    end: int
    doc_ids: List[int]


def _coerce_to_str(data: object) -> str:
    if isinstance(data, np.ndarray):
        return json.dumps(data.tolist(), ensure_ascii=False)
    if isinstance(data, (list, tuple)):
        return json.dumps(list(data), ensure_ascii=False)
    if isinstance(data, dict):
        return json.dumps(data, sort_keys=True, ensure_ascii=False)
    return str(data)


def _load_hf_model(
    model_path: str,
    *,
    device: Optional[str] = None,
    use_fp16: bool = False,
    trust_remote_code: bool = True,
):
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[encoder] loading model: {model_path} (device={target_device})")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model.eval()
    try:
        model.to(target_device)
    except RuntimeError as exc:
        message = str(exc).lower()
        if "out of memory" in message and target_device.startswith("cuda"):
            print("[encoder] CUDA OOM while loading model; falling back to CPU.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            target_device = "cpu"
            model.to(target_device)
        else:
            raise
    if use_fp16 and target_device.startswith("cuda"):
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )
    return model, tokenizer, target_device, config


class HFEncoder:
    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int = 32,
        max_length: int = 512,
        precision: str = "fp32",
        device: Optional[str] = None,
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.normalize = normalize
        prec = precision.lower()
        use_fp16 = prec == "fp16"
        model, tok, device_str, config = _load_hf_model(
            model_name,
            device=device,
            use_fp16=use_fp16,
            trust_remote_code=True,
        )
        self.model = model
        self.tokenizer = tok
        self.device = torch.device(device_str)
        self.config = config
        self.native_dim = int(getattr(self.config, "hidden_size", 1024))

        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        all_vecs: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            toks = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            toks = {k: v.to(self.device) for k, v in toks.items()}
            out = self.model(**toks)
            last = out.last_hidden_state
            mask = toks["attention_mask"].unsqueeze(-1).type_as(last)
            summed = (last * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-6)
            emb = summed / counts
            emb = emb.detach().cpu().to(torch.float32).numpy()
            all_vecs.append(emb)
        X = np.vstack(all_vecs) if all_vecs else np.zeros((0, self.native_dim), dtype=np.float32)
        if self.normalize and X.size > 0:
            n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
            X = X / n
        return X.astype("float32")

    def encode_items(self, items: List[MemoryItem]) -> np.ndarray:
        texts = [_coerce_to_str(it.data) for it in items]
        return self.encode_texts(texts)

    def encode_queries(self, items: List[MemoryItem]) -> np.ndarray:
        texts = [_coerce_to_str(it.data) for it in items]
        return self.encode_texts(texts)

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        return len(ids)


class SimpleHTTPChatModel:
    """
    Simple HTTP chat model.
    Expects:
      POST LLM_HTTP_URL
      Body: {"messages": [{"role": "...", "content": "..."}]}
      Response: {"text": "..."} or OpenAI-style {"choices":[{"message":{"content":"..."}}]}
    """

    def __init__(self, url: str, model: str, timeout_s: float = 120.0) -> None:
        self.url = url
        self.model = model
        self.timeout_s = timeout_s

    def _message_payload(self, messages: List[Message]) -> List[Dict[str, str]]:
        payload: List[Dict[str, str]] = []
        for message in messages:
            role = "user"
            if message.role is MessageRole.system:
                role = "system"
            elif message.role is MessageRole.assistant:
                role = "assistant"
            text_parts: List[str] = []
            for part in message.content or []:
                if isinstance(part, TextContent):
                    text_parts.append(part.text or "")
            payload.append({"role": role, "content": "".join(text_parts)})
        return payload

    def chat(self, messages: List[Message]) -> str:
        import urllib.request

        msg_payload = self._message_payload(messages)
        use_openai = bool(os.getenv("LLM_OPENAI", "").strip()) or self.url.rstrip("/").endswith("/v1/chat/completions")
        if use_openai:
            payload = {"model": self.model, "messages": msg_payload}
        else:
            payload = {"messages": msg_payload}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            body = resp.read().decode("utf-8")
        obj = json.loads(body)
        if "choices" in obj:
            choice0 = obj.get("choices", [{}])[0] or {}
            msg = choice0.get("message", {}) or {}
            return msg.get("content", "") or ""
        return obj.get("text", "") or ""


def build_llm(args: argparse.Namespace) -> SimpleHTTPChatModel:
    url = (args.llm_url or os.getenv("LLM_HTTP_URL", "")).strip()
    if not url:
        raise SystemExit("Set --llm-url or LLM_HTTP_URL.")
    model = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    return SimpleHTTPChatModel(url, model=model, timeout_s=args.llm_timeout)


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
        "letta_llm_agent_workflow.py detected that Letta is configured to use SQLite (settings.database_engine=sqlite). "
        "This benchmark requires a Postgres backend with pgvector; delete or update "
        f"{config_path} and/or export LETTA_DATABASE_ENGINE=postgres / LETTA_PG_URI before rerunning."
    )


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


def file_fingerprint(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def init_hf_encoder(model_name: str, batch_size: int) -> HFEncoder:
    return HFEncoder(model_name=model_name, batch_size=batch_size)


def encode_query_vectors_local(encoder: HFEncoder, items: List[MemoryItem]) -> np.ndarray:
    vecs = encoder.encode_queries(items)
    return np.ascontiguousarray(vecs, dtype=np.float32)


def encode_insert_vectors_local(encoder: HFEncoder, items: List[MemoryItem]) -> np.ndarray:
    mat = encoder.encode_items(items)
    return np.ascontiguousarray(mat, dtype=np.float32)


def make_query_items(texts: Sequence[str]) -> List[MemoryItem]:
    return [MemoryItem(id=f"q-{idx}", data=text) for idx, text in enumerate(texts)]


def make_insert_items(texts: Sequence[str], base_offset: int) -> List[MemoryItem]:
    items: List[MemoryItem] = []
    for idx, text in enumerate(texts):
        items.append(MemoryItem(id=str(base_offset + idx), data=text))
    return items


def build_context(passages: Sequence[Tuple[PydanticPassage, float, dict]]) -> str:
    lines: List[str] = []
    for idx, (passage, _score, _meta) in enumerate(passages):
        text = _sanitize_text(getattr(passage, "text", "") or "")
        if not text:
            continue
        snippet = f"[{idx + 1}] {text}"
        lines.append(snippet)
    return "\n".join(lines)


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
        name=f"letta-faiss-workflow-{dataset}",
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
    pbar = tqdm(total=total, desc="hydrate", unit="passage", leave=False)
    try:
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
            pbar.update(batch_end - batch_start)
            pbar.set_postfix({"at": f"{batch_end}/{total}"}, refresh=False)
    finally:
        pbar.close()
    elapsed = time.perf_counter() - start
    print(f"[hydrate] Seeded archival memory with {total} passages in {elapsed:.2f}s")


async def maybe_invoke_llm(
    *,
    llm_client,
    agent_state: AgentState,
    query_text: str,
    passages: Sequence[Tuple[PydanticPassage, float, dict]],
    ) -> Optional[str]:
    if llm_client is None:
        return None
    if not passages:
        return None
    context = build_context(passages)
    if not context:
        return None
    messages = build_llm_messages(query_text, context)
    response_text = await asyncio.to_thread(llm_client.chat, messages)
    return response_text
