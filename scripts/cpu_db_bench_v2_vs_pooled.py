#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


def _apply_mode_env(mode: str) -> None:
    if mode == "v2":
        os.environ.setdefault("LETTA_DISABLE_SQLALCHEMY_POOLING", "true")
    elif mode == "v2_pooled":
        os.environ.setdefault("LETTA_DISABLE_SQLALCHEMY_POOLING", "false")
        os.environ.setdefault("LETTA_PG_POOL_SIZE", "20")
        os.environ.setdefault("LETTA_PG_MAX_OVERFLOW", "10")
        os.environ.setdefault("LETTA_PG_POOL_TIMEOUT", "30")
        os.environ.setdefault("LETTA_PG_POOL_RECYCLE", "1800")
        os.environ.setdefault("LETTA_POOL_PRE_PING", "true")
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def _percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    if pct <= 0:
        return min(values)
    if pct >= 100:
        return max(values)
    ordered = sorted(values)
    k = (len(ordered) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(ordered) - 1)
    if f == c:
        return ordered[f]
    d0 = ordered[f] * (c - k)
    d1 = ordered[c] * (k - f)
    return d0 + d1


def _summarize_latencies(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "count": 0,
            "avg_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "min_ms": None,
            "max_ms": None,
        }
    return {
        "count": len(values),
        "avg_ms": statistics.mean(values),
        "p50_ms": _percentile(values, 50.0),
        "p95_ms": _percentile(values, 95.0),
        "p99_ms": _percentile(values, 99.0),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU-only DB benchmark for v2 vs v2_pooled.")
    parser.add_argument("--mode", choices=["v2", "v2_pooled", "both"], default="both")
    parser.add_argument("--base-count", type=int, default=1000, help="Baseline corpus size.")
    parser.add_argument("--ops", type=int, default=1000, help="Number of search+insert ops.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K for vector search.")
    parser.add_argument("--parallel", type=int, default=10, help="Max concurrent ops.")
    parser.add_argument("--hydrate-batch", type=int, default=200, help="Batch size for baseline inserts.")
    parser.add_argument("--dim", type=int, default=0, help="Embedding dimension (0 = MAX_EMBEDDING_DIM).")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    parser.add_argument("--agent-model", type=str, default="letta/letta-free")
    parser.add_argument("--embedding-model", type=str, default="letta/letta-free")
    parser.add_argument("--csv", type=str, default=None, help="Optional CSV output path.")
    parser.add_argument("--log-file", type=str, default=None, help="Optional JSON summary output path.")
    return parser.parse_args()


async def _run_once(args: argparse.Namespace, mode: str) -> Dict[str, object]:
    import numpy as np

    from letta.constants import MAX_EMBEDDING_DIM
    from letta.schemas.passage import Passage as PydanticPassage
    from letta_llm_agent_helper import create_benchmark_agent, create_server

    dim = MAX_EMBEDDING_DIM if args.dim <= 0 else args.dim

    rng = np.random.default_rng(args.seed)

    base_vecs = rng.standard_normal((args.base_count, dim)).astype(np.float32)
    base_texts = [f"baseline-{i}" for i in range(args.base_count)]

    query_vecs = rng.standard_normal((args.ops, dim)).astype(np.float32)
    query_texts = [f"query-{i}" for i in range(args.ops)]

    insert_vecs = rng.standard_normal((args.ops, dim)).astype(np.float32)
    insert_texts = [f"insert-{i}" for i in range(args.ops)]

    server = await create_server()
    actor = await server.user_manager.get_actor_or_default_async()

    agent_state = None
    agent_state = await create_benchmark_agent(
        server,
        actor,
        dataset=f"cpu-bench-{mode}",
        agent_model=args.agent_model,
        embedding_model=args.embedding_model,
    )

    try:
        archive = await server.archive_manager.get_or_create_default_archive_for_agent_async(
            agent_state=agent_state, actor=actor
        )

        print(f"[hydrate:{mode}] inserting baseline passages={args.base_count} batch={args.hydrate_batch}")
        hydrate_start = time.perf_counter()
        for start in range(0, args.base_count, args.hydrate_batch):
            end = min(start + args.hydrate_batch, args.base_count)
            passages = [
                PydanticPassage(
                    text=base_texts[i],
                    embedding=base_vecs[i].tolist(),
                    embedding_config=agent_state.embedding_config,
                    organization_id=actor.organization_id,
                    archive_id=archive.id,
                    metadata={"seed": args.seed, "type": "baseline"},
                )
                for i in range(start, end)
            ]
            await server.passage_manager.create_many_archival_passages_async(passages, actor=actor)
        hydrate_elapsed = time.perf_counter() - hydrate_start
        print(f"[hydrate:{mode}] done in {hydrate_elapsed:.2f}s")

        search_latencies_ms: List[Optional[float]] = [None] * args.ops
        insert_latencies_ms: List[Optional[float]] = [None] * args.ops

        semaphore = asyncio.Semaphore(max(1, args.parallel))

        async def run_op(index: int) -> None:
            async with semaphore:
                search_start = time.perf_counter()
                await server.agent_manager.query_agent_passages_async(
                    actor=actor,
                    agent_id=agent_state.id,
                    query_text=query_texts[index],
                    limit=args.top_k,
                    embed_query=True,
                    embedding_config=agent_state.embedding_config,
                    query_embedding=query_vecs[index].tolist(),
                )
                search_latencies_ms[index] = (time.perf_counter() - search_start) * 1000.0

                insert_start = time.perf_counter()
                passage = PydanticPassage(
                    text=insert_texts[index],
                    embedding=insert_vecs[index].tolist(),
                    embedding_config=agent_state.embedding_config,
                    organization_id=actor.organization_id,
                    archive_id=archive.id,
                    metadata={"seed": args.seed, "type": "insert"},
                )
                await server.passage_manager.create_many_archival_passages_async([passage], actor=actor)
                insert_latencies_ms[index] = (time.perf_counter() - insert_start) * 1000.0

        print(f"[run:{mode}] ops={args.ops} parallel={args.parallel} top_k={args.top_k}")
        run_start = time.perf_counter()
        await asyncio.gather(*(run_op(i) for i in range(args.ops)))
        run_elapsed = time.perf_counter() - run_start

        search_values = [v for v in search_latencies_ms if v is not None]
        insert_values = [v for v in insert_latencies_ms if v is not None]

        summary = {
            "mode": mode,
            "base_count": args.base_count,
            "ops": args.ops,
            "parallel": args.parallel,
            "top_k": args.top_k,
            "dim": dim,
            "seed": args.seed,
            "hydrate_time_s": hydrate_elapsed,
            "wall_time_s": run_elapsed,
            "throughput_ops_per_s": (args.ops / run_elapsed) if run_elapsed > 0 else None,
            "search": _summarize_latencies(search_values),
            "insert": _summarize_latencies(insert_values),
        }

        if args.csv:
            csv_path = Path(args.csv)
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            csv_path = Path("benchmarks/cpu_db_bench") / f"{mode}_{timestamp}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8") as fp:
            fp.write("op_index,search_ms,insert_ms\n")
            for idx in range(args.ops):
                s = search_latencies_ms[idx]
                ins = insert_latencies_ms[idx]
                s_val = f"{s}" if s is not None else ""
                ins_val = f"{ins}" if ins is not None else ""
                fp.write(f"{idx},{s_val},{ins_val}\n")
        summary["csv_path"] = str(csv_path)

        if args.log_file:
            log_path = Path(args.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(json.dumps(summary, indent=2))

        return summary
    finally:
        if agent_state is not None:
            await server.agent_manager.delete_agent_async(agent_state.id, actor=actor)


async def _main_single(args: argparse.Namespace) -> None:
    _apply_mode_env(args.mode)
    summary = await _run_once(args, args.mode)
    print(json.dumps(summary, indent=2))


def _run_subprocess(script_path: Path, mode: str, args: argparse.Namespace, log_path: Path, csv_path: Optional[Path]) -> Dict[str, object]:
    cmd = [
        sys.executable,
        str(script_path),
        "--mode",
        mode,
        "--base-count",
        str(args.base_count),
        "--ops",
        str(args.ops),
        "--top-k",
        str(args.top_k),
        "--parallel",
        str(args.parallel),
        "--hydrate-batch",
        str(args.hydrate_batch),
        "--dim",
        str(args.dim),
        "--seed",
        str(args.seed),
        "--agent-model",
        args.agent_model,
        "--embedding-model",
        args.embedding_model,
        "--log-file",
        str(log_path),
    ]
    if csv_path is not None:
        cmd.extend(["--csv", str(csv_path)])

    subprocess.run(cmd, check=True)
    return json.loads(log_path.read_text())


def main() -> None:
    args = _parse_args()
    if args.mode == "both":
        script_path = Path(__file__).resolve()
        logs_dir = Path("benchmarks/cpu_db_bench")
        logs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        v2_log = logs_dir / f"v2_{timestamp}.json"
        pooled_log = logs_dir / f"v2_pooled_{timestamp}.json"
        base_csv = Path(args.csv) if args.csv else None
        v2_csv = None
        pooled_csv = None
        if base_csv is not None:
            v2_csv = base_csv.with_name(f"{base_csv.stem}_v2{base_csv.suffix or '.csv'}")
            pooled_csv = base_csv.with_name(f"{base_csv.stem}_v2_pooled{base_csv.suffix or '.csv'}")

        v2_summary = _run_subprocess(script_path, "v2", args, v2_log, v2_csv)
        pooled_summary = _run_subprocess(script_path, "v2_pooled", args, pooled_log, pooled_csv)
        print(json.dumps({"v2": v2_summary, "v2_pooled": pooled_summary}, indent=2))
    else:
        asyncio.run(_main_single(args))


if __name__ == "__main__":
    main()
