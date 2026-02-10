#!/usr/bin/env python
"""
DIAGNOSTIC VERSION: Test search-only vs search+insert to isolate bottleneck.
This will show if slow performance is from vector search or insert contention.
"""

from __future__ import annotations

import os
import sys

# Connection pool config
os.environ.setdefault("LETTA_DISABLE_SQLALCHEMY_POOLING", "false")
os.environ.setdefault("LETTA_PG_POOL_SIZE", "20")
os.environ.setdefault("LETTA_PG_MAX_OVERFLOW", "10")
os.environ.setdefault("LETTA_PG_POOL_TIMEOUT", "30")
os.environ.setdefault("LETTA_PG_POOL_RECYCLE", "1800")
os.environ.setdefault("LETTA_POOL_PRE_PING", "true")

import asyncio
import time
import statistics
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import json

from letta.schemas.agent import AgentState
from letta.schemas.passage import Passage as PydanticPassage
from letta.schemas.user import User
from letta.server.server import SyncServer


@dataclass
class QueryMetrics:
    """Metrics for a single query operation."""
    query_id: int
    search_latency_ms: float
    insert_latency_ms: Optional[float]
    total_latency_ms: float
    error: Optional[str]
    timeout: bool
    retrieved_count: int


@dataclass
class StrategyResults:
    """Aggregate results for a strategy."""
    strategy: str
    total_queries: int
    total_time_s: float
    
    # Search metrics
    avg_search_ms: float
    p50_search_ms: float
    p95_search_ms: float
    p99_search_ms: float
    
    # Insert metrics (if applicable)
    avg_insert_ms: Optional[float]
    p95_insert_ms: Optional[float]
    
    # End-to-end latency
    avg_total_ms: float
    p95_total_ms: float
    p99_total_ms: float
    
    # Throughput
    queries_per_second: float
    
    # Errors
    total_errors: int
    total_timeouts: int
    
    # DB operation estimates
    estimated_db_ops: int


class LettaPoolTester:
    """Test Letta agent with different parallelization strategies."""
    
    def __init__(
        self,
        server: SyncServer,
        actor: User,
        agent_state: AgentState,
        archive_id: str,
        num_queries: int = 100,
        top_k: int = 5,
        chunk_size: int = 10,
    ):
        self.server = server
        self.actor = actor
        self.agent_state = agent_state
        self.archive_id = archive_id
        self.num_queries = num_queries
        self.top_k = top_k
        self.chunk_size = chunk_size
        
        # Pre-generate test data
        self.query_texts = [
            f"test query {i}: what is the meaning of life number {i}?"
            for i in range(num_queries)
        ]
        
        # Simple embedding vectors (normally would use real encoder)
        import numpy as np
        self.query_vectors = [
            np.random.randn(1024).astype(np.float32).tolist()
            for _ in range(num_queries)
        ]
    
    async def _perform_single_search(
        self,
        query_id: int,
    ) -> QueryMetrics:
        """Perform a single vector search operation."""
        start = time.perf_counter()
        error = None
        timeout = False
        retrieved_count = 0
        
        try:
            tuples = await self.server.agent_manager.query_agent_passages_async(
                actor=self.actor,
                agent_id=self.agent_state.id,
                query_text=self.query_texts[query_id],
                limit=self.top_k,
                embed_query=True,
                embedding_config=self.agent_state.embedding_config,
                query_embedding=self.query_vectors[query_id],
            )
            retrieved_count = len(tuples) if tuples else 0
            
        except asyncio.TimeoutError:
            timeout = True
            error = "TimeoutError"
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)}"
        
        search_latency = (time.perf_counter() - start) * 1000  # ms
        
        return QueryMetrics(
            query_id=query_id,
            search_latency_ms=search_latency,
            insert_latency_ms=None,
            total_latency_ms=search_latency,
            error=error,
            timeout=timeout,
            retrieved_count=retrieved_count,
        )
    
    async def _perform_single_insert(
        self,
        query_id: int,
        text: str,
        embedding: List[float],
    ) -> float:
        """Perform a single passage insert operation. Returns latency in ms."""
        passage = PydanticPassage(
            text=text,
            embedding=embedding,
            embedding_config=self.agent_state.embedding_config,
            organization_id=self.actor.organization_id,
            archive_id=self.archive_id,
            metadata={"test_id": query_id, "strategy": "test"},
        )
        
        start = time.perf_counter()
        await self.server.passage_manager.create_many_archival_passages_async(
            [passage],
            actor=self.actor,
        )
        return (time.perf_counter() - start) * 1000  # ms
    
    async def _perform_search_and_insert(
        self,
        query_id: int,
    ) -> QueryMetrics:
        """Perform search followed by insert."""
        # Search
        search_result = await self._perform_single_search(query_id)
        
        if search_result.error:
            return search_result
        
        # Insert
        try:
            insert_latency = await self._perform_single_insert(
                query_id,
                f"generated response for query {query_id}",
                self.query_vectors[query_id],
            )
            
            search_result.insert_latency_ms = insert_latency
            search_result.total_latency_ms = search_result.search_latency_ms + insert_latency
            
        except Exception as e:
            search_result.error = f"Insert failed: {type(e).__name__}"
        
        return search_result
    
    async def test_sequential(self, include_inserts: bool = True) -> StrategyResults:
        """Strategy 1: Process queries one at a time."""
        strategy_name = "Sequential" + (" + Insert" if include_inserts else " (Search Only)")
        print(f"\n{'='*70}")
        print(f"STRATEGY: {strategy_name}")
        print(f"  Processing {self.num_queries} queries one at a time...")
        print(f"{'='*70}")
        
        results: List[QueryMetrics] = []
        start_time = time.perf_counter()
        
        for i in range(self.num_queries):
            if include_inserts:
                result = await self._perform_search_and_insert(i)
            else:
                result = await self._perform_single_search(i)
            
            results.append(result)
            
            if (i + 1) % 25 == 0:
                elapsed = time.perf_counter() - start_time
                rate = (i + 1) / elapsed
                print(f"  Progress: {i+1}/{self.num_queries} ({rate:.1f} qps)")
        
        total_time = time.perf_counter() - start_time
        
        return self._calculate_results(strategy_name, results, total_time, include_inserts)
    
    async def test_chunked_parallel(self, include_inserts: bool = True) -> StrategyResults:
        """Strategy 3: Process in controlled chunks (the fix)."""
        strategy_name = f"Chunked Parallel (chunk={self.chunk_size})" + (" + Insert" if include_inserts else " (Search Only)")
        print(f"\n{'='*70}")
        print(f"STRATEGY: {strategy_name}")
        print(f"  Processing {self.num_queries} queries in chunks of {self.chunk_size}...")
        print(f"{'='*70}")
        
        results: List[QueryMetrics] = []
        start_time = time.perf_counter()
        
        num_chunks = (self.num_queries + self.chunk_size - 1) // self.chunk_size
        
        for chunk_idx in range(0, self.num_queries, self.chunk_size):
            chunk_end = min(chunk_idx + self.chunk_size, self.num_queries)
            chunk_num = chunk_idx // self.chunk_size + 1
            
            # Create tasks ONLY for this chunk
            if include_inserts:
                tasks = [
                    self._perform_search_and_insert(i)
                    for i in range(chunk_idx, chunk_end)
                ]
            else:
                tasks = [
                    self._perform_single_search(i)
                    for i in range(chunk_idx, chunk_end)
                ]
            
            print(f"  Chunk {chunk_num}/{num_chunks}: processing queries {chunk_idx}-{chunk_end-1} ({len(tasks)} concurrent)")
            
            # Wait for this chunk
            chunk_results = await asyncio.gather(*tasks)
            results.extend(chunk_results)
        
        total_time = time.perf_counter() - start_time
        
        return self._calculate_results(strategy_name, results, total_time, include_inserts)
    
    def _calculate_results(
        self,
        strategy: str,
        metrics: List[QueryMetrics],
        total_time: float,
        include_inserts: bool,
    ) -> StrategyResults:
        """Calculate aggregate metrics."""
        search_latencies = [m.search_latency_ms for m in metrics]
        search_sorted = sorted(search_latencies)
        
        insert_latencies = [m.insert_latency_ms for m in metrics if m.insert_latency_ms is not None]
        insert_sorted = sorted(insert_latencies) if insert_latencies else []
        
        total_latencies = [m.total_latency_ms for m in metrics]
        total_sorted = sorted(total_latencies)
        
        errors = sum(1 for m in metrics if m.error)
        timeouts = sum(1 for m in metrics if m.timeout)
        
        # Estimate DB operations
        searches = len(metrics)
        inserts = len([m for m in metrics if m.insert_latency_ms is not None])
        estimated_ops = (searches * 4) + (inserts * 3)
        
        return StrategyResults(
            strategy=strategy,
            total_queries=len(metrics),
            total_time_s=total_time,
            avg_search_ms=statistics.mean(search_latencies),
            p50_search_ms=search_sorted[len(search_sorted) // 2],
            p95_search_ms=search_sorted[int(len(search_sorted) * 0.95)],
            p99_search_ms=search_sorted[int(len(search_sorted) * 0.99)],
            avg_insert_ms=statistics.mean(insert_sorted) if insert_sorted else None,
            p95_insert_ms=insert_sorted[int(len(insert_sorted) * 0.95)] if insert_sorted else None,
            avg_total_ms=statistics.mean(total_latencies),
            p95_total_ms=total_sorted[int(len(total_sorted) * 0.95)],
            p99_total_ms=total_sorted[int(len(total_sorted) * 0.99)],
            queries_per_second=len(metrics) / total_time,
            total_errors=errors,
            total_timeouts=timeouts,
            estimated_db_ops=estimated_ops,
        )


def print_comparison(results: List[StrategyResults]):
    """Print comparison table."""
    print(f"\n{'='*100}")
    print("PERFORMANCE COMPARISON - DIAGNOSTIC MODE")
    print(f"{'='*100}")
    print(f"{'Strategy':<40} {'Time(s)':<10} {'Search(ms)':<12} {'Insert(ms)':<12} {'QPS':<8} {'Errors'}")
    print(f"{'-'*100}")
    
    for r in results:
        insert_str = f"{r.avg_insert_ms:.1f}" if r.avg_insert_ms else "N/A"
        print(
            f"{r.strategy:<40} "
            f"{r.total_time_s:<10.2f} "
            f"{r.avg_search_ms:<12.2f} "
            f"{insert_str:<12} "
            f"{r.queries_per_second:<8.1f} "
            f"{r.total_errors + r.total_timeouts}"
        )
    
    print(f"{'='*100}\n")
    
    # Analysis
    print("🔍 DIAGNOSTIC ANALYSIS")
    print(f"{'-'*100}")
    
    if len(results) >= 4:
        seq_search = results[0]  # Sequential search-only
        seq_insert = results[1]  # Sequential search+insert
        chunk_search = results[2]  # Chunked search-only
        chunk_insert = results[3]  # Chunked search+insert
        
        print(f"\n1. SEQUENTIAL PERFORMANCE:")
        print(f"   Search-only latency: {seq_search.avg_search_ms:.2f}ms")
        print(f"   Search+insert latency: {seq_insert.avg_search_ms:.2f}ms search + {seq_insert.avg_insert_ms:.2f}ms insert")
        
        insert_overhead = seq_insert.avg_search_ms - seq_search.avg_search_ms
        if insert_overhead > 100:
            print(f"   ⚠️  Insert adds {insert_overhead:.2f}ms to search latency (index contention!)")
        else:
            print(f"   ✓  Insert overhead minimal: {insert_overhead:.2f}ms")
        
        print(f"\n2. CHUNKED PARALLEL PERFORMANCE:")
        print(f"   Search-only: {chunk_search.total_time_s:.2f}s total ({chunk_search.avg_search_ms:.2f}ms avg)")
        print(f"   Search+insert: {chunk_insert.total_time_s:.2f}s total ({chunk_insert.avg_search_ms:.2f}ms avg)")
        
        chunk_slowdown = chunk_insert.avg_search_ms / seq_search.avg_search_ms
        if chunk_slowdown > 3:
            print(f"   ⚠️  Chunked is {chunk_slowdown:.1f}x SLOWER than sequential search!")
            print(f"   ⚠️  This indicates severe pool/index contention")
        elif chunk_slowdown > 1.5:
            print(f"   ⚠️  Chunked has {chunk_slowdown:.1f}x latency increase (moderate contention)")
        else:
            print(f"   ✓  Chunked latency acceptable ({chunk_slowdown:.1f}x sequential)")
        
        print(f"\n3. THROUGHPUT GAINS:")
        seq_throughput_gain = chunk_search.queries_per_second / seq_search.queries_per_second
        print(f"   Search-only: {seq_throughput_gain:.2f}x faster with chunking")
        
        if seq_throughput_gain < 3:
            print(f"   ⚠️  Expected ~{chunk_search.chunk_size}x, got {seq_throughput_gain:.2f}x")
            print(f"   ⚠️  Pool is bottlenecked - consider increasing LETTA_PG_POOL_SIZE")
        
        print(f"\n4. RECOMMENDATION:")
        if chunk_slowdown > 3:
            print(f"   🔴 CRITICAL: Reduce chunk size from {chunk_search.chunk_size} to {chunk_search.chunk_size // 2}")
            print(f"   🔴 OR increase pool: LETTA_PG_POOL_SIZE=40 LETTA_PG_MAX_OVERFLOW=20")
        elif chunk_slowdown > 1.5:
            print(f"   🟡 MODERATE: Try chunk_size={chunk_search.chunk_size // 2} or pool_size=30")
        else:
            print(f"   🟢 OPTIMAL: Current settings work well")


async def create_test_agent(server: SyncServer, actor: User) -> tuple[AgentState, str]:
    """Create a test agent with minimal configuration."""
    from letta.schemas.agent import CreateAgent
    from letta.schemas.embedding_config import EmbeddingConfig
    from letta.schemas.llm_config import LLMConfig
    
    print("\n[setup] Creating test agent...")
    
    agent_create = CreateAgent(
        name="pool-test-agent",
        llm_config=LLMConfig.default_config("gpt-4"),
        embedding_config=EmbeddingConfig.default_config(model_name="letta"),
    )
    agent_state = await server.agent_manager.create_agent_async(agent_create=agent_create, actor=actor)
    
    archive = await server.archive_manager.get_or_create_default_archive_for_agent_async(
        agent_state=agent_state,
        actor=actor,
    )
    
    # Seed with baseline passages
    print("[setup] Seeding archival memory with 50 baseline passages...")
    import numpy as np
    
    passages = [
        PydanticPassage(
            text=f"baseline passage {i}: some contextual information",
            embedding=np.random.randn(1024).astype(np.float32).tolist(),
            embedding_config=agent_state.embedding_config,
            organization_id=actor.organization_id,
            archive_id=archive.id,
            metadata={"baseline": True, "index": i},
        )
        for i in range(50)
    ]
    
    await server.passage_manager.create_many_archival_passages_async(
        passages,
        actor=actor,
    )
    
    print(f"[setup] Agent created: {agent_state.id}")
    return agent_state, archive.id


async def cleanup_agent(server: SyncServer, actor: User, agent_id: str):
    """Clean up test agent."""
    print(f"\n[cleanup] Deleting agent {agent_id}...")
    await server.agent_manager.delete_agent_async(agent_id, actor=actor)


async def main():
    """Run the diagnostic test suite."""
    print(f"\n{'#'*100}")
    print("LETTA POOL DIAGNOSTIC TEST - SEARCH vs SEARCH+INSERT")
    print(f"{'#'*100}")
    
    # Configuration
    num_queries = int(os.environ.get("TEST_NUM_QUERIES", "50"))  # Reduced for faster diagnostics
    chunk_size = int(os.environ.get("TEST_CHUNK_SIZE", "10"))
    pool_size = int(os.environ.get("LETTA_PG_POOL_SIZE", "20"))
    pool_overflow = int(os.environ.get("LETTA_PG_MAX_OVERFLOW", "10"))
    
    print(f"Configuration:")
    print(f"  • Queries: {num_queries}")
    print(f"  • Chunk size: {chunk_size}")
    print(f"  • Pool size: {pool_size}")
    print(f"  • Pool overflow: {pool_overflow}")
    print(f"  • Total pool capacity: {pool_size + pool_overflow}")
    print(f"\nThis test will isolate whether slowness is from:")
    print(f"  1. Vector search operations")
    print(f"  2. Insert operations / index contention")
    print(f"  3. Connection pool exhaustion")
    print(f"{'#'*100}")
    
    server = SyncServer()
    actor = await server.user_manager.get_actor_or_default_async()
    
    agent_state, archive_id = await create_test_agent(server, actor)
    
    try:
        tester = LettaPoolTester(
            server=server,
            actor=actor,
            agent_state=agent_state,
            archive_id=archive_id,
            num_queries=num_queries,
            top_k=5,
            chunk_size=chunk_size,
        )
        
        all_results = []
        
        # Test 1: Sequential search-only (baseline)
        result_seq_search = await tester.test_sequential(include_inserts=False)
        all_results.append(result_seq_search)
        
        # Test 2: Sequential search+insert (shows insert overhead)
        result_seq_insert = await tester.test_sequential(include_inserts=True)
        all_results.append(result_seq_insert)
        
        # Test 3: Chunked search-only (shows parallel search performance)
        result_chunk_search = await tester.test_chunked_parallel(include_inserts=False)
        all_results.append(result_chunk_search)
        
        # Test 4: Chunked search+insert (shows if inserts kill performance)
        result_chunk_insert = await tester.test_chunked_parallel(include_inserts=True)
        all_results.append(result_chunk_insert)
        
        # Print diagnostic analysis
        print_comparison(all_results)
        
        # Save results
        output_file = Path("letta_pool_diagnostic.json")
        with output_file.open("w") as f:
            json.dump(
                [asdict(r) for r in all_results],
                f,
                indent=2,
            )
        print(f"\n📁 Detailed results saved to: {output_file}")
        
    finally:
        await cleanup_agent(server, actor, agent_state.id)


if __name__ == "__main__":
    asyncio.run(main())