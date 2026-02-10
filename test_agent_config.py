#!/usr/bin/env python
"""
SYSTEM CHOICES DIAGNOSTIC - Test 4 Key Configuration Dimensions

This script tests the impact of:
1. LETTA_DB_PARALLEL (1 vs higher values)
2. Connection Pooling (enabled vs disabled)
3. Search Chunk Size (how many queries per batch)
4. Insert Batch Size (how many inserts per batch)

Goal: Find optimal configuration with data to justify each choice.
"""

from __future__ import annotations

import asyncio
import os
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional
import json

import numpy as np


@dataclass
class SystemConfig:
    """A specific system configuration to test."""
    name: str
    db_parallel: int
    pooling_enabled: bool
    pool_size: int
    search_chunk_size: int
    insert_batch_size: int


@dataclass
class TestResult:
    """Results from testing one configuration."""
    config_name: str
    
    # Configuration details
    db_parallel: int
    pooling_enabled: bool
    pool_size: int
    search_chunk_size: int
    insert_batch_size: int
    
    # Search metrics
    search_total_time_s: float
    search_avg_latency_ms: float
    search_p95_latency_ms: float
    search_qps: float
    
    # Insert metrics
    insert_total_time_s: float
    insert_avg_latency_ms: float
    insert_qps: float
    
    # Combined metrics
    total_time_s: float
    total_qps: float
    
    # Quality metrics
    search_errors: int
    insert_errors: int


class LettaSystemTester:
    """Test Letta with different system configurations."""
    
    def __init__(
        self,
        num_queries: int = 100,
        top_k: int = 5,
    ):
        self.num_queries = num_queries
        self.top_k = top_k
        
        # Pre-generate test data
        self.query_texts = [
            f"test query {i}: what is the meaning of life number {i}?"
            for i in range(num_queries)
        ]
        
        # Simple embedding vectors
        self.query_vectors = [
            np.random.randn(1024).astype(np.float32).tolist()
            for _ in range(num_queries)
        ]
        
        # Response texts for inserts
        self.response_texts = [
            f"generated response for query {i}: the answer is {i * 42}"
            for i in range(num_queries)
        ]
        
        self.response_vectors = [
            np.random.randn(1024).astype(np.float32).tolist()
            for _ in range(num_queries)
        ]
    
    def _apply_config(self, config: SystemConfig):
        """Apply a configuration to environment variables."""
        # DB Parallel
        os.environ["LETTA_DB_PARALLEL"] = str(config.db_parallel)
        
        # Pooling
        if config.pooling_enabled:
            os.environ["LETTA_DISABLE_SQLALCHEMY_POOLING"] = "false"
            os.environ["LETTA_PG_POOL_SIZE"] = str(config.pool_size)
            os.environ["LETTA_PG_MAX_OVERFLOW"] = str(config.pool_size // 2)
        else:
            os.environ["LETTA_DISABLE_SQLALCHEMY_POOLING"] = "true"
        
        os.environ["LETTA_PG_POOL_TIMEOUT"] = "30"
        os.environ["LETTA_PG_POOL_RECYCLE"] = "1800"
        os.environ["LETTA_POOL_PRE_PING"] = "true"
    
    async def _setup_agent(self):
        """Create a fresh agent for testing. Reimport to pick up new env vars."""
        # Force reimport of letta modules to pick up new config
        import sys
        letta_modules = [k for k in sys.modules.keys() if k.startswith('letta')]
        for mod in letta_modules:
            del sys.modules[mod]
        
        from letta.schemas.agent import CreateAgent
        from letta.schemas.embedding_config import EmbeddingConfig
        from letta.schemas.llm_config import LLMConfig
        from letta.schemas.passage import Passage as PydanticPassage
        from letta.server.server import SyncServer
        
        server = SyncServer()
        actor = await server.user_manager.get_actor_or_default_async()
        
        agent_create = CreateAgent(
            name="system-test-agent",
            llm_config=LLMConfig.default_config("gpt-4"),
            embedding_config=EmbeddingConfig.default_config(model_name="letta"),
        )
        agent_state = await server.agent_manager.create_agent_async(
            agent_create=agent_create, 
            actor=actor
        )
        
        archive = await server.archive_manager.get_or_create_default_archive_for_agent_async(
            agent_state=agent_state,
            actor=actor,
        )
        
        # Seed with baseline passages
        passages = [
            PydanticPassage(
                text=f"baseline passage {i}: contextual information",
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
        
        return server, actor, agent_state, archive.id
    
    async def _cleanup_agent(self, server, actor, agent_id: str):
        """Clean up test agent."""
        await server.agent_manager.delete_agent_async(agent_id, actor=actor)
    
    async def _run_searches(
        self,
        server,
        actor,
        agent_state,
        chunk_size: int,
    ) -> tuple[float, List[float], int]:
        """Run all searches with specified chunk size. Returns (total_time, latencies, errors)."""
        from letta.schemas.passage import Passage as PydanticPassage
        
        latencies_ms = []
        errors = 0
        start_time = time.perf_counter()
        
        for chunk_start in range(0, self.num_queries, chunk_size):
            chunk_end = min(chunk_start + chunk_size, self.num_queries)
            
            tasks = []
            for i in range(chunk_start, chunk_end):  # FIX: was range(chunk_start, chunk_size)
                tasks.append(
                    self._single_search(server, actor, agent_state, i)
                )
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    errors += 1
                    latencies_ms.append(0.0)
                else:
                    latencies_ms.append(result)
        
        total_time = time.perf_counter() - start_time
        return total_time, latencies_ms, errors
    
    async def _single_search(
        self,
        server,
        actor,
        agent_state,
        query_index: int,
    ) -> float:
        """Perform single search, return latency in ms."""
        start = time.perf_counter()
        
        try:
            tuples = await server.agent_manager.query_agent_passages_async(
                actor=actor,
                agent_id=agent_state.id,
                query_text=self.query_texts[query_index],
                limit=self.top_k,
                embed_query=True,
                embedding_config=agent_state.embedding_config,
                query_embedding=self.query_vectors[query_index],
            )
        except Exception as e:
            raise e
        
        return (time.perf_counter() - start) * 1000.0
    
    async def _run_inserts(
        self,
        server,
        actor,
        agent_state,
        archive_id: str,
        batch_size: int,
    ) -> tuple[float, List[float], int]:
        """Run all inserts with specified batch size. Returns (total_time, latencies, errors)."""
        from letta.schemas.passage import Passage as PydanticPassage
        
        latencies_ms = []
        errors = 0
        start_time = time.perf_counter()
        
        for batch_start in range(0, self.num_queries, batch_size):
            batch_end = min(batch_start + batch_size, self.num_queries)
            
            passages = [
                PydanticPassage(
                    text=self.response_texts[i],
                    embedding=self.response_vectors[i],
                    embedding_config=agent_state.embedding_config,
                    organization_id=actor.organization_id,
                    archive_id=archive_id,
                    metadata={"test_id": i, "strategy": "system_test"},
                )
                for i in range(batch_start, batch_end)
            ]
            
            batch_start_time = time.perf_counter()
            try:
                await server.passage_manager.create_many_archival_passages_async(
                    passages,
                    actor=actor,
                )
                batch_latency = (time.perf_counter() - batch_start_time) * 1000.0
                per_item_latency = batch_latency / len(passages)
                latencies_ms.extend([per_item_latency] * len(passages))
            except Exception as e:
                errors += len(passages)
                latencies_ms.extend([0.0] * len(passages))
        
        total_time = time.perf_counter() - start_time
        return total_time, latencies_ms, errors
    
    async def test_configuration(self, config: SystemConfig) -> TestResult:
        """Test a specific configuration."""
        print(f"\n{'='*80}")
        print(f"TESTING: {config.name}")
        print(f"  DB Parallel: {config.db_parallel}")
        print(f"  Pooling: {'enabled' if config.pooling_enabled else 'disabled'}")
        if config.pooling_enabled:
            print(f"  Pool Size: {config.pool_size}")
        print(f"  Search Chunk: {config.search_chunk_size}")
        print(f"  Insert Batch: {config.insert_batch_size}")
        print(f"{'='*80}")
        
        # Apply configuration
        self._apply_config(config)
        
        # Setup agent
        server, actor, agent_state, archive_id = await self._setup_agent()
        
        try:
            # Run searches
            print(f"  Running {self.num_queries} searches...")
            search_time, search_latencies, search_errors = await self._run_searches(
                server, actor, agent_state, config.search_chunk_size
            )
            
            # Validate we ran all queries
            if len(search_latencies) != self.num_queries:
                print(f"  ⚠️  WARNING: Expected {self.num_queries} search results, got {len(search_latencies)}")
            
            # Run inserts
            print(f"  Running {self.num_queries} inserts...")
            insert_time, insert_latencies, insert_errors = await self._run_inserts(
                server, actor, agent_state, archive_id, config.insert_batch_size
            )
            
            # Validate inserts
            if len(insert_latencies) != self.num_queries:
                print(f"  ⚠️  WARNING: Expected {self.num_queries} insert results, got {len(insert_latencies)}")
            
            # Calculate metrics
            search_sorted = sorted([l for l in search_latencies if l > 0])
            insert_sorted = sorted([l for l in insert_latencies if l > 0])
            
            result = TestResult(
                config_name=config.name,
                db_parallel=config.db_parallel,
                pooling_enabled=config.pooling_enabled,
                pool_size=config.pool_size,
                search_chunk_size=config.search_chunk_size,
                insert_batch_size=config.insert_batch_size,
                search_total_time_s=search_time,
                search_avg_latency_ms=statistics.mean(search_sorted) if search_sorted else 0,
                search_p95_latency_ms=search_sorted[int(len(search_sorted) * 0.95)] if search_sorted else 0,
                search_qps=self.num_queries / search_time if search_time > 0 else 0,
                insert_total_time_s=insert_time,
                insert_avg_latency_ms=statistics.mean(insert_sorted) if insert_sorted else 0,
                insert_qps=self.num_queries / insert_time if insert_time > 0 else 0,
                total_time_s=search_time + insert_time,
                total_qps=self.num_queries / (search_time + insert_time) if (search_time + insert_time) > 0 else 0,
                search_errors=search_errors,
                insert_errors=insert_errors,
            )
            
            print(f"  ✓ Search: {search_time:.2f}s ({result.search_avg_latency_ms:.1f}ms avg, {result.search_qps:.1f} qps)")
            print(f"  ✓ Insert: {insert_time:.2f}s ({result.insert_avg_latency_ms:.1f}ms avg, {result.insert_qps:.1f} qps)")
            print(f"  ✓ Total: {result.total_time_s:.2f}s ({result.total_qps:.1f} qps)")
            
            return result
            
        finally:
            await self._cleanup_agent(server, actor, agent_state.id)


def print_comparison(results: List[TestResult]):
    """Print detailed comparison table."""
    print(f"\n{'='*120}")
    print("SYSTEM CHOICES COMPARISON")
    print(f"{'='*120}")
    print(f"{'Config':<30} {'DBPar':<7} {'Pool':<6} {'SrchCh':<8} {'InsBat':<8} {'Search(s)':<11} {'Insert(s)':<11} {'Total(s)':<10} {'QPS':<8}")
    print(f"{'-'*120}")
    
    for r in results:
        pool_str = f"{r.pool_size}" if r.pooling_enabled else "OFF"
        print(
            f"{r.config_name:<30} "
            f"{r.db_parallel:<7} "
            f"{pool_str:<6} "
            f"{r.search_chunk_size:<8} "
            f"{r.insert_batch_size:<8} "
            f"{r.search_total_time_s:>6.2f}({r.search_avg_latency_ms:>4.0f}ms) "
            f"{r.insert_total_time_s:>6.2f}({r.insert_avg_latency_ms:>4.0f}ms) "
            f"{r.total_time_s:<10.2f} "
            f"{r.total_qps:<8.1f}"
        )
    
    print(f"{'='*120}\n")
    
    # Analysis
    baseline = results[0]
    print("🔍 DIAGNOSTIC INSIGHTS")
    print(f"{'-'*120}")
    
    # 1. DB Parallel Impact
    print(f"\n1️⃣  DATABASE PARALLELISM (LETTA_DB_PARALLEL)")
    db_par_results = [r for r in results if "DB Parallel" in r.config_name or r.config_name == "Baseline"]
    if len(db_par_results) >= 2:
        db_par_sorted = sorted(db_par_results, key=lambda r: r.db_parallel)
        
        print(f"   {'Value':<10} {'Total Time':<12} {'Search Latency':<18} {'Speedup vs Baseline':<20}")
        print(f"   {'-'*70}")
        baseline_db = [r for r in db_par_sorted if r.db_parallel == 1][0]
        for r in db_par_sorted:
            speedup = baseline_db.total_time_s / r.total_time_s
            print(f"   DB={r.db_parallel:<8} {r.total_time_s:>6.2f}s      {r.search_avg_latency_ms:>6.0f}ms           {speedup:>6.2f}x")
        
        best = min(db_par_sorted, key=lambda r: r.total_time_s)
        best_speedup = baseline_db.total_time_s / best.total_time_s
        
        if best_speedup > 1.3:
            print(f"   ✅ BENEFICIAL: DB_PARALLEL={best.db_parallel} is {best_speedup:.2f}x faster")
            print(f"   💡 RECOMMENDATION: Use LETTA_DB_PARALLEL={best.db_parallel}")
        elif best_speedup < 1.1:
            print(f"   ⚠️  MARGINAL: Best improvement is only {best_speedup:.2f}x")
            print(f"   💡 RECOMMENDATION: Keep LETTA_DB_PARALLEL=1 (DB parallelism doesn't help)")
        else:
            print(f"   ✅ MODERATE BENEFIT: DB_PARALLEL={best.db_parallel} is {best_speedup:.2f}x faster")
            print(f"   💡 RECOMMENDATION: Use LETTA_DB_PARALLEL={best.db_parallel}")
    
    # 2. Pooling Impact
    print(f"\n2️⃣  CONNECTION POOLING")
    pooling_results = [r for r in results if "Pool" in r.config_name or r.config_name == "Baseline"]
    if len(pooling_results) >= 2:
        pool_sorted = sorted([r for r in pooling_results if r.pooling_enabled], key=lambda r: r.pool_size)
        no_pool = [r for r in pooling_results if not r.pooling_enabled]
        
        print(f"   {'Config':<20} {'Total Time':<12} {'Search Latency':<18} {'Speedup vs No Pool':<20}")
        print(f"   {'-'*80}")
        
        baseline_no_pool = no_pool[0] if no_pool else None
        
        for r in pool_sorted:
            if baseline_no_pool:
                speedup = baseline_no_pool.total_time_s / r.total_time_s
                print(f"   Pool={r.pool_size:<16} {r.total_time_s:>6.2f}s      {r.search_avg_latency_ms:>6.0f}ms           {speedup:>6.2f}x")
            else:
                print(f"   Pool={r.pool_size:<16} {r.total_time_s:>6.2f}s      {r.search_avg_latency_ms:>6.0f}ms")
        
        if baseline_no_pool:
            print(f"   {'No pooling':<20} {baseline_no_pool.total_time_s:>6.2f}s      {baseline_no_pool.search_avg_latency_ms:>6.0f}ms           1.00x (baseline)")
        
        if pool_sorted:
            best_pool = min(pool_sorted, key=lambda r: r.total_time_s)
            if baseline_no_pool:
                speedup = baseline_no_pool.total_time_s / best_pool.total_time_s
                if speedup > 1.2:
                    print(f"   ✅ POOLING HELPS: Pool size={best_pool.pool_size} is {speedup:.2f}x faster than no pooling")
                    print(f"   💡 RECOMMENDATION: Enable pooling with POOL_SIZE={best_pool.pool_size}")
                elif speedup < 0.9:
                    print(f"   ❌ POOLING HURTS: {1/speedup:.2f}x slower")
                    print(f"   💡 RECOMMENDATION: Disable pooling")
                else:
                    print(f"   ⚠️  MARGINAL: Pooling gives {speedup:.2f}x improvement")
                    print(f"   💡 RECOMMENDATION: Enable pooling with POOL_SIZE={best_pool.pool_size} (slight benefit)")
            else:
                # Compare pool sizes to baseline
                baseline_pool_20 = [r for r in pool_sorted if r.pool_size == 20][0] if any(r.pool_size == 20 for r in pool_sorted) else pool_sorted[0]
                if best_pool.pool_size != baseline_pool_20.pool_size:
                    speedup = baseline_pool_20.total_time_s / best_pool.total_time_s
                    print(f"   ✅ OPTIMAL: Pool size={best_pool.pool_size} is {speedup:.2f}x faster than pool_size={baseline_pool_20.pool_size}")
                    print(f"   💡 RECOMMENDATION: Use POOL_SIZE={best_pool.pool_size}")
    
    # 3. Search Chunk Size Impact
    print(f"\n3️⃣  SEARCH CHUNK SIZE (Concurrent Queries)")
    chunk_results = [r for r in results if "Chunk" in r.config_name or r.config_name == "Baseline"]
    if len(chunk_results) >= 2:
        chunk_results_sorted = sorted(chunk_results, key=lambda r: r.search_chunk_size)
        
        print(f"   {'Chunk Size':<12} {'Total Time':<12} {'Search Latency':<18} {'QPS':<10} {'Latency Penalty':<18}")
        print(f"   {'-'*80}")
        
        min_latency = min(r.search_avg_latency_ms for r in chunk_results_sorted)
        
        for r in chunk_results_sorted:
            penalty = r.search_avg_latency_ms / min_latency
            print(f"   Chunk={r.search_chunk_size:<7} {r.search_total_time_s:>6.2f}s      {r.search_avg_latency_ms:>6.0f}ms           {r.search_qps:>6.1f}     {penalty:>6.2f}x")
        
        best_throughput = min(chunk_results_sorted, key=lambda r: r.search_total_time_s)
        best_latency = min(chunk_results_sorted, key=lambda r: r.search_avg_latency_ms)
        
        if best_throughput.search_chunk_size == best_latency.search_chunk_size:
            print(f"   ✅ OPTIMAL: chunk_size={best_throughput.search_chunk_size} gives best throughput AND latency")
            print(f"   💡 RECOMMENDATION: Use search_chunk_size={best_throughput.search_chunk_size}")
        else:
            latency_penalty = best_throughput.search_avg_latency_ms / best_latency.search_avg_latency_ms
            throughput_gain = best_latency.search_total_time_s / best_throughput.search_total_time_s
            
            if latency_penalty > 3.0:
                print(f"   ⚠️  TRADEOFF: chunk_size={best_throughput.search_chunk_size} is fastest ({throughput_gain:.2f}x) but has {latency_penalty:.1f}x latency penalty")
                print(f"   💡 For LOW LATENCY: Use chunk_size={best_latency.search_chunk_size}")
                print(f"   💡 For HIGH THROUGHPUT: Use chunk_size={best_throughput.search_chunk_size}")
            else:
                print(f"   ✅ GOOD SCALING: chunk_size={best_throughput.search_chunk_size} is {throughput_gain:.2f}x faster with {latency_penalty:.1f}x latency penalty")
                print(f"   💡 RECOMMENDATION: Use chunk_size={best_throughput.search_chunk_size}")
    
    # 4. Insert Batch Size Impact
    print(f"\n4️⃣  INSERT BATCH SIZE")
    insert_results = [r for r in results if "Insert Batch" in r.config_name or r.config_name == "Baseline"]
    if len(insert_results) >= 2:
        insert_results_sorted = sorted(insert_results, key=lambda r: r.insert_batch_size)
        
        print(f"   {'Batch Size':<12} {'Total Time':<12} {'Insert Latency':<18} {'QPS':<10} {'Speedup vs Batch=10':<20}")
        print(f"   {'-'*85}")
        
        baseline_insert = insert_results_sorted[0]  # Smallest batch
        
        for r in insert_results_sorted:
            speedup = baseline_insert.insert_total_time_s / r.insert_total_time_s
            print(f"   Batch={r.insert_batch_size:<7} {r.insert_total_time_s:>6.2f}s      {r.insert_avg_latency_ms:>6.1f}ms           {r.insert_qps:>6.1f}     {speedup:>6.2f}x")
        
        best = min(insert_results_sorted, key=lambda r: r.insert_total_time_s)
        speedup = baseline_insert.insert_total_time_s / best.insert_total_time_s
        
        if speedup > 2.0:
            print(f"   ✅ BATCHING CRITICAL: batch={best.insert_batch_size} is {speedup:.1f}x faster than batch={baseline_insert.insert_batch_size}")
            print(f"   💡 RECOMMENDATION: Use insert_batch_size={best.insert_batch_size}")
        elif speedup > 1.3:
            print(f"   ✅ BATCHING HELPS: batch={best.insert_batch_size} is {speedup:.1f}x faster")
            print(f"   💡 RECOMMENDATION: Use insert_batch_size={best.insert_batch_size}")
        else:
            print(f"   ⚠️  MARGINAL: Best improvement is only {speedup:.2f}x")
            print(f"   💡 RECOMMENDATION: insert_batch_size doesn't matter much, use {best.insert_batch_size}")
    
    # Final recommendation
    print(f"\n{'='*120}")
    print("🎯 FINAL RECOMMENDATION")
    print(f"{'='*120}")
    
    best_overall = min(results, key=lambda r: r.total_time_s)
    
    print(f"Best configuration: {best_overall.config_name}")
    print(f"  • LETTA_DB_PARALLEL={best_overall.db_parallel}")
    print(f"  • Pooling: {'enabled' if best_overall.pooling_enabled else 'disabled'}")
    if best_overall.pooling_enabled:
        print(f"  • LETTA_PG_POOL_SIZE={best_overall.pool_size}")
    print(f"  • search_chunk_size={best_overall.search_chunk_size}")
    print(f"  • insert_batch_size={best_overall.insert_batch_size}")
    print(f"  • Performance: {best_overall.total_qps:.1f} qps ({best_overall.total_time_s:.2f}s total)")
    
    improvement = baseline.total_time_s / best_overall.total_time_s
    print(f"\n  Improvement over baseline: {improvement:.2f}x faster")
    print(f"{'='*120}\n")


async def main():
    """Run the system choices diagnostic."""
    print(f"\n{'#'*120}")
    print("LETTA SYSTEM CHOICES DIAGNOSTIC")
    print(f"{'#'*120}")
    
    num_queries = int(os.environ.get("TEST_NUM_QUERIES", "500"))
    
    print(f"Configuration:")
    print(f"  • Queries: {num_queries}")
    print(f"  • This will test 4 system dimensions:")
    print(f"    1. DB Parallelism (LETTA_DB_PARALLEL: 5, 10, 15, 20)")
    print(f"    2. Connection Pooling (enabled with different sizes, disabled)")
    print(f"    3. Search Chunk Size (10, 50, 100, 200)")
    print(f"    4. Insert Batch Size (10, 50, 100, 200)")
    print(f"{'#'*120}\n")
    
    tester = LettaSystemTester(num_queries=num_queries)
    
    # Define test configurations
    configs = [
    # ============================================================
    # BASELINE
    # ============================================================
    # SystemConfig(
    #     name="Baseline (Sequential)",
    #     db_parallel=1,
    #     pooling_enabled=True,
    #     pool_size=10,
    #     search_chunk_size=1,
    #     insert_batch_size=50,
    # ),
    
    # ============================================================
    # TEST 1: SEARCH CHUNK SIZE - FULL RANGE
    # Goal: Find where performance tops out or degrades
    # ============================================================
    # SystemConfig(
    #     name="Search Chunk=5",
    #     db_parallel=1,
    #     pooling_enabled=True,
    #     pool_size=15,              # pool > chunk for headroom
    #     search_chunk_size=5,
    #     insert_batch_size=100,     # Keep insert batch consistent
    # ),
    # SystemConfig(
    #     name="Search Chunk=10",
    #     db_parallel=1,
    #     pooling_enabled=True,
    #     pool_size=20,
    #     search_chunk_size=10,
    #     insert_batch_size=100,
    # ),
    # SystemConfig(
    #     name="Search Chunk=20",
    #     db_parallel=1,
    #     pooling_enabled=True,
    #     pool_size=30,
    #     search_chunk_size=20,
    #     insert_batch_size=100,
    # ),
    # SystemConfig(
    #     name="Search Chunk=50",
    #     db_parallel=1,
    #     pooling_enabled=True,
    #     pool_size=60,
    #     search_chunk_size=50,
    #     insert_batch_size=100,
    # ),
    # SystemConfig(
    #     name="Search Chunk=100",
    #     db_parallel=1,
    #     pooling_enabled=True,
    #     pool_size=110,
    #     search_chunk_size=100,
    #     insert_batch_size=100,
    # ),
    # SystemConfig(
    #     name="Search Chunk=256",     # ← Added!
    #     db_parallel=1,
    #     pooling_enabled=True,
    #     pool_size=270,               # Pool > chunk
    #     search_chunk_size=256,
    #     insert_batch_size=100,
    # ),
    
    # ============================================================
    # TEST 2: INSERT BATCH SIZE - FULL RANGE
    # Keep search chunk constant at optimal-ish value (10)
    # ============================================================

    SystemConfig(
        name="Insert Batch=50",
        db_parallel=1,
        pooling_enabled=True,
        pool_size=20,
        search_chunk_size=10,
        insert_batch_size=50,
    ),
    SystemConfig(
        name="Insert Batch=100",
        db_parallel=1,
        pooling_enabled=True,
        pool_size=20,
        search_chunk_size=10,
        insert_batch_size=100,
    ),
    SystemConfig(
        name="Insert Batch=256",     # ← Added!
        db_parallel=1,
        pooling_enabled=True,
        pool_size=20,
        search_chunk_size=10,
        insert_batch_size=256,
    ),
    
    # ============================================================
    # TEST 3: POOL SIZE vs CHUNK SIZE RELATIONSHIP
    # Goal: Does pool > chunk actually matter?
    # ============================================================
    SystemConfig(
        name="Pool=Chunk (20:20)",   # Tight fit
        db_parallel=1,
        pooling_enabled=True,
        pool_size=20,
        search_chunk_size=20,
        insert_batch_size=100,
    ),
    SystemConfig(
        name="Pool=1.5×Chunk (30:20)", # Modest headroom
        db_parallel=1,
        pooling_enabled=True,
        pool_size=30,
        search_chunk_size=20,
        insert_batch_size=100,
    ),
    SystemConfig(
        name="Pool=2×Chunk (40:20)",   # Lots of headroom
        db_parallel=1,
        pooling_enabled=True,
        pool_size=40,
        search_chunk_size=20,
        insert_batch_size=100,
    ),
    
    # ============================================================
    # TEST 4: EXTREME CONFIGURATIONS
    # Goal: See where it breaks or plateaus
    # ============================================================
    # SystemConfig(
    #     name="EXTREME: Chunk=256, Pool=300",
    #     db_parallel=1,
    #     pooling_enabled=True,
    #     pool_size=300,
    #     search_chunk_size=256,
    #     insert_batch_size=256,
    # ),
    # SystemConfig(
    #     name="EXTREME: Chunk=500, Pool=550",  # Really push it
    #     db_parallel=1,
    #     pooling_enabled=True,
    #     pool_size=550,
    #     search_chunk_size=500,
    #     insert_batch_size=500,
    # ),
    
    # ============================================================
    # TEST 5: NO POOLING BASELINE
    # ============================================================
    SystemConfig(
        name="No Pooling",
        db_parallel=1,
        pooling_enabled=False,
        pool_size=0,
        search_chunk_size=1,         # Must be 1 without pooling
        insert_batch_size=100,
    ),
    ]
    results = []
    
    for config in configs:
        try:
            result = await tester.test_configuration(config)
            results.append(result)
        except Exception as e:
            print(f"  ❌ ERROR testing {config.name}: {e}")
    
    # Print comparison
    print_comparison(results)
    
    # Save results
    output_file = Path("letta_system_choices_results.json")
    with output_file.open("w") as f:
        json.dump(
            [asdict(r) for r in results],
            f,
            indent=2,
        )
    print(f"📁 Detailed results saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())