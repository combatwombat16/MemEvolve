"""
Google Memory Bank Provider — Vertex AI Agent Engine Memory Bank integration.

Uses Google's managed memory service for automatic fact extraction, consolidation,
and embedding-based similarity search. All memory storage is cloud-side; the local
storage directory only holds an engine name file for cross-run persistence.

Usage (direct eval):
    python run_flash_searcher_mm_gaia.py \\
        --memory_provider google_memory_bank --sample_num 20 --max_steps 40

Usage (evolution):
    python evolve_cli.py auto-evolve gaia \\
        --provider google_memory_bank --num-rounds 3

Scope strategy options (configured via scope_key / scope_value in config):
    - Shared (default): {"agent_id": "memevolve_agent"} — single namespace
    - Dataset-partitioned: {"agent_id": "memevolve_agent", "dataset": "gaia"}
    - Per-round isolation: {"agent_id": "memevolve_agent", "round": "01"}
"""

import logging
import os
import sys
import uuid
from typing import Any, Dict, List, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from ..base_memory import BaseMemoryProvider
from ..memory_types import (
    MemoryItem,
    MemoryItemType,
    MemoryRequest,
    MemoryResponse,
    MemoryStatus,
    MemoryType,
    TrajectoryData,
)

logger = logging.getLogger(__name__)


class GoogleMemoryBankProvider(BaseMemoryProvider):

    def __init__(self, config: Optional[dict] = None):
        super().__init__(MemoryType.GOOGLE_MEMORY_BANK, config)

        self.project_id = self.config.get("project_id", "")
        self.location = self.config.get("location", "us-central1")
        self.engine_name = self.config.get("engine_name", "")
        self.scope_key = self.config.get("scope_key", "agent_id")
        self.scope_value = self.config.get("scope_value", "memevolve_agent")
        self.top_k = self.config.get("top_k", 3)
        self.enable_consolidation = self.config.get("enable_consolidation", True)
        self.storage_dir = self.config.get("storage_dir", "./storage/google_memory_bank")
        self.model = self.config.get("model", None)

        self.client = None
        self.agent_engine_name: Optional[str] = None

    # ------------------------------------------------------------------
    # initialize
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        try:
            import vertexai  # noqa: F811 — deferred import to keep module loadable without SDK
        except ImportError:
            logger.error("google-cloud-aiplatform is not installed. Run: pip install google-cloud-aiplatform>=1.111.0")
            return False

        if not self.project_id:
            logger.error("GOOGLE_CLOUD_PROJECT is not set. Cannot initialize Google Memory Bank provider.")
            return False

        try:
            os.makedirs(self.storage_dir, exist_ok=True)

            self.client = vertexai.Client(project=self.project_id, location=self.location)

            # Engine resolution: env var → persisted file → create new
            engine_name_file = os.path.join(self.storage_dir, "engine_name.txt")

            if self.engine_name:
                # Path 1: env var / config provides the engine name
                self.agent_engine_name = self.engine_name
                logger.info("Using engine name from config/env: %s", self.agent_engine_name)
            elif os.path.isfile(engine_name_file):
                # Path 2: persisted from a previous run
                with open(engine_name_file, "r") as f:
                    self.agent_engine_name = f.read().strip()
                logger.info("Loaded engine name from file: %s", self.agent_engine_name)
            else:
                # Path 3: create a new engine
                engine = self.client.agent_engines.create()
                self.agent_engine_name = engine.name
                with open(engine_name_file, "w") as f:
                    f.write(self.agent_engine_name)
                logger.info("Created new engine: %s", self.agent_engine_name)

            return True

        except Exception as e:
            logger.error("Error initializing Google Memory Bank provider: %s", e)
            return False

    # ------------------------------------------------------------------
    # provide_memory
    # ------------------------------------------------------------------

    def provide_memory(self, request: MemoryRequest) -> MemoryResponse:
        empty = MemoryResponse(
            memories=[],
            memory_type=self.memory_type,
            total_count=0,
            request_id=str(uuid.uuid4()),
        )

        # IN-phase: return empty (matches AgentKB / SkillWeaver pattern)
        if request.status != MemoryStatus.BEGIN:
            return empty

        if not self.client or not self.agent_engine_name:
            return empty

        try:
            scope = {self.scope_key: self.scope_value}
            results = list(
                self.client.agent_engines.memories.retrieve(
                    name=self.agent_engine_name,
                    scope=scope,
                    similarity_search_params={
                        "search_query": request.query,
                        "top_k": self.top_k,
                    },
                )
            )

            if not results:
                return empty

            memory_items = []
            for mem in results:
                distance = getattr(mem, "distance", 0.0)
                memory_items.append(
                    MemoryItem(
                        id=mem.name,
                        content=mem.fact,
                        metadata={
                            "scope": getattr(mem, "scope", {}),
                            "distance": distance,
                        },
                        score=1.0 / (1.0 + distance),
                        type=MemoryItemType.TEXT,
                    )
                )

            # Optional synthesis: condense facts into coherent guidance
            if self.model and memory_items:
                synthesized = self._synthesize_guidance(memory_items, request)
                if synthesized:
                    memory_items = [synthesized]

            return MemoryResponse(
                memories=memory_items,
                memory_type=self.memory_type,
                total_count=len(memory_items),
                request_id=str(uuid.uuid4()),
            )

        except Exception as e:
            logger.error("Error retrieving memories: %s", e)
            return empty

    def _synthesize_guidance(self, items: List[MemoryItem], request: MemoryRequest) -> Optional[MemoryItem]:
        """Condense raw facts into actionable guidance using the injected model."""
        try:
            facts_text = "\n".join(f"- {item.content}" for item in items)
            prompt = (
                "Based on the following relevant memories from past experiences, provide "
                "concise actionable guidance for the current task.\n\n"
                f"Current Task: {request.query}\n\n"
                f"Retrieved Memories:\n{facts_text}\n\n"
                "Provide 2-3 specific suggestions. Use gentle, suggestive language."
            )

            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            response = self.model(messages)
            guidance = getattr(response, "content", str(response)).strip()

            if not guidance:
                return None

            avg_score = sum(item.score for item in items if item.score) / len(items)
            return MemoryItem(
                id=f"synthesized_{uuid.uuid4()}",
                content=f"Memory Bank Guidance:\n{guidance}",
                metadata={
                    "num_sources": len(items),
                    "avg_score": avg_score,
                    "source_ids": [item.id for item in items],
                },
                score=avg_score,
            )

        except Exception as e:
            logger.error("Error synthesizing guidance: %s", e)
            return None

    # ------------------------------------------------------------------
    # take_in_memory
    # ------------------------------------------------------------------

    def take_in_memory(self, trajectory_data: TrajectoryData) -> tuple[bool, str]:
        # Success gate
        metadata = trajectory_data.metadata or {}
        is_correct = metadata.get("is_correct", False)
        task_success = metadata.get("task_success", False)
        if not (is_correct or task_success):
            return True, "Skipping ingestion: task was not successful"

        if not self.client or not self.agent_engine_name:
            return False, "Provider not initialized"

        try:
            events = self._trajectory_to_events(trajectory_data)
            scope = {self.scope_key: self.scope_value}

            result = self.client.agent_engines.memories.generate(
                name=self.agent_engine_name,
                direct_contents_source={"events": events},
                scope=scope,
                config={
                    "wait_for_completion": True,
                    "disable_consolidation": not self.enable_consolidation,
                },
            )

            # Parse response
            actions = []
            if hasattr(result, "generated_memories"):
                for gen_mem in result.generated_memories:
                    action = getattr(gen_mem, "action", "unknown")
                    name = getattr(gen_mem.memory, "name", "?") if hasattr(gen_mem, "memory") else "?"
                    actions.append(f"{action}: {name}")

            return True, f"Generated {len(actions)} memories: {'; '.join(actions)}" if actions else "Memory generation completed (no new memories)"

        except Exception as e:
            error_msg = f"Error generating memories: {e}"
            logger.error(error_msg)
            return False, error_msg

    def _trajectory_to_events(self, trajectory_data: TrajectoryData) -> list:
        """Convert TrajectoryData into Memory Bank event format."""
        events: List[Dict[str, Any]] = []

        # Original query as user event
        events.append({
            "content": {
                "role": "user",
                "parts": [{"text": trajectory_data.query}],
            }
        })

        # Each trajectory step as model event
        for step in trajectory_data.trajectory:
            text = step.get("content", str(step)) if isinstance(step, dict) else str(step)
            events.append({
                "content": {
                    "role": "model",
                    "parts": [{"text": text}],
                }
            })

        # Final result
        if trajectory_data.result:
            events.append({
                "content": {
                    "role": "model",
                    "parts": [{"text": str(trajectory_data.result)}],
                }
            })

        return events

    # ------------------------------------------------------------------
    # purge (provider-specific extension, not part of BaseMemoryProvider)
    # ------------------------------------------------------------------

    def _purge_scope_memories(self):
        """Purge all memories in the configured scope. Used for evolution resets."""
        if not self.client or not self.agent_engine_name:
            logger.warning("Cannot purge: provider not initialized")
            return

        try:
            self.client.agent_engines.memories.purge(
                name=self.agent_engine_name,
                filter=f'scope.{self.scope_key}="{self.scope_value}"',
                force=True,
                config={"wait_for_completion": True},
            )
            logger.info("Purged memories for scope %s=%s", self.scope_key, self.scope_value)
        except Exception as e:
            logger.error("Error purging memories: %s", e)
