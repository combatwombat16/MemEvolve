# GoogleMemoryBankProvider — Implementation Plan

## 1. Overview

This document describes the implementation plan for integrating [Google Vertex AI Agent Engine Memory Bank](https://docs.cloud.google.com/agent-builder/agent-engine/memory-bank/overview) as a memory provider within the EvolveLab framework. Memory Bank is a managed cloud service that provides long-term memory storage with automatic fact extraction, consolidation, and embedding-based similarity search.

The provider will be named `GOOGLE_MEMORY_BANK` and will follow all existing MemEvolve conventions for registration, initialization, memory retrieval, trajectory ingestion, evolution, and storage management.

---

## 2. System Integration Points

The provider participates in the following flow, identical to all existing providers:

### 2.1 Registration & Loading

Providers are dynamically loaded via `load_memory_provider()` (defined in each runner script, e.g., `run_flash_searcher_mm_gaia.py:209-247`). The factory resolves a string name to a `MemoryType` enum, looks up `(ClassName, ModuleName)` in `PROVIDER_MAPPING` (`EvolveLab/memory_types.py:48-63`), dynamically imports the module from `EvolveLab/providers/`, fetches config via `get_memory_config()` (`EvolveLab/config.py:115`), instantiates the class, and calls `initialize()`.

Each worker thread in the runner creates its own independent provider instance for thread safety.

### 2.2 Memory Retrieval (`provide_memory`)

Called at two points in the agent execution loop (`FlashOAgents/agents.py`):

- **BEGIN phase** (line ~413): During the planning step, before any tool calls. The returned guidance is prepended to the planning prompt as a `"----Memory System Guidance----"` block.
- **IN phase** (line ~782): At the top of every action step. Guidance is injected into `memory_messages` before the LLM call.

Both calls go through `_get_memory_guidance()` (line ~597), which constructs a `MemoryRequest` with:
- `query`: the agent's current task string
- `context`: current agent context (truncated)
- `status`: `MemoryStatus.BEGIN` or `MemoryStatus.IN`
- `additional_params`: `{"step_number": N}`

### 2.3 Memory Ingestion (`take_in_memory`)

Called in the runner scripts **after** a task completes and is judged. The runner constructs a `TrajectoryData` object (e.g., `run_flash_searcher_mm_gaia.py:319-338`):

```python
TrajectoryData(
    query=original_question,
    trajectory=agent_messages,       # from write_memory_to_messages()
    result=result.get("agent_result"),
    metadata={
        "task_id": task_id,
        "status": "success",
        "is_correct": is_correct,    # boolean, set by judgment logic
        "task_success": True/False,
        "full_query": question,
    }
)
```

Existing providers check `metadata["is_correct"]` and/or `metadata["task_success"]` before ingesting — only successful trajectories should produce memories.

### 2.4 Evolution Tournament

During `auto-evolve`, `AutoEvolver` (`MemEvolve/core/auto_evolver.py`) runs the provider as a subprocess via `run_provider()` (`MemEvolve/utils/run_provider.py:23`), which invokes the dataset runner with `--memory_provider google_memory_bank`. After execution, `TrajectoryFeedbackAggregator` (`MemEvolve/utils/trajectory_tools.py:33`) computes metrics from JSON task logs:

- **accuracy**: binary (1/0 from `is_correct`/`judgement`/`score`)
- **steps**: total, action, plan, summary counts
- **tools**: total calls, unique tools, error-like calls
- **memory guidance**: count and ratio of steps with guidance
- **text**: answer/question/observation lengths
- **tokens**: total/prompt/completion token counts

Tournament selection (`_select_top()`, line ~763) ranks by accuracy descending, then total_tokens ascending. Pareto mode uses multi-objective non-dominated sorting on (accuracy, tokens, execution_time).

### 2.5 Storage Cleanup

When `--clear-storage-per-round` is enabled (default), `AutoEvolver` renames `./storage/{provider_name}/` at the start of each round for fairness (`auto_evolver.py:815-830`). The provider must tolerate its storage directory being absent and re-initialize cleanly.

For a cloud-backed provider like Memory Bank, this means:
- Local cache/state files are subject to rename/deletion
- The cloud-side Agent Engine instance persists across rounds
- The provider should support a "reset" path: either create a new Agent Engine per round, or purge memories in the existing engine's scope

---

## 3. Files to Create/Modify

### 3.1 New File: `EvolveLab/providers/google_memory_bank_provider.py`

The full provider implementation (~200-250 lines). See Section 5 for detailed class design.

### 3.2 Modify: `EvolveLab/memory_types.py`

Add enum entry above the marker comment (line 39):

```python
GOOGLE_MEMORY_BANK = "google_memory_bank"
```

Add PROVIDER_MAPPING entry above the marker comment (line 62):

```python
MemoryType.GOOGLE_MEMORY_BANK: ("GoogleMemoryBankProvider", "google_memory_bank_provider"),
```

### 3.3 Modify: `EvolveLab/config.py`

Add config block to `DEFAULT_CONFIG["providers"]` above the marker comment (line 110):

```python
MemoryType.GOOGLE_MEMORY_BANK: {
    "project_id": os.environ.get("GOOGLE_CLOUD_PROJECT", ""),
    "location": os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
    "engine_name": os.environ.get("MEMORY_BANK_ENGINE_NAME", ""),
    "scope_key": "agent_id",
    "scope_value": "memevolve_agent",
    "top_k": 3,
    "enable_consolidation": True,
    "storage_dir": os.path.join(STORAGE_BASE_DIR, "google_memory_bank"),
},
```

### 3.4 Modify: `Flash-Searcher-main/requirements.txt`

Add:

```
google-cloud-aiplatform>=1.111.0
```

### 3.5 Modify: `Flash-Searcher-main/.env.example`

Add:

```bash
# Google Memory Bank
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
MEMORY_BANK_ENGINE_NAME=               # leave blank to auto-create
```

---

## 4. Dependencies & Prerequisites

### 4.1 Google Cloud Setup

- A Google Cloud project with billing enabled
- Vertex AI API enabled (`aiplatform.googleapis.com`)
- IAM role: `roles/aiplatform.user` on the service account or user
- Authentication: Application Default Credentials (ADC) via `gcloud auth application-default login` or a service account key file

### 4.2 Python SDK

```bash
pip install google-cloud-aiplatform>=1.111.0
```

### 4.3 Network

The provider makes outbound HTTPS calls to Vertex AI endpoints. Environments without internet access (offline CI, air-gapped clusters) will need the provider to gracefully degrade or be excluded from evaluation.

---

## 5. Class Design: `GoogleMemoryBankProvider`

### 5.1 Constructor: `__init__(self, config)`

```python
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

        # Optional model for guidance synthesis (injected by runner)
        self.model = self.config.get("model", None)

        # Initialized lazily
        self.client = None
        self.agent_engine_name = None
```

### 5.2 `initialize(self) -> bool`

1. Create local storage directory (`os.makedirs(self.storage_dir, exist_ok=True)`)
2. Initialize the Vertex AI client:
   ```python
   import vertexai
   self.client = vertexai.Client(project=self.project_id, location=self.location)
   ```
3. Resolve or create the Agent Engine instance:
   - If `self.engine_name` is set: use it directly as `self.agent_engine_name`
   - Else if a persisted engine name exists at `{storage_dir}/engine_name.txt`: load it
   - Else: create a new engine via `self.client.agent_engines.create()`, persist the name to `{storage_dir}/engine_name.txt`
4. Return `True` on success, `False` with printed error on failure

**Resilience note**: If the storage directory was renamed/deleted by `AutoEvolver` storage cleanup, the persisted engine name file is lost. The provider should fall back to creating a new engine or, preferably, the `engine_name` should be set via the env var `MEMORY_BANK_ENGINE_NAME` for persistence across rounds.

### 5.3 `provide_memory(self, request: MemoryRequest) -> MemoryResponse`

#### BEGIN Phase

Use Memory Bank's similarity search to retrieve relevant facts:

```python
scope = {self.scope_key: self.scope_value}
results = list(self.client.agent_engines.memories.retrieve(
    name=self.agent_engine_name,
    scope=scope,
    similarity_search_params={
        "search_query": request.query,
        "top_k": self.top_k,
    }
))
```

Convert results to `MemoryItem` objects:

```python
for mem in results:
    distance = getattr(mem, 'distance', 0.0)
    MemoryItem(
        id=mem.name,                          # fully-qualified resource name
        content=mem.fact,                      # the stored fact string
        metadata={
            "scope": getattr(mem, 'scope', {}),
            "distance": distance,
        },
        score=1.0 / (1.0 + distance),         # Euclidean distance -> similarity
        type=MemoryItemType.TEXT,
    )
```

**Optional synthesis step**: If `self.model` is available, synthesize the retrieved facts into cohesive guidance (following the pattern in `AgentKBProvider._synthesize_all_memories`). This produces a single consolidated `MemoryItem` with human-readable guidance rather than raw facts.

Example synthesis prompt:
```
Based on the following relevant memories from past experiences, provide concise
actionable guidance for the current task.

Current Task: {request.query}

Retrieved Memories:
{numbered list of facts}

Provide 2-3 specific suggestions. Use gentle, suggestive language.
```

#### IN Phase

Two options (choose based on experimentation):
- **Option A**: Return empty `MemoryResponse` (like AgentKB, SkillWeaver)
- **Option B**: Perform a lighter similarity search using `request.context` for in-progress guidance

Start with Option A (return empty) for simplicity and parity with AgentKB.

#### Error Handling

All API calls wrapped in try/except. On failure, return an empty `MemoryResponse` and log the error. Never let a cloud API error crash the agent pipeline.

### 5.4 `take_in_memory(self, trajectory_data: TrajectoryData) -> tuple[bool, str]`

#### Step 1: Check Success

Following the established convention (see `AgentKBProvider._is_task_successful`, `SkillWeaverProvider.take_in_memory`):

```python
metadata = trajectory_data.metadata or {}
is_correct = metadata.get("is_correct", False)
task_success = metadata.get("task_success", False)
if not (is_correct or task_success):
    return True, "Skipping ingestion: task was not successful"
```

Return `(True, msg)` on skip to avoid blocking the pipeline.

#### Step 2: Convert Trajectory to Events

Transform EvolveLab's `TrajectoryData` into Memory Bank's event format:

```python
def _trajectory_to_events(self, trajectory_data: TrajectoryData) -> list:
    events = []
    # Original query as user event
    events.append({
        "content": {
            "role": "user",
            "parts": [{"text": trajectory_data.query}]
        }
    })
    # Each trajectory step as model event
    for step in trajectory_data.trajectory:
        text = step.get("content", str(step))
        events.append({
            "content": {
                "role": "model",
                "parts": [{"text": text}]
            }
        })
    # Final result
    if trajectory_data.result:
        events.append({
            "content": {
                "role": "model",
                "parts": [{"text": str(trajectory_data.result)}]
            }
        })
    return events
```

#### Step 3: Generate Memories

Use `direct_contents_source` with the converted events:

```python
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
```

**Why `generate()` over `create()`**:
- `generate()` performs automatic fact extraction from conversation data — it identifies what's worth remembering
- It consolidates with existing memories (merging, updating, or deleting contradictory facts)
- It prevents memory bloat across many trajectories
- `create()` would require us to manually extract facts first, duplicating LLM work

#### Step 4: Report Results

Parse the response to report what was created/updated/deleted:

```python
actions = []
for gen_mem in result.generated_memories:
    actions.append(f"{gen_mem.action}: {gen_mem.memory.name}")
return True, f"Generated {len(actions)} memories: {'; '.join(actions)}"
```

On error, return `(False, error_msg)`.

### 5.5 Helper: `_purge_scope_memories(self)`

For evolution fairness (storage cleanup between rounds), provide a method to purge all memories in the current scope:

```python
def _purge_scope_memories(self):
    """Purge all memories in the configured scope. Used for evolution resets."""
    self.client.agent_engines.memories.purge(
        name=self.agent_engine_name,
        filter=f'scope.{self.scope_key}="{self.scope_value}"',
        force=True,
        config={"wait_for_completion": True},
    )
```

This is not called automatically by `take_in_memory` or `provide_memory`, but may be invoked by the evolution framework if cloud-side memory reset is needed between rounds.

---

## 6. Scope Strategy

Memory Bank isolates memories by **scope** — a dict with up to 5 key-value pairs. Memories are only retrievable when the retrieval scope exactly matches the stored scope.

### Default Scope

```python
{"agent_id": "memevolve_agent"}
```

All memories share a single namespace. This is the simplest approach and matches how other providers (AgentKB, SkillWeaver) use a single shared storage.

### Dataset-Partitioned Scope (Alternative)

```python
{"agent_id": "memevolve_agent", "dataset": "gaia"}
```

Prevents cross-dataset memory pollution. Useful if running multiple datasets in sequence without wanting memories from GAIA to influence WebWalkerQA results.

### Per-Round Scope (Evolution)

```python
{"agent_id": "memevolve_agent", "round": "01"}
```

Ensures each evolution round starts fresh without needing to purge. The scope value changes per round, effectively creating isolated memory namespaces.

The scope strategy is configurable via `scope_key` / `scope_value` in the provider config.

---

## 7. Storage & State Management

### 7.1 Local Storage (`./storage/google_memory_bank/`)

- `engine_name.txt` — persisted Agent Engine resource name for reuse across runs
- No local memory database — all memory storage is cloud-managed

### 7.2 Cloud Storage (Memory Bank)

- All facts stored in Google's managed infrastructure
- Automatic embedding generation for similarity search
- TTL-based expiration (configurable at engine level)
- Memory revisions tracked automatically

### 7.3 Evolution Round Cleanup

When `AutoEvolver` renames `./storage/google_memory_bank/` between rounds:
- The `engine_name.txt` file is archived with the backup
- On re-initialization, the provider either:
  - Uses the env var `MEMORY_BANK_ENGINE_NAME` (recommended for evolution)
  - Creates a new engine (expensive, but clean)
  - Calls `_purge_scope_memories()` to clear cloud-side memories while reusing the engine

**Recommendation**: Set `MEMORY_BANK_ENGINE_NAME` in `.env` and use per-round scoping to avoid engine recreation costs.

---

## 8. Alignment with TrajectoryFeedbackAggregator

The provider does **not** affect how metrics are collected. `TrajectoryFeedbackAggregator` (`MemEvolve/utils/trajectory_tools.py:33-235`) reads JSON task logs produced by the runner scripts. These logs contain:

- `agent_trajectory`: step-by-step execution with `memory_guidance` fields
- `judgement` / `is_correct` / `score`: correctness indicators
- `metrics.total_tokens`, `metrics.elapsed_time`: resource usage

The provider's `provide_memory()` output appears in the trajectory as `memory_guidance` on planning and action steps (injected by `FlashOAgents/agents.py:_get_memory_guidance()`). The aggregator tracks `memory_guidance.count` and `memory_guidance.ratio` — these will reflect how often the Google Memory Bank provider returned non-empty guidance.

No changes to the aggregator or trajectory capture are needed.

---

## 9. Alignment with PhaseAnalyzer

During evolution, `PhaseAnalyzer` uses `TrajectoryViewerTool` and `StepViewerTool` to inspect execution logs and identify memory system bottlenecks. The tools read from JSON task logs and display:

- Memory guidance content at each step
- Whether guidance was effective (correlated with task success)
- Tool usage patterns influenced by memory

`MemoryDatabaseViewerTool` (`trajectory_tools.py:511`) allows the analyzer to inspect the provider's stored memories. For Google Memory Bank, the local storage directory will be mostly empty (just `engine_name.txt`). The analyzer should be informed (via the analysis prompt or provider metadata) that this provider stores memories in the cloud, and the `view_memory_database` tool may return minimal local data.

**Consideration**: A future enhancement could add a local JSON mirror of cloud memories for analyzer inspection, but this is not required for the initial implementation.

---

## 10. Thread Safety

Runner scripts create one provider instance per worker thread (`run_flash_searcher_mm_gaia.py:595`). Each `GoogleMemoryBankProvider` instance will hold its own `vertexai.Client` instance. The Vertex AI Python SDK is documented as thread-safe for independent client instances.

No shared mutable state between instances. The cloud-side Memory Bank handles concurrent writes via its own consistency model.

---

## 11. Error Handling Strategy

| Scenario | Behavior |
|---|---|
| Missing `GOOGLE_CLOUD_PROJECT` | `initialize()` returns `False`, provider is skipped |
| Authentication failure | `initialize()` returns `False` with error log |
| Network timeout on `retrieve()` | `provide_memory()` returns empty `MemoryResponse` |
| Network timeout on `generate()` | `take_in_memory()` returns `(False, error_msg)` |
| Engine not found (deleted/wrong region) | `initialize()` returns `False` |
| Empty retrieval results | Return empty `MemoryResponse` (normal case for cold start) |
| Rate limiting (429) | Log warning, return gracefully (same as timeout) |

Never raise exceptions that would crash the agent or runner pipeline.

---

## 12. Usage

### Direct Evaluation

```bash
cd Flash-Searcher-main

python run_flash_searcher_mm_gaia.py \
  --infile ./data/gaia/validation/metadata.jsonl \
  --outfile ./output/results.jsonl \
  --memory_provider google_memory_bank \
  --sample_num 20 \
  --max_steps 40
```

### Evolution Tournament

```bash
python evolve_cli.py auto-evolve gaia \
  --num-rounds 3 \
  --provider google_memory_bank \
  --num-systems 3 \
  --task-batch-x 40 \
  --top-t 2 \
  --extra-sample-y 20 \
  --creativity 0.5
```

### Manual Provider Inspection

```bash
python evolve_cli.py list    # Should show GOOGLE_MEMORY_BANK
```

---

## 13. Testing Plan

### 13.1 Unit Tests

- `initialize()`: mock `vertexai.Client`, verify engine creation/reuse logic
- `provide_memory()`: mock `memories.retrieve()`, verify `MemoryItem` conversion and score calculation
- `take_in_memory()`: mock `memories.generate()`, verify trajectory-to-events conversion
- Success filtering: verify that failed trajectories are skipped
- Error paths: verify graceful degradation on API errors

### 13.2 Integration Smoke Test

- Run a single task with `--memory_provider google_memory_bank --sample_num 1`
- Verify: provider initializes, `provide_memory()` returns (possibly empty on first run), `take_in_memory()` succeeds, task log contains `memory_guidance` field

### 13.3 Evolution Compatibility

- Run `evolve_cli.py auto-evolve` for 1 round with `--provider google_memory_bank --num-systems 1 --task-batch-x 5`
- Verify: provider evaluation completes, metrics are collected, storage cleanup doesn't crash

---

## 14. API Reference Summary

Key Vertex AI SDK calls used in the implementation:

| Operation | SDK Method | When Called |
|---|---|---|
| Init client | `vertexai.Client(project, location)` | `initialize()` |
| Create engine | `client.agent_engines.create()` | `initialize()` (once) |
| Retrieve memories | `client.agent_engines.memories.retrieve(name, scope, similarity_search_params)` | `provide_memory()` |
| Generate memories | `client.agent_engines.memories.generate(name, direct_contents_source, scope, config)` | `take_in_memory()` |
| Create memory (direct) | `client.agent_engines.memories.create(name, fact, scope)` | Alternative ingestion path |
| Purge memories | `client.agent_engines.memories.purge(name, filter, force, config)` | Evolution reset |

---

## 15. Key Design Decisions

| Decision | Rationale |
|---|---|
| Use `generate()` over `create()` for ingestion | Automatic fact extraction + consolidation prevents memory bloat and handles deduplication |
| Use `direct_contents_source` over Sessions | EvolveLab trajectories are post-hoc data, not live conversations |
| Cloud-only storage (no local embedding index) | Offloads similarity search compute to Google's infrastructure; reduces local dependencies |
| BEGIN-only retrieval (IN phase returns empty) | Matches AgentKB pattern; avoids excessive API calls during step execution |
| Configurable scope strategy | Supports shared, dataset-partitioned, and per-round isolation modes |
| Engine name persisted to env var + local file | Avoids recreating engines (which are billable resources) across runs |

---

## 16. Open Questions

1. **Cost**: Memory Bank pricing per generate/retrieve call at scale (thousands of trajectories across evolution rounds). May need a local caching layer to reduce API calls during evolution tournaments.

2. **Latency**: `generate()` with `wait_for_completion=True` may add significant latency to `take_in_memory()`. Consider `wait_for_completion=False` (async) for evolution runs where immediate memory availability isn't critical.

3. **Cold start**: On the first task of a fresh evaluation, the memory bank is empty and `provide_memory()` returns nothing. This is consistent with other providers but worth noting for benchmark interpretation.

4. **Memory topics**: Memory Bank supports managed topics (USER_PREFERENCES, KEY_CONVERSATION_DETAILS, etc.) and custom topics. Agent task trajectories don't map cleanly to user-preference categories. Custom topics like `"task_strategy"` and `"tool_usage_pattern"` could improve extraction quality — worth experimenting with after the initial implementation.

5. **Analyzer visibility**: Since memories live in the cloud, the PhaseAnalyzer's `view_memory_database` tool won't show stored memories. Consider adding a local JSON export of cloud memories for analysis compatibility.
