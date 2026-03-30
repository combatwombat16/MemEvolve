# GoogleMemoryBankProvider — Implementation Task List

> Derived from [GoogleMemoryBankProvider.md](file:///home/pandamancer/repo/MemEvolve/GoogleMemoryBankProvider.md).
> Reviewed through architecture-patterns, ai-agents-architect, and data-scientist lenses.

---

## Phase 0 — Prerequisites & Environment

- [ ] **0.1** Enable Vertex AI API (`aiplatform.googleapis.com`) on target GCP project
- [ ] **0.2** Confirm IAM role `roles/aiplatform.user` on service account or ADC identity
- [x] **0.3** Add `google-cloud-aiplatform>=1.111.0` to `Flash-Searcher-main/requirements.txt`
- [x] **0.4** Add Memory Bank env vars to `Flash-Searcher-main/.env.example`
  - `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, `MEMORY_BANK_ENGINE_NAME`
- [ ] **0.5** Install dependency locally: `pip install google-cloud-aiplatform>=1.111.0`

> **⚠️ Gap:** 0.1, 0.2, 0.5 are manual GCP setup tasks that cannot be done in code. They are prerequisites before running Phases 5–6.

---

## Phase 1 — Framework Registration

Register the provider so the dynamic loader can discover it.

- [x] **1.1** Add `GOOGLE_MEMORY_BANK = "google_memory_bank"` to `MemoryType` enum in [memory_types.py](file:///home/pandamancer/repo/MemEvolve/Flash-Searcher-main/EvolveLab/memory_types.py) (above the `# add new memory type upside this line(Enum)` marker)
- [x] **1.2** Add `PROVIDER_MAPPING` entry:

  ```python
  MemoryType.GOOGLE_MEMORY_BANK: ("GoogleMemoryBankProvider", "google_memory_bank_provider"),
  ```

  (above the `# add new memory type upside this line(PROVIDER_MAPPING)` marker)
- [x] **1.3** Add config block to `DEFAULT_CONFIG["providers"]` in [config.py](file:///home/pandamancer/repo/MemEvolve/Flash-Searcher-main/EvolveLab/config.py) (above the `# add new memory type upside this line` marker)
  - Keys: `project_id`, `location`, `engine_name`, `scope_key`, `scope_value`, `top_k`, `enable_consolidation`, `storage_dir`
- [ ] **1.4** Verify: run `python evolve_cli.py list` — `GOOGLE_MEMORY_BANK` should appear

> **⚠️ Gap:** 1.4 requires a full environment with all dependencies installed (Phase 0.5). Cannot be verified in the lightweight poetry dev env.

> **🏗 Architecture review:** Registration follows the established dynamic-loader pattern. The provider mapping is the only coupling between the core framework and the provider module — clean seam.

---

## Phase 2 — Provider Core Implementation

Create `EvolveLab/providers/google_memory_bank_provider.py` (~200-250 lines).

### 2.1 Class Skeleton & Constructor

- [x] **2.1.1** Create file; import `BaseMemoryProvider`, memory types, `vertexai`, `os`, `logging`
- [x] **2.1.2** Implement `__init__(self, config)`: extract config keys `project_id`, `location`, `engine_name`, `scope_key`, `scope_value`, `top_k`, `enable_consolidation`, `storage_dir`, `model`; set `self.client = None`, `self.agent_engine_name = None`

> **🤖 Agent architecture note:** Each worker thread gets its own provider instance (`run_flash_searcher_mm_gaia.py:595`). No shared mutable state. `vertexai.Client` is safe for independent instances.

### 2.2 `initialize(self) → bool`

- [x] **2.2.1** Create local `storage_dir` (`os.makedirs`, `exist_ok=True`)
- [x] **2.2.2** Instantiate `vertexai.Client(project=..., location=...)`
- [x] **2.2.3** Engine resolution logic:
  1. Use `self.engine_name` if set (env var path — recommended for evolution)
  2. Else load from `{storage_dir}/engine_name.txt` if file exists
  3. Else create via `client.agent_engines.create()` and persist name to file
- [x] **2.2.4** Wrap in try/except; return `False` + log on failure, `True` on success
- [x] **2.2.5** Handle missing-directory resilience (storage cleanup between evolution rounds)

> **🧪 Data science note:** The "engine resolution" step introduces a non-deterministic branch (create-vs-reuse). Log which path is taken so we can trace provenance of memory state in experiment analysis.

### 2.3 `provide_memory(self, request: MemoryRequest) → MemoryResponse`

- [x] **2.3.1** Implement BEGIN-phase retrieval via `client.agent_engines.memories.retrieve()` with `similarity_search_params`
- [x] **2.3.2** Convert results to `MemoryItem` objects:
  - `id` = `mem.name` (fully-qualified resource name)
  - `content` = `mem.fact`
  - `score` = `1.0 / (1.0 + distance)` (Euclidean → similarity transform)
  - `metadata` includes `scope` and `distance`
- [x] **2.3.3** Optional: implement synthesis step if `self.model` is available (condense facts into coherent guidance, following `AgentKBProvider._synthesize_all_memories` pattern)
- [x] **2.3.4** IN-phase: return empty `MemoryResponse` (matches AgentKB/SkillWeaver pattern)
- [x] **2.3.5** Error handling: wrap all API calls in try/except → return empty `MemoryResponse` on failure

> **🤖 Agent architecture note (memory hoarding anti-pattern):** `top_k` defaults to 3, consistent with other providers. The `1/(1+d)` score transform gives a bounded [0,1] similarity that the framework already expects. Returning empty on IN-phase avoids excessive API call volume per step.

> **🧪 Data science note:** The `1/(1+d)` transform is monotonically decreasing and bounded — good for ranking. However, Euclidean distance in embedding space may not be linearly comparable across providers. Document this transform choice so cross-provider score comparisons are not made naïvely. The `memory_guidance.count` and `.ratio` metrics from `TrajectoryFeedbackAggregator` will automatically capture retrieval frequency.

### 2.4 `take_in_memory(self, trajectory_data: TrajectoryData) → tuple[bool, str]`

- [x] **2.4.1** Success gate: check `metadata["is_correct"]` or `metadata["task_success"]`; skip on failure following `AgentKBProvider`/`SkillWeaverProvider` convention
- [x] **2.4.2** Implement `_trajectory_to_events()`: convert `TrajectoryData` → Memory Bank event format
  - User event from `trajectory_data.query`
  - Model events from each trajectory step (`step.get("content", str(step))`)
  - Final result as closing model event
- [x] **2.4.3** Call `client.agent_engines.memories.generate()` with `direct_contents_source` and scope
- [x] **2.4.4** Parse response: report created/updated/deleted memories
- [x] **2.4.5** Error handling: return `(False, error_msg)` on API failure; never raise

> **🤖 Agent architecture note (trajectory data fidelity):** Trajectory steps in EvolveLab are `List[Dict[str, Any]]`. Steps contain `content`, `role`, `tool_calls`, etc. The converter must handle the varying shapes: some steps have a top-level `content` string, others have nested structures. Use `str(step)` as fallback to avoid data loss while still feeding the generate() API valid text. This is a "graceful degradation" pattern.

> **🧪 Data science note:** Using `generate()` over `create()` is the correct choice: automatic fact extraction and consolidation prevent memory bloat across hundreds of trajectories. This is essential for experiment reproducibility — manual fact extraction would introduce uncontrolled variance. The `wait_for_completion=True` flag ensures deterministic memory state before the next task begins. Consider adding a timing metric for `generate()` latency to track API cost over evolution rounds.

### 2.5 `_purge_scope_memories(self)`

- [x] **2.5.1** Implement memory purge for evolution resets using `client.agent_engines.memories.purge()`
- [x] **2.5.2** Filter by scope: `f'scope.{scope_key}="{scope_value}"'`
- [x] **2.5.3** Use `force=True`, `wait_for_completion=True`

> **🏗 Architecture review:** The purge method is not part of the `BaseMemoryProvider` interface — it's a provider-specific extension. This is correct; it should be called explicitly by evolution control logic, not by the standard memory pipeline.

---

## Phase 3 — Scope Strategy & Evolution Integration

- [x] **3.1** Implement default scope: `{scope_key: scope_value}` → `{"agent_id": "memevolve_agent"}`
- [x] **3.2** Validate scope is applied consistently in `retrieve()`, `generate()`, and `purge()`
- [x] **3.3** Document scope configuration options in provider docstring:
  - Shared (default), dataset-partitioned, per-round isolation
- [x] **3.4** Ensure `initialize()` tolerates absent `storage_dir` (created fresh after `AutoEvolver` rename)
- [x] **3.5** Confirm env var `MEMORY_BANK_ENGINE_NAME` overrides file-based persistence for evolution stability

> **🏗 Architecture review:** The scope isolation strategy is the key differentiator for evolution fairness. Per-round scoping (`{"agent_id": "...", "round": "01"}`) avoids the need for purge calls entirely. The config-driven scope makes this testable without cloud side-effects.

> **🧪 Data science note:** For rigorous A/B comparison against other providers, per-round scoping is critical. Memory carry-over across rounds would confound accuracy metrics. Ensure that `--clear-storage-per-round` interacts correctly with the cloud-backed memory: local storage cleanup + scope rotation = clean experimental isolation.

---

## Phase 4 — Unit Tests

Create `tests/test_google_memory_bank_provider.py` (or suitable location).

All 19 tests passing (`poetry run pytest Flash-Searcher-main/tests/test_google_memory_bank_provider.py`).

- [x] **4.1** `test_initialize_success`: mock `vertexai.Client`, verify engine creation/reuse
- [x] **4.2** `test_initialize_with_env_engine_name`: mock client, verify env var path taken
- [x] **4.3** `test_initialize_with_persisted_engine_name`: mock file read, verify reuse path
- [x] **4.4** `test_initialize_failure_missing_project`: verify returns `False`
- [x] **4.5** `test_initialize_failure_auth`: mock auth error, verify returns `False`
- [x] **4.6** `test_provide_memory_begin_phase`: mock `memories.retrieve()`, verify `MemoryItem` conversion, score calculation (`1/(1+d)`)
- [x] **4.7** `test_provide_memory_in_phase_returns_empty`: verify empty `MemoryResponse`
- [x] **4.8** `test_provide_memory_api_error`: mock retrieve exception, verify graceful empty response
- [x] **4.9** `test_provide_memory_empty_results`: verify cold-start returns empty `MemoryResponse`
- [x] **4.10** `test_take_in_memory_success_trajectory`: mock `memories.generate()`, verify event conversion and report
- [x] **4.11** `test_take_in_memory_skips_failed_task`: pass `metadata={"is_correct": False}`, verify skip message
- [x] **4.12** `test_take_in_memory_api_error`: mock generate exception, verify returns `(False, error_msg)`
- [x] **4.13** `test_trajectory_to_events_shape`: validate event list structure matches Memory Bank API schema
- [x] **4.14** `test_trajectory_to_events_missing_content`: verify fallback `str(step)` for malformed steps
- [x] **4.15** `test_purge_scope_memories`: mock `memories.purge()`, verify filter string and flags

> **🧪 Data science note:** Tests 4.6 and 4.13 are the most scientifically important — they validate the data transform contract. If `MemoryItem` scores or event shapes drift, downstream metrics (`memory_guidance.count`, accuracy correlations) become meaningless. Pin expected values, don't just assert non-null.

> **🤖 Agent architecture note:** Mock at the `vertexai.Client` boundary, not deeper. This keeps tests stable against SDK internal changes while verifying our data contract.

---

## Phase 5 — Integration Smoke Test

> **Status: NOT STARTED.** Requires GCP project setup (Phase 0.1–0.2) and full dependency install (Phase 0.5).

- [ ] **5.1** Run single-task evaluation:

  ```bash
  cd Flash-Searcher-main
  python run_flash_searcher_mm_gaia.py \
    --infile ./data/gaia/validation/metadata.jsonl \
    --outfile ./output/gmb_smoke_results.jsonl \
    --memory_provider google_memory_bank \
    --sample_num 1 --max_steps 10
  ```

- [ ] **5.2** Verify: provider initializes without error
- [ ] **5.3** Verify: `provide_memory()` returns (possibly empty on first run; non-empty on subsequent)
- [ ] **5.4** Verify: `take_in_memory()` succeeds on correct task
- [ ] **5.5** Verify: task log JSON contains `memory_guidance` field in trajectory steps
- [ ] **5.6** Verify: `TrajectoryFeedbackAggregator` produces valid metrics from the task log

---

## Phase 6 — Evolution Compatibility Test

> **Status: NOT STARTED.** Requires Phase 5 to pass first.

- [ ] **6.1** Run minimal evolution:

  ```bash
  python evolve_cli.py auto-evolve gaia \
    --num-rounds 1 --provider google_memory_bank \
    --num-systems 1 --task-batch-x 5 --top-t 1
  ```

- [ ] **6.2** Verify: evaluation completes, metrics collected
- [ ] **6.3** Verify: `--clear-storage-per-round` renames `./storage/google_memory_bank/` without crash
- [ ] **6.4** Verify: provider re-initializes cleanly after storage cleanup
- [ ] **6.5** Verify: `PhaseAnalyzer` tools can inspect the provider's task logs
- [ ] **6.6** Confirm: `MemoryDatabaseViewerTool` returns minimal local data (expected — memories are cloud-side)

---

## Phase 7 — Documentation & Polish

- [x] **7.1** Add usage examples to provider module docstring (direct eval + evolution commands)
- [x] **7.2** Add inline comments on scope strategy options
- [x] **7.3** Log provider initialization path (`engine_name` source: env/file/created)
- [ ] **7.4** Add note for `PhaseAnalyzer` that this provider's memory DB is cloud-hosted
- [ ] **7.5** Update project README if applicable

> **⚠️ Gap — 7.4:** The `PhaseAnalyzer` analysis prompt (in `MemEvolve/core/`) has not been updated to inform the analyzer that this provider's memories live in the cloud and `view_memory_database` will return minimal local data. This is a minor documentation gap that would improve analysis quality.

> **⚠️ Gap — 7.5:** Project README has not been updated. Low priority — deferrable.

---

## Open Items (Post-MVP)

These are recorded for future consideration and do not block initial implementation.

- [ ] **O.1** Cost analysis: benchmark `generate()`/`retrieve()` call volume at scale
- [ ] **O.2** Latency experiment: compare `wait_for_completion=True` vs `False` for evolution throughput
- [ ] **O.3** Custom memory topics: experiment with `"task_strategy"`, `"tool_usage_pattern"` for extraction quality
- [ ] **O.4** Local JSON mirror: export cloud memories for `MemoryDatabaseViewerTool` compatibility
- [ ] **O.5** IN-phase retrieval (Option B): lighter context-based search during action steps
- [ ] **O.6** Add `generate()` latency timing metric to task logs for API cost tracking

---

## Cross-Cutting Concerns

| Concern | Status | Notes |
|---|---|---|
| Thread safety | ✅ Addressed in §2.1 | One client per instance; no shared state |
| Error isolation | ✅ Addressed in §2.3, §2.4 | All API calls wrapped; never crash pipeline |
| Evolution fairness | ✅ Addressed in §3 | Per-round scope or purge; storage cleanup tolerant |
| Metric alignment | ✅ No changes needed | `TrajectoryFeedbackAggregator` reads standard JSON logs |
| Analyzer visibility | ⚠️ Partial | Cloud memories not visible to `MemoryDatabaseViewerTool`; see O.4 |
| Data contract fidelity | ✅ Addressed in §4 | Tests pin `MemoryItem` shape and score transforms |

---

## Audit Summary

### Completed: 38 of 52 tasks

| Phase | Done | Total | Status |
|---|---|---|---|
| Phase 0 — Prerequisites | 2 | 5 | Partial (3 manual GCP tasks remain) |
| Phase 1 — Registration | 3 | 4 | Partial (1.4 live verify blocked on env) |
| Phase 2 — Core Implementation | 16 | 16 | **Complete** |
| Phase 3 — Scope & Evolution | 5 | 5 | **Complete** |
| Phase 4 — Unit Tests | 15 | 15 | **Complete** (19 tests, all passing) |
| Phase 5 — Integration Smoke | 0 | 6 | Not started (blocked on Phase 0) |
| Phase 6 — Evolution Compat | 0 | 6 | Not started (blocked on Phase 5) |
| Phase 7 — Documentation | 3 | 5 | Partial (7.4, 7.5 remain) |

### Gaps & Inconsistencies Found

1. **`take_in_memory` return on skip** — The task list (§2.4.1) says to return `True` on skip per AgentKB convention. The implementation returns `(True, "Skipping ingestion: task was not successful")`. However, `AgentKBProvider._is_task_successful` returns `False` and the caller returns `(False, msg)`. This is a **minor inconsistency** with AgentKB but arguably the *better* behavior — returning `True` on skip avoids logging it as a failure in the pipeline. The spec doc (§5.4) explicitly says `Return (True, msg) on skip`. ✅ Implementation matches spec.

2. **`_synthesize_guidance` not unit-tested** — The `_synthesize_guidance()` method (provider line 175–208) is implemented but has **no dedicated test**. It's exercised only when `self.model` is set. The task list doesn't explicitly call for a synthesis test, but the data science note on 2.3.3 flags synthesis as important. Consider adding a test that sets `config["model"]` to a mock callable.

3. **PhaseAnalyzer cloud-hosted note (7.4)** — Not done. The analyzer may waste time trying to inspect an empty local memory database. This is the most impactful remaining doc gap.

4. **No `conftest.py`** — The test file uses a complex module-loading preamble to avoid heavy provider imports. A shared `conftest.py` would clean this up if more provider tests are added later. Not a gap per the task list, but worth noting.

5. **`take_in_memory` return value on empty actions** — Line 247 of the provider has a subtle precedence issue: `return True, f"..." if actions else "Memory generation completed..."`. Due to Python operator precedence, this evaluates as `return True, (f"..." if actions else "...")` which is correct. No bug, but the ternary could be clearer with parentheses.
