# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MemEvolve is a meta-evolution framework for agent memory systems. It implements a dual-loop process that evolves both memory content and memory architecture for LLM-based agents. The codebase lives under `Flash-Searcher-main/`.

## Setup

```bash
conda create -n memevolve python=3.10
conda activate memevolve
cd Flash-Searcher-main
pip install -r requirements.txt
cp .env.example .env  # then fill in API keys
```

## Key Commands

All commands run from `Flash-Searcher-main/`:

```bash
# Run evaluation on datasets
python run_flash_searcher_mm_gaia.py --infile ./data/gaia/validation/metadata.jsonl --outfile ./output/results.jsonl --memory_provider agent_kb --sample_num 20 --max_steps 40
python run_flash_searcher_webwalkerqa.py --infile ./data/webwalkerqa/webwalkerqa_subset_170.jsonl --outfile ./output/results.jsonl --memory_provider lightweight_memory --sample_num 20 --max_steps 40
python run_flash_searcher_mm_xbench.py --infile ./data/xbench/DeepSearch.csv --outfile ./output/results.jsonl --memory_provider lightweight_memory --sample_num 20 --max_steps 40

# Manual evolution workflow (single system)
python evolve_cli.py analyze ./test/test_traj/lightweight_memory --work-dir ./memevolve_work --provider agent_kb
python evolve_cli.py generate --work-dir ./memevolve_work --creativity 0.5
python evolve_cli.py create --work-dir ./memevolve_work
python evolve_cli.py validate --work-dir ./memevolve_work

# Automatic multi-round evolution
python evolve_cli.py auto-evolve gaia --num-rounds 3 --provider agent_kb --num-systems 3 --task-batch-x 40 --top-t 2 --extra-sample-y 20 --creativity 0.5

# Utilities
python evolve_cli.py list       # List all memory systems
python evolve_cli.py status     # Show evolution state
python evolve_cli.py delete --memory-type SYSTEM_NAME --yes
```

## Architecture

### Three Main Subsystems

1. **FlashOAgents** (`FlashOAgents/`) — DAG-based parallel agent execution framework (adapted from Flash-Searcher). Provides `ToolCallingAgent`, step types (`ActionStep`, `PlanningStep`, etc.), and tool integrations (web search, crawl, vision, audio). `base_agent.py` wraps this into `BaseAgent` and `AnalysisAgent`.

2. **EvolveLab** (`EvolveLab/`) — Unified memory provider system. All memory providers extend `BaseMemoryProvider` (in `base_memory.py`) and implement three methods:
   - `provide_memory(request)` — retrieve memories at BEGIN/IN phases
   - `take_in_memory(trajectory_data)` — store new memories after task completion
   - `initialize()` — setup provider state

   Memory types are registered in `memory_types.py` via the `MemoryType` enum and `PROVIDER_MAPPING` dict. Provider implementations live in `EvolveLab/providers/`. Configuration for each provider is in `EvolveLab/config.py`. Storage is under `./storage/{provider_name}/`.

3. **MemEvolve** (`MemEvolve/`) — The evolution engine with four phases:
   - `PhaseAnalyzer` (phase 1) — Uses an LLM agent with trajectory-viewing tools to analyze execution logs and identify memory system bottlenecks
   - `PhaseGenerator` (phase 2) — Uses LLM to generate new memory system configurations from analysis reports; creativity index (0-1) controls temperature
   - `MemorySystemCreator` (phase 3) — Creates provider Python files, registers in `memory_types.py` enum/mapping, and updates `EvolveLab/config.py`
   - `PhaseValidator` (phase 4) — Static checks + runtime smoke tests; uses `SWEAgentValidator` (mini-swe-agent) for auto-fixing

   `MemoryEvolver` orchestrates single-system evolution. `AutoEvolver` orchestrates multi-round tournament-style evolution with checkpointing, parallel provider evaluation, and Pareto-based selection.

### Evolution Flow (AutoEvolver)

Each round: run base provider on X tasks → generate N candidate systems (each with independent analysis/generation/creation/validation) → tournament evaluation of N+1 systems → top T advance to finals on expanded task set → winner becomes next round's base. Checkpoints are saved per round for resumability.

### Adding a New Memory Provider Manually

1. Create `EvolveLab/providers/{name}_provider.py` extending `BaseMemoryProvider`
2. Add enum entry to `MemoryType` in `EvolveLab/memory_types.py` (above the marker comment)
3. Add `PROVIDER_MAPPING` entry (above the marker comment)
4. Add config entry to `DEFAULT_CONFIG["providers"]` in `EvolveLab/config.py` (above the marker comment)

### Environment Variables

Configured via `Flash-Searcher-main/.env`:
- `DEFAULT_MODEL` — fallback model for all operations
- `ANALYSIS_MODEL` / `GENERATION_MODEL` — override models for specific phases
- `OPENAI_API_KEY` / `OPENAI_BASE_URL` — LLM API access
- `SERPER_API_KEY` — web search
- `JINA_API_KEY` / `WEB_ACCESS_PROVIDER` — web crawling (alternative: `crawl4ai`)

### Key Configuration Constants

Located in `MemEvolve/config.py`: dataset paths (`DEFAULT_DATASETS`), runner scripts (`DEFAULT_RUNNERS`), evolution defaults (`EVOLVE_TASK_BATCH_X`, `EVOLVE_TOP_T`, etc.), and creativity-to-temperature mapping.
