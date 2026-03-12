"""
Unit tests for GoogleMemoryBankProvider.

All Vertex AI SDK calls are mocked at the vertexai.Client boundary.
The providers __init__.py is pre-registered as an empty module to avoid
importing heavy ML dependencies (numpy, sklearn, etc.) that other
providers require.
"""

import importlib
import importlib.util
import os
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is importable
_project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _project_root)

# Stub vertexai so the provider module can be imported without the SDK
sys.modules.setdefault("vertexai", MagicMock())

# Import the lightweight EvolveLab modules first
from EvolveLab.memory_types import MemoryRequest, MemoryResponse, MemoryStatus, TrajectoryData  # noqa: E402
from EvolveLab.base_memory import BaseMemoryProvider  # noqa: E402, F401

# Pre-register EvolveLab.providers as an empty package so our provider's
# relative import (from ..base_memory) doesn't trigger __init__.py
if "EvolveLab.providers" not in sys.modules:
    _pkg = types.ModuleType("EvolveLab.providers")
    _pkg.__path__ = [os.path.join(_project_root, "EvolveLab", "providers")]
    _pkg.__package__ = "EvolveLab.providers"
    sys.modules["EvolveLab.providers"] = _pkg

# Now load our provider module directly
_provider_path = os.path.join(
    _project_root, "EvolveLab", "providers", "google_memory_bank_provider.py"
)
_spec = importlib.util.spec_from_file_location(
    "EvolveLab.providers.google_memory_bank_provider", _provider_path,
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
GoogleMemoryBankProvider = _mod.GoogleMemoryBankProvider


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_storage(tmp_path):
    return str(tmp_path / "google_memory_bank")


@pytest.fixture
def base_config(tmp_storage):
    return {
        "project_id": "test-project",
        "location": "us-central1",
        "engine_name": "projects/test-project/locations/us-central1/agentEngines/123",
        "scope_key": "agent_id",
        "scope_value": "test_agent",
        "top_k": 3,
        "enable_consolidation": True,
        "storage_dir": tmp_storage,
    }


@pytest.fixture
def mock_client():
    client = MagicMock()
    return client


def _make_provider(config, mock_client_instance):
    """Helper: create provider and mock initialize.

    Since vertexai is imported inside initialize() via `import vertexai`,
    we patch it in sys.modules so the deferred import picks up our mock.
    """
    provider = GoogleMemoryBankProvider(config)
    mock_vx = MagicMock()
    mock_vx.Client.return_value = mock_client_instance
    with patch.dict(sys.modules, {"vertexai": mock_vx}):
        result = provider.initialize()
    return provider, result


# ---------------------------------------------------------------------------
# 4.1 test_initialize_success
# ---------------------------------------------------------------------------

class TestInitialize:

    def test_initialize_success(self, base_config, mock_client):
        provider, ok = _make_provider(base_config, mock_client)
        assert ok is True
        assert provider.agent_engine_name == base_config["engine_name"]
        assert os.path.isdir(base_config["storage_dir"])

    def test_initialize_with_env_engine_name(self, base_config, mock_client):
        """When engine_name is set in config, use it directly (no file or create)."""
        provider, ok = _make_provider(base_config, mock_client)
        assert ok is True
        assert provider.agent_engine_name == base_config["engine_name"]
        # Should NOT call agent_engines.create
        mock_client.agent_engines.create.assert_not_called()

    def test_initialize_with_persisted_engine_name(self, base_config, mock_client):
        """When engine_name is empty but file exists, load from file."""
        config = {**base_config, "engine_name": ""}
        os.makedirs(config["storage_dir"], exist_ok=True)
        persisted_name = "projects/test/locations/us-central1/agentEngines/456"
        with open(os.path.join(config["storage_dir"], "engine_name.txt"), "w") as f:
            f.write(persisted_name)

        provider, ok = _make_provider(config, mock_client)
        assert ok is True
        assert provider.agent_engine_name == persisted_name
        mock_client.agent_engines.create.assert_not_called()

    def test_initialize_creates_new_engine(self, base_config, mock_client):
        """When engine_name is empty and no file, create a new engine."""
        config = {**base_config, "engine_name": ""}
        new_engine = SimpleNamespace(name="projects/test/locations/us-central1/agentEngines/789")
        mock_client.agent_engines.create.return_value = new_engine

        provider, ok = _make_provider(config, mock_client)
        assert ok is True
        assert provider.agent_engine_name == new_engine.name
        mock_client.agent_engines.create.assert_called_once()
        # Verify persisted to file
        engine_file = os.path.join(config["storage_dir"], "engine_name.txt")
        assert os.path.isfile(engine_file)
        with open(engine_file) as f:
            assert f.read().strip() == new_engine.name

    def test_initialize_failure_missing_project(self, base_config):
        config = {**base_config, "project_id": ""}
        provider = GoogleMemoryBankProvider(config)
        assert provider.initialize() is False

    def test_initialize_failure_auth(self, base_config):
        config = {**base_config, "engine_name": ""}
        provider = GoogleMemoryBankProvider(config)
        mock_vx = MagicMock()
        mock_vx.Client.side_effect = Exception("Auth error")
        with patch.dict(sys.modules, {"vertexai": mock_vx}):
            assert provider.initialize() is False


# ---------------------------------------------------------------------------
# 4.6–4.9 test_provide_memory
# ---------------------------------------------------------------------------

class TestProvideMemory:

    def _make_initialized_provider(self, base_config, mock_client):
        provider, _ = _make_provider(base_config, mock_client)
        return provider

    def test_provide_memory_begin_phase(self, base_config, mock_client):
        """Verify MemoryItem conversion and score calculation."""
        mem1 = SimpleNamespace(name="mem/1", fact="Tool X works best for PDFs", distance=0.5, scope={"agent_id": "test"})
        mem2 = SimpleNamespace(name="mem/2", fact="Always verify dates", distance=1.0, scope={"agent_id": "test"})
        mock_client.agent_engines.memories.retrieve.return_value = [mem1, mem2]

        provider = self._make_initialized_provider(base_config, mock_client)
        request = MemoryRequest(query="test query", context="", status=MemoryStatus.BEGIN)

        response = provider.provide_memory(request)

        assert isinstance(response, MemoryResponse)
        assert response.total_count == 2
        assert len(response.memories) == 2

        # Verify score: 1/(1+d)
        assert response.memories[0].score == pytest.approx(1.0 / 1.5)  # d=0.5
        assert response.memories[1].score == pytest.approx(1.0 / 2.0)  # d=1.0
        assert response.memories[0].content == "Tool X works best for PDFs"
        assert response.memories[0].id == "mem/1"
        assert response.memories[0].metadata["distance"] == 0.5

    def test_provide_memory_in_phase_returns_empty(self, base_config, mock_client):
        provider = self._make_initialized_provider(base_config, mock_client)
        request = MemoryRequest(query="test", context="", status=MemoryStatus.IN)

        response = provider.provide_memory(request)

        assert response.total_count == 0
        assert response.memories == []
        mock_client.agent_engines.memories.retrieve.assert_not_called()

    def test_provide_memory_api_error(self, base_config, mock_client):
        mock_client.agent_engines.memories.retrieve.side_effect = Exception("API timeout")

        provider = self._make_initialized_provider(base_config, mock_client)
        request = MemoryRequest(query="test", context="", status=MemoryStatus.BEGIN)

        response = provider.provide_memory(request)

        assert response.total_count == 0
        assert response.memories == []

    def test_provide_memory_empty_results(self, base_config, mock_client):
        """Cold start: no memories yet."""
        mock_client.agent_engines.memories.retrieve.return_value = []

        provider = self._make_initialized_provider(base_config, mock_client)
        request = MemoryRequest(query="first query ever", context="", status=MemoryStatus.BEGIN)

        response = provider.provide_memory(request)

        assert response.total_count == 0
        assert response.memories == []

    def test_provide_memory_not_initialized(self, base_config):
        """Provider that was never initialized returns empty."""
        provider = GoogleMemoryBankProvider(base_config)
        request = MemoryRequest(query="test", context="", status=MemoryStatus.BEGIN)

        response = provider.provide_memory(request)

        assert response.total_count == 0


# ---------------------------------------------------------------------------
# 4.10–4.14 test_take_in_memory
# ---------------------------------------------------------------------------

class TestTakeInMemory:

    def _make_initialized_provider(self, base_config, mock_client):
        provider, _ = _make_provider(base_config, mock_client)
        return provider

    def test_take_in_memory_success_trajectory(self, base_config, mock_client):
        gen_mem = SimpleNamespace(
            action="CREATE",
            memory=SimpleNamespace(name="mem/new1"),
        )
        mock_client.agent_engines.memories.generate.return_value = SimpleNamespace(
            generated_memories=[gen_mem]
        )

        provider = self._make_initialized_provider(base_config, mock_client)
        traj = TrajectoryData(
            query="What is X?",
            trajectory=[
                {"content": "Searching for X..."},
                {"content": "Found X = 42"},
            ],
            result="42",
            metadata={"is_correct": True, "task_success": True},
        )

        ok, msg = provider.take_in_memory(traj)

        assert ok is True
        assert "Generated 1 memories" in msg
        mock_client.agent_engines.memories.generate.assert_called_once()
        call_kwargs = mock_client.agent_engines.memories.generate.call_args
        events = call_kwargs.kwargs.get("direct_contents_source", call_kwargs[1].get("direct_contents_source", {})).get("events", [])
        # 1 user + 2 model steps + 1 result = 4 events
        assert len(events) == 4

    def test_take_in_memory_skips_failed_task(self, base_config, mock_client):
        provider = self._make_initialized_provider(base_config, mock_client)
        traj = TrajectoryData(
            query="What is X?",
            trajectory=[{"content": "step"}],
            result="wrong",
            metadata={"is_correct": False, "task_success": False},
        )

        ok, msg = provider.take_in_memory(traj)

        assert ok is True
        assert "Skipping" in msg
        mock_client.agent_engines.memories.generate.assert_not_called()

    def test_take_in_memory_api_error(self, base_config, mock_client):
        mock_client.agent_engines.memories.generate.side_effect = Exception("Quota exceeded")

        provider = self._make_initialized_provider(base_config, mock_client)
        traj = TrajectoryData(
            query="test",
            trajectory=[{"content": "step"}],
            result="result",
            metadata={"is_correct": True},
        )

        ok, msg = provider.take_in_memory(traj)

        assert ok is False
        assert "Quota exceeded" in msg


# ---------------------------------------------------------------------------
# 4.13–4.14 test_trajectory_to_events
# ---------------------------------------------------------------------------

class TestTrajectoryToEvents:

    def test_trajectory_to_events_shape(self, base_config, mock_client):
        """Validate event list structure matches Memory Bank API schema."""
        provider, _ = _make_provider(base_config, mock_client)
        traj = TrajectoryData(
            query="What is 2+2?",
            trajectory=[
                {"content": "Calculating...", "role": "assistant"},
                {"content": "The answer is 4"},
            ],
            result="4",
        )

        events = provider._trajectory_to_events(traj)

        assert len(events) == 4  # 1 user + 2 steps + 1 result
        # First event is user
        assert events[0]["content"]["role"] == "user"
        assert events[0]["content"]["parts"][0]["text"] == "What is 2+2?"
        # Middle events are model
        assert events[1]["content"]["role"] == "model"
        assert events[1]["content"]["parts"][0]["text"] == "Calculating..."
        # Last event is result
        assert events[3]["content"]["role"] == "model"
        assert events[3]["content"]["parts"][0]["text"] == "4"

    def test_trajectory_to_events_missing_content(self, base_config, mock_client):
        """Verify fallback str(step) for malformed steps."""
        provider, _ = _make_provider(base_config, mock_client)
        malformed_step = {"tool_calls": [{"name": "search", "args": {"q": "test"}}]}
        traj = TrajectoryData(
            query="test",
            trajectory=[malformed_step],
            result=None,
        )

        events = provider._trajectory_to_events(traj)

        assert len(events) == 2  # 1 user + 1 step (no result since None)
        # Fallback: str(step) used since no "content" key
        step_text = events[1]["content"]["parts"][0]["text"]
        assert "tool_calls" in step_text
        assert "search" in step_text

    def test_trajectory_to_events_no_result(self, base_config, mock_client):
        """No result event when trajectory_data.result is None."""
        provider, _ = _make_provider(base_config, mock_client)
        traj = TrajectoryData(query="q", trajectory=[], result=None)

        events = provider._trajectory_to_events(traj)

        assert len(events) == 1  # Only the user event


# ---------------------------------------------------------------------------
# 4.15 test_purge_scope_memories
# ---------------------------------------------------------------------------

class TestPurgeScopeMemories:

    def test_purge_scope_memories(self, base_config, mock_client):
        provider, _ = _make_provider(base_config, mock_client)
        provider._purge_scope_memories()

        mock_client.agent_engines.memories.purge.assert_called_once()
        call_kwargs = mock_client.agent_engines.memories.purge.call_args
        # Verify filter string and flags
        assert 'scope.agent_id="test_agent"' in str(call_kwargs)
        assert call_kwargs.kwargs.get("force", call_kwargs[1].get("force")) is True

    def test_purge_not_initialized(self, base_config):
        """Purge on uninitialized provider does not raise."""
        provider = GoogleMemoryBankProvider(base_config)
        provider._purge_scope_memories()  # Should not raise
