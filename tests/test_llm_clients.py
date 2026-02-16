from typing import Any, cast

import models.llm_clients.client as client_module
import pytest
from models.llm_clients import LLMClientManager, LLMProvider, LMStudioClientAdapter


class FakeChat:
    """Fake chat object for adapter tests."""

    def __init__(self, prompt: str) -> None:
        self.prompt = prompt
        self.user_messages: list[str] = []

    def add_user_message(self, message: str) -> None:
        """Append user message to fake chat history."""
        self.user_messages.append(message)


class FakeModel:
    """Fake model that captures calls and supports configurable behavior."""

    def __init__(self) -> None:
        self.respond_calls: list[tuple[FakeChat, dict[str, Any]]] = []
        self.unload_calls = 0
        self.raise_on_respond = False

    def respond(self, chat: FakeChat, response_format: dict[str, Any]):
        """Return parsed response or raise configured runtime error."""
        self.respond_calls.append((chat, response_format))
        if self.raise_on_respond:
            raise RuntimeError("respond failed")
        return type("FakeResponse", (), {"parsed": {"ok": True}})

    def unload(self) -> None:
        """Track unload calls for assertion."""
        self.unload_calls += 1


class FakeLLM:
    """Fake LLM accessor that provides model by id."""

    def __init__(self) -> None:
        self.models_by_id: dict[str, FakeModel] = {}

    def model(self, model_id: str) -> FakeModel:
        """Get or create fake model instance for id."""
        if model_id not in self.models_by_id:
            self.models_by_id[model_id] = FakeModel()
        return self.models_by_id[model_id]


class FakeClient:
    """Fake LM Studio client for adapter tests."""

    def __init__(self, api_host: str) -> None:
        self.api_host = api_host
        self.llm = FakeLLM()
        self.close_calls = 0

    def close(self) -> None:
        """Track close calls for assertion."""
        self.close_calls += 1


@pytest.fixture
def patched_lmstudio(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch LM Studio SDK objects with fakes."""
    monkeypatch.setattr(client_module, "LMStudioClient", FakeClient)
    monkeypatch.setattr(client_module, "LMStudioChat", FakeChat)


@pytest.fixture
def adapter(patched_lmstudio: None) -> LMStudioClientAdapter:
    """Create LMStudio adapter with fake SDK dependencies."""
    return LMStudioClientAdapter(api_host="localhost:1234", system_prompt="System prompt")


class TestLLMClientManager:
    """Unit tests for LLM client manager behavior."""

    def test_get_client_returns_lmstudio_adapter(self, patched_lmstudio: None) -> None:
        """Return LMStudio adapter for supported provider."""
        manager = LLMClientManager()

        client = manager.get_client(LLMProvider.LMSTUDIO, api_host="localhost:1234", system_prompt="Prompt")

        assert isinstance(client, LMStudioClientAdapter)
        assert client.api_host == "localhost:1234"
        assert client.system_prompt == "Prompt"

    def test_get_client_raises_for_unsupported_provider(self, patched_lmstudio: None) -> None:
        """Raise ValueError when provider is not registered."""
        manager = LLMClientManager()

        with pytest.raises(ValueError, match="not supported"):
            manager.get_client("unknown", api_host="localhost:1234")  # type: ignore[arg-type]


class TestLMStudioClientAdapter:
    """Unit tests for LMStudio adapter behavior."""

    @pytest.mark.parametrize(
        "api_host",
        [
            "localhost:1234",
            "127.0.0.1:80",
            "api.example.com:65535",
        ],
    )
    def test_init_accepts_valid_api_host(self, patched_lmstudio: None, api_host: str) -> None:
        """Create adapter when api_host has valid host:port format."""
        adapter = LMStudioClientAdapter(api_host=api_host)

        assert adapter.api_host == api_host

    @pytest.mark.parametrize(
        "api_host",
        [
            "http://localhost:1234",
            "localhost",
            "localhost:0",
            "localhost:65536",
            "localhost:abc",
            "localhost:1234/path",
        ],
    )
    def test_init_rejects_invalid_api_host(self, patched_lmstudio: None, api_host: str) -> None:
        """Raise ValueError for invalid api_host format or range."""
        with pytest.raises(ValueError):
            LMStudioClientAdapter(api_host=api_host)

    def test_get_client_returns_same_long_lived_instance(self, adapter: LMStudioClientAdapter) -> None:
        """Yield the same underlying client across multiple sessions."""
        with adapter.get_client() as first:
            first_client = first
        with adapter.get_client() as second:
            second_client = second

        assert first_client is second_client

    def test_close_closes_underlying_client(self, adapter: LMStudioClientAdapter) -> None:
        """Close long-lived client explicitly via adapter API."""
        fake_client = adapter.client

        adapter.close()

        assert isinstance(fake_client, FakeClient)
        assert fake_client.close_calls == 1

    def test_generate_response_builds_chat_and_returns_parsed_payload(self, adapter: LMStudioClientAdapter) -> None:
        """Send raw data to chat and return parsed model response."""
        schema = {"type": "object"}
        with adapter.get_client() as client:
            fake_client = cast(FakeClient, client)
            result = adapter.generate_response(
                client=client,
                model_id="model-a",
                json_schema=schema,
                raw_data="input data",
            )
            model = fake_client.llm.models_by_id["model-a"]

        assert result == {"ok": True}
        assert len(model.respond_calls) == 1
        chat, response_format = model.respond_calls[0]
        assert chat.prompt == "System prompt"
        assert chat.user_messages == ["input data"]
        assert response_format == schema
        assert model.unload_calls == 1

    def test_generate_response_unloads_model_on_exception(self, adapter: LMStudioClientAdapter) -> None:
        """Unload model even when respond raises runtime error."""
        schema = {"type": "object"}
        with adapter.get_client() as client:
            fake_client = cast(FakeClient, client)
            model = fake_client.llm.model("model-error")
            model.raise_on_respond = True

            with pytest.raises(RuntimeError, match="respond failed"):
                adapter.generate_response(
                    client=client,
                    model_id="model-error",
                    json_schema=schema,
                    raw_data="input data",
                )

        assert model.unload_calls == 1

    def test_run_test_returns_result_for_each_model_case_pair(self, adapter: LMStudioClientAdapter) -> None:
        """Generate one result for each model-case combination."""
        cases = [
            client_module.LLMTestCase(name="case-1", raw_data="raw-1", json_schema={"type": "object"}),
            client_module.LLMTestCase(name="case-2", raw_data="raw-2", json_schema={"type": "object"}),
        ]
        model_ids = ["model-a", "model-b"]

        results = adapter.run_test(model_ids=model_ids, cases=cases)

        assert len(results) == 4
        assert {(r.model_id, r.case_name) for r in results} == {
            ("model-a", "case-1"),
            ("model-a", "case-2"),
            ("model-b", "case-1"),
            ("model-b", "case-2"),
        }

    def test_run_test_does_not_close_long_lived_client(self, adapter: LMStudioClientAdapter) -> None:
        """Keep client open after run_test for explicit lifecycle control."""
        cases = [client_module.LLMTestCase(name="case-1", raw_data="raw-1", json_schema={"type": "object"})]

        adapter.run_test(model_ids=["model-a"], cases=cases)

        assert isinstance(adapter.client, FakeClient)
        assert adapter.client.close_calls == 0
