from .client import LLMClientAdapter, LLMProvider, LMStudioClientAdapter


class LLMClientManager:
    def __init__(self) -> None:
        self.clients: dict[LLMProvider, type[LLMClientAdapter]] = {
            LLMProvider.LMSTUDIO: LMStudioClientAdapter,
        }

    def get_client(self, provider: LLMProvider, api_host: str, system_prompt: str = "") -> LLMClientAdapter:
        if provider not in self.clients:
            raise ValueError(f"LLM provider '{provider}' is not supported.")
        return self.clients[provider](api_host=api_host, system_prompt=system_prompt)
