# AEGIS Model Abstraction & Registry — Current Status

Ollama is the only inference backend. The configured aliases are:
`qwen-general → qwen3.5:9b`, `qwen-coder → qwen2.5-coder:7b`,
`qwen-vision → qwen3-vl:8b`, and `llama-small → llama3.2:1b`.
Availability is checked locally; unavailable models produce controlled
structured failures. Model identifiers stay in configuration/registry rather
than Master planning logic.

---

## 1. Provider Abstraction (`models/base.py`)

The application isolates all model interactions behind the `ModelProvider` abstract base class:

```python
class ModelProvider(ABC):
    @property
    def name(self) -> str: ...
    @property
    def model_id(self) -> str: ...
    
    @abstractmethod
    def get_chat_model(self, **kwargs) -> BaseChatModel: ...
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> ModelResponse: ...
    
    @abstractmethod
    async def chat(self, messages: list[dict[str, str]], **kwargs) -> ModelResponse: ...
    
    @abstractmethod
    async def health_check(self) -> bool: ...
```

This guarantees that:
1. Business logic, LangGraph nodes, and tools never couple directly to Ollama REST APIs.
2. Future backends (e.g. `vLLMProvider`, `LlamaCppProvider`, or high-throughput batch inference engines) can be added by implementing this single interface.

---

## 2. Ollama Implementation (`models/ollama.py`)

`OllamaProvider` wraps the official `langchain-ollama` library (`ChatOllama`):
* **Cached Model Instances:** Chat model instances are cached by model name and temperature.
* **Context Capacity:** Automatically configures `num_ctx` based on `ModelConfig.context_length`.
* **Health Check & Tag Matching:** Performs asynchronous health checks against `http://localhost:11434/api/tags` with prefix and tag fuzzy-matching.

---

## 3. Configuration & Registry (`models/registry.py`)

The `ModelRegistry` class loads configuration from `config/models.yaml`:

```yaml
models:
  qwen-general:
    provider: ollama
    model: "qwen3:8b"
    capabilities:
      - general
      - reasoning
      - summarization
      - document_analysis
    context_length: 32768
    priority: 10

  qwen-coder:
    provider: ollama
    model: "qwen2.5-coder:7b"
    capabilities:
      - coding
      - debugging
      - calculation
    context_length: 32768
    priority: 10

  llama-small:
    provider: ollama
    model: "llama3.2:1b"
    capabilities:
      - general
      - lightweight
    context_length: 8192
    priority: 1
    is_fallback: true
```

### Registry Operations
* `get_provider(name: str)`: Resolves an instantiated provider by registry key.
* `find_by_capability(*capabilities)`: Filters providers supporting all specified capabilities.
* `get_fallback()`: Retrieves the designated lightweight fallback model when no primary model matches or is online.
* `check_availability()`: Asynchronously pings all registered models and returns an availability map.
