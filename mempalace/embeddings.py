"""
embeddings.py — Pluggable embedding function for ChromaDB.

Controls which model is used to embed text chunks at mine-time and query-time.
The SAME provider must be used for both; mixing providers produces garbage results.

IMPORTANT — provider or model change requires a full re-ingest:
  Vectors from different models are incompatible (different dimensions/spaces).
  Delete the palace and re-mine when switching:
    rm -rf ~/.mempalace/   # or wherever palace_path points

──────────────────────────────────────────────────────────────────
Provider options  (MEMPALACE_EMBED_PROVIDER env var)
──────────────────────────────────────────────────────────────────

  local  (default)
    ChromaDB's built-in all-MiniLM-L6-v2 via onnxruntime.
    Runs in-process on CPU — no server, no key — but inference is
    single-threaded and saturates the CPU for large codebases.

  ollama  ← recommended for local/NPU use
    Offloads to a running Ollama server. Ollama on Windows uses
    DirectML and will accelerate on an NPU or iGPU automatically.
    Significantly faster than `local` for bulk ingestion.

    Setup:
      1. Install Ollama (https://ollama.com)
      2. ollama pull nomic-embed-text   # or any embedding model below
      3. export MEMPALACE_EMBED_PROVIDER=ollama
      4. (optional) export MEMPALACE_EMBED_MODEL=nomic-embed-text

    Recommended models (trade off size vs quality):
      nomic-embed-text      768 dims  ~274 MB  great all-round (default)
      nomic-embed-text:v1.5 768 dims  ~274 MB  slightly better retrieval
      mxbai-embed-large     1024 dims ~670 MB  highest quality
      bge-small-en-v1.5     384 dims  ~67 MB   fastest, still solid

    Config vars:
      MEMPALACE_EMBED_PROVIDER=ollama
      MEMPALACE_EMBED_MODEL=nomic-embed-text       (default)
      MEMPALACE_OLLAMA_URL=http://localhost:11434   (default)

  lmstudio
    LM Studio exposes an OpenAI-compatible API at localhost:1234.
    Load any GGUF or ONNX embedding model in LM Studio; ONNX models
    with DirectML are the fastest path for an NPU.
    No real API key required — a dummy value is used.

    Setup:
      1. Load an embedding model in LM Studio (e.g. nomic-embed-text)
      2. Start the local server in LM Studio
      3. export MEMPALACE_EMBED_PROVIDER=lmstudio

    Config vars:
      MEMPALACE_EMBED_PROVIDER=lmstudio
      MEMPALACE_EMBED_MODEL=nomic-embed-text        (must match loaded model)
      MEMPALACE_LMSTUDIO_URL=http://localhost:1234  (default)

  openai
    OpenAI Embeddings API. Requires OPENAI_API_KEY.
    Fast and cheap (~$0.02/1M tokens) but needs a key and external access.

    Config vars:
      MEMPALACE_EMBED_PROVIDER=openai
      OPENAI_API_KEY=sk-...   (or MEMPALACE_OPENAI_API_KEY)
      MEMPALACE_EMBED_MODEL=text-embedding-3-small  (default)

  azure-ai
    Azure AI Foundry/Azure OpenAI Embeddings. Uses managed embedding models
    from your Azure subscription. No local VRAM concerns. Good for corporate
    environments with Azure infrastructure already in place.

    Setup:
      1. Install Azure SDK: pip install azure-ai-inference azure-identity azure-mgmt-cognitiveservices
      2. Log in via CLI: az login
      3. Set default subscription: az account set --subscription <id>
      4. export MEMPALACE_EMBED_PROVIDER=azure-ai

    Requires Azure CLI login with default subscription set:
      az login
      az account set --subscription <your-subscription-id>

    Config vars:
      MEMPALACE_EMBED_PROVIDER=azure-ai
      MEMPALACE_EMBED_MODEL=text-embedding-3-small  (default; or text-embedding-3-large)
      MEMPALACE_AZURE_ENDPOINT=https://<resource>.openai.azure.com  (optional; auto-discovered if omitted)
"""

import os


def _first_env(*names: str) -> str | None:
    """Return the first non-empty environment variable value from the given names."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def get_embedding_function():
    """
    Return a ChromaDB-compatible embedding function, or None for the default.

    None → ChromaDB uses all-MiniLM-L6-v2 via onnxruntime (CPU-heavy, in-process).
    """
    provider = os.environ.get("MEMPALACE_EMBED_PROVIDER", "azure-ai").lower()

    # ── Ollama ────────────────────────────────────────────────────────────────
    if provider == "ollama":
        from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

        base_url = os.environ.get("MEMPALACE_OLLAMA_URL", "http://localhost:11434")
        model = os.environ.get("MEMPALACE_EMBED_MODEL", "nomic-embed-text")
        # ChromaDB's OllamaEmbeddingFunction expects the /api/embeddings endpoint
        url = base_url.rstrip("/") + "/api/embeddings"
        return OllamaEmbeddingFunction(url=url, model_name=model)

    # ── LM Studio / OpenAI (openai v2 compatible) ───────────────────────────
    if provider in ("lmstudio", "openai"):
        return _openai_compat_ef(provider)

    # ── Azure AI Foundry / Azure OpenAI ───────────────────────────────────
    if provider == "azure-ai":
        return _azure_ai_ef()

    # ── local (default) ───────────────────────────────────────────────────────
    # Returns None → ChromaDB uses all-MiniLM-L6-v2 via onnxruntime in-process.
    # Single-threaded CPU inference; slow for large ingests.
    return None


def ef_kwargs():
    """Return a dict suitable for **-splatting into get_collection / create_collection."""
    ef = get_embedding_function()
    return {"embedding_function": ef} if ef is not None else {}


# ── Custom OpenAI-v2-compatible embedding function ────────────────────────────

class _OpenAICompatEF:
    """
    ChromaDB EmbeddingFunction that uses the openai>=2.0 client directly.

    ChromaDB 0.6.x's built-in OpenAIEmbeddingFunction only detects openai v1
    (version.startswith("1.")) — with openai v2 it falls back to the removed
    openai.Embedding API.  This class avoids that by calling the v2 client.
    """

    def __init__(self, *, api_key: str, model: str, base_url: str | None = None):
        import openai

        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = openai.OpenAI(**kwargs)
        self._model = model

    def __call__(self, input):
        input = [t.replace("\n", " ") for t in input]
        resp = self._client.embeddings.create(input=input, model=self._model)
        sorted_data = sorted(resp.data, key=lambda e: e.index)
        return [e.embedding for e in sorted_data]


def _openai_compat_ef(provider: str):
    """Build an _OpenAICompatEF for lmstudio or openai providers."""
    if provider == "lmstudio":
        base_url = os.environ.get("MEMPALACE_LMSTUDIO_URL", "http://localhost:1234")
        model = os.environ.get("MEMPALACE_EMBED_MODEL", "nomic-embed-text")
        api_base = base_url.rstrip("/") + "/v1"
        return _OpenAICompatEF(api_key="lm-studio", model=model, base_url=api_base)

    # openai
    api_key = (
        os.environ.get("MEMPALACE_OPENAI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        raise RuntimeError(
            "MEMPALACE_EMBED_PROVIDER=openai requires OPENAI_API_KEY "
            "or MEMPALACE_OPENAI_API_KEY to be set."
        )
    model = os.environ.get("MEMPALACE_EMBED_MODEL", "text-embedding-3-small")
    return _OpenAICompatEF(api_key=api_key, model=model)


# ── Azure AI Foundry / Azure OpenAI embedding function ────────────────────

class _AzureAIEF:
    """
    ChromaDB EmbeddingFunction using Azure AI embeddings (via Azure SDK).

    Uses an API key when one is available, otherwise falls back to Azure CLI /
    DefaultAzureCredential. Automatically discovers the endpoint if not explicitly
    provided.
    """

    def __init__(self, model: str, endpoint: str | None = None):
        from azure.core.credentials import AzureKeyCredential
        from azure.identity import DefaultAzureCredential
        from azure.identity import get_bearer_token_provider
        from openai import AzureOpenAI

        self._model = model
        self._openai_client = None
        self._client = None

        # Try explicit endpoint, then provider-specific envs, then auto-discover.
        endpoint = endpoint or _first_env(
            "MEMPALACE_AZURE_ENDPOINT",
            "AZURE_AIFOUNDRY_URI",
            "AZURE_AIFOUNDRY_URI_AJACOBM",
            "AZURE_OPENAI_ENDPOINT",
        )
        api_key = _first_env(
            "MEMPALACE_AZURE_API_KEY",
            "AZURE_AIFOUNDRY_API_KEY",
            "AZURE_AIFOUNDRY_API_KEY_AJACOBM",
            "AZURE_OPENAI_API_KEY",
        )
        use_aad = endpoint and ".openai.azure.com" in endpoint

        if not endpoint:
            # Auto-discover: find first deployment of the model in default subscription
            try:
                from azure.mgmt.cognitiveservices import CognitiveServicesManagementClient
                from azure.identity import DefaultAzureCredential
                from azure.core.exceptions import HttpResponseError

                cred = DefaultAzureCredential()
                # Get subscription from CLI context
                import subprocess

                sub = subprocess.check_output(
                    ["az", "account", "show", "--query", "id", "-o", "tsv"],
                    text=True,
                ).strip()

                # List cognitive accounts with a search for embedding models
                client = CognitiveServicesManagementClient(cred, sub)
                for account in client.accounts.list():
                    if "openai" in account.kind.lower():
                        endpoint = f"https://{account.name}.openai.azure.com"
                        break

                if not endpoint:
                    raise RuntimeError(
                        "Could not auto-discover Azure OpenAI endpoint. "
                        "Set MEMPALACE_AZURE_ENDPOINT explicitly."
                    )
            except Exception as e:
                raise RuntimeError(
                    f"Azure endpoint discovery failed: {e}. "
                    "Set MEMPALACE_AZURE_ENDPOINT=https://<resource>.openai.azure.com"
                )

        if use_aad:
            token_provider = get_bearer_token_provider(
                DefaultAzureCredential(),
                "https://cognitiveservices.azure.com/.default",
            )
            self._openai_client = AzureOpenAI(
                azure_endpoint=endpoint.rstrip("/"),
                azure_ad_token_provider=token_provider,
                api_version="2024-10-21",
            )
        elif api_key:
            from azure.ai.inference import EmbeddingsClient

            cred = AzureKeyCredential(api_key)
            self._client = EmbeddingsClient(endpoint=endpoint, credential=cred)
        else:
            from azure.ai.inference import EmbeddingsClient

            # Use DefaultAzureCredential (respects az login)
            cred = DefaultAzureCredential()
            self._client = EmbeddingsClient(endpoint=endpoint, credential=cred)

    def __call__(self, input):
        input = [t.replace("\n", " ") for t in input]
        if self._openai_client is not None:
            resp = self._openai_client.embeddings.create(model=self._model, input=input)
            return [item.embedding for item in resp.data]

        resp = self._client.embed(input=input, model=self._model)
        sorted_data = sorted(resp.data, key=lambda e: e.index)
        return [e.embedding for e in sorted_data]


def _azure_ai_ef():
    """Build an _AzureAIEF for Azure AI provider."""
    model = os.environ.get("MEMPALACE_EMBED_MODEL", "text-embedding-3-small")
    endpoint = _first_env(
        "MEMPALACE_AZURE_ENDPOINT",
        "AZURE_AIFOUNDRY_URI",
        "AZURE_AIFOUNDRY_URI_AJACOBM",
        "AZURE_OPENAI_ENDPOINT",
    ) or "https://ajacobm-5877-resource.openai.azure.com"
    return _AzureAIEF(model=model, endpoint=endpoint)
