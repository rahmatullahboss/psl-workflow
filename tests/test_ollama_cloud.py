from psl_workflow.generation.engine import OllamaDraftLLM


def test_ollama_cloud_endpoint_uses_single_api_path() -> None:
    llm = OllamaDraftLLM(base_url="https://ollama.com/api")

    assert llm._endpoint("generate") == "https://ollama.com/api/generate"


def test_ollama_host_endpoint_adds_api_path() -> None:
    llm = OllamaDraftLLM(base_url="https://ollama.com")

    assert llm._endpoint("generate") == "https://ollama.com/api/generate"
