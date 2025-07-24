# init_client, LLM_CFG
import os
from typing import Dict
from easonsi.llm.openai_client import OpenAIClient
from easonsi.llm.ollama_client import OllamaClient

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed

LLM_CFG = {}

def add_openai_models():
    global LLM_CFG
    model_list = [
        "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4",
    ]
    for model in model_list:
        assert model not in LLM_CFG, f"{model} already in LLM_CFG"
        LLM_CFG[model] = {
            "model_name": model,
            "base_url": os.getenv("OPENAI_BASE_URL"),
            "api_key": os.getenv("OPENAI_API_KEY"),
            "client_type": "openai"
        }

def add_ollama_models():
    global LLM_CFG
    # Common Ollama models
    model_list = [
        "gemma3:latest", "gemma3:2b", "gemma3:9b", "gemma3:27b",
        "gemma2:latest", "gemma2:9b", "gemma2:27b",
        "llama3.2:latest", "llama3.2:3b", "llama3.2:1b",
        "llama3.1:latest", "llama3.1:8b", "llama3.1:70b",
        "qwen2.5:latest", "qwen2.5:7b", "qwen2.5:14b",
        "mistral:latest", "mistral:7b",
        "codellama:latest", "codellama:7b", "codellama:13b",
    ]
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    for model in model_list:
        if model not in LLM_CFG:  # Avoid conflicts with OpenAI models
            LLM_CFG[model] = {
                "model_name": model,
                "base_url": ollama_base_url,
                "api_key": None,
                "client_type": "ollama"
            }

add_openai_models()
add_ollama_models()


def init_client(llm_cfg: Dict):
    client_type = llm_cfg.get("client_type", "openai")
    
    if client_type == "ollama":
        client = OllamaClient(
            model_name=llm_cfg["model_name"], 
            base_url=llm_cfg["base_url"]
        )
    else:  # Default to OpenAI
        client = OpenAIClient(
            model_name=llm_cfg["model_name"], 
            base_url=llm_cfg["base_url"], 
            api_key=llm_cfg["api_key"]
        )
    return client

