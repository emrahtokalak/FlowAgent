"""
Ollama client for local LLM models
"""

import sys, os, json, re, time, traceback, requests
from typing import Dict, Union, Tuple, List

class OllamaClient:
    """
    Ollama client that mimics OpenAI API interface
    """
    model_name: str = "gemma3:latest"
    temperature: float = 0.7
    max_tokens: int = 4096
    base_url: str = "http://localhost:11434"
    use_cache: bool = False
    retries: int = 3
    backoff_factor: float = 0.5
    n_thread: int = 5
    
    def __init__(
        self, model_name: str = None, temperature: float = None, max_tokens: int = None,
        base_url: str = "http://localhost:11434", api_key=None, print_url=False, 
    ):
        if print_url:
            print(f"[INFO] Ollama base_url: {base_url}")
        self.base_url = base_url
        if model_name: 
            self.model_name = model_name
        if temperature: 
            self.temperature = temperature
        if max_tokens: 
            self.max_tokens = max_tokens

    def _make_request(self, messages: List[Dict], **kwargs) -> Dict:
        """Make request to Ollama API"""
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens)
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            
            # Create OpenAI-compatible response
            class MockChoice:
                def __init__(self, content):
                    self.message = type('obj', (object,), {'content': content})
            
            class MockCompletion:
                def __init__(self, content):
                    self.choices = [MockChoice(content)]
            
            content = result.get("message", {}).get("content", "")
            return MockCompletion(content)
            
        except Exception as e:
            print(f"[ERROR] Ollama request failed: {str(e)}")
            raise Exception(f"Ollama request failed: {str(e)}")

    def query_one_raw(self, text, **args):
        """Query with single text input"""
        messages = [{"role": "user", "content": text}]
        return self._make_request(messages, **args)

    def query_one(self, text, **args) -> str:
        """Query and return text response"""
        try:
            response = self.query_one_raw(text, **args)
            return response.choices[0].message.content
        except Exception as e:
            print(f"[ERROR] Ollama query failed: {e}")
            return f"Error: {str(e)}"

    def query_conversation_raw(self, messages: List[Dict], **args):
        """Query with conversation history"""
        return self._make_request(messages, **args)

    def query_conversation(self, messages: List[Dict], **args) -> str:
        """Query conversation and return text response"""
        try:
            response = self.query_conversation_raw(messages, **args)
            return response.choices[0].message.content
        except Exception as e:
            print(f"[ERROR] Ollama conversation query failed: {e}")
            return f"Error: {str(e)}"

    def check_connection(self) -> bool:
        """Check if Ollama server is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    def list_models(self) -> List[str]:
        """List available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
            return []
        except:
            return []
