import requests

class OllamaClient:
        def __init__(self, base_url: str = "http://localhost:11434"):
                self.base_url = base_url
        
        def generate(self, prompt: str, model: str = "qwen3:1.7b") -> str:     
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model,
                    "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
                    "stream": False,
                },
            )  
            response.raise_for_status()
            data = response.json()

            return data["message"]["content"]  