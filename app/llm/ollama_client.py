import os
os.environ["OLLAMA_HOST"] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
import ollama

def generate_response(prompt: str) -> str:
    response = ollama.chat(
        model="llama3",  # используйте вашу модель
        messages=[{"role": "user", "content": prompt}],
        stream=False
    )
    return response['message']['content']