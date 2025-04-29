import requests

def call_ollama(messages, model, temperature=0.7, max_tokens=800, top_p=0.95, frequency_penalty=0, presence_penalty=0):
    # Use the OpenAI-compatible endpoint for Ollama.
    url = "http://localhost:11434/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()
