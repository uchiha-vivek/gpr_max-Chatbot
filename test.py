import requests

response = requests.post(
    "http://localhost:11434/api/embeddings",
    json={"model": "your-model-name", "text": "Hello World"}
)

print(response.json())  # This should return a JSON object with embeddings
