import ray
from ray import serve
from starlette.requests import Request
from typing import Dict
from transformers import pipeline
import uvicorn

ray.init(address="auto")  # Connect to Ray or start locally
serve.start()

# 2: Define the Sentiment Analysis Deployment
@serve.deployment(route_prefix="/")
class SentimentAnalysisDeployment:
    def __init__(self):
        self._model = pipeline("sentiment-analysis")

    async def __call__(self, request: Request) -> Dict:
        # Extract text from JSON payload
        data = await request.json()
        text = data.get("text", "")
        if not text:
            return {"error": "No text provided."}
        result = self._model(text)[0]
        return result

# 3: Deploy the Sentiment Analysis Service
serve.run(SentimentAnalysisDeployment.bind(), route_prefix="/")

print("Sentiment analysis service is running at http://localhost:8000/")

# 4: Keep the service alive
if __name__ == "__main__":
    try:
        while True:
            pass  # Keeps the process alive indefinitely
    except KeyboardInterrupt:
        print("Shutting down service...")
        ray.shutdown()
