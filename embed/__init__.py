import logging
import json
import os
import openai
import azure.functions as func
import numpy as np
from textwrap import wrap

# Setup OpenAI
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = "2023-12-01-Preview"
openai.api_key = os.getenv("AZURE_OPENAI_KEY")
deployment_name = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")

# Helper to average embeddings
def average_embeddings(vectors):
    return np.mean(vectors, axis=0).tolist()

# Chunking function (~1,500 tokens = ~6,000 characters max safe size)
def chunk_text(text, max_chars=6000):
    return wrap(text, max_chars)

def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        body = req.get_json()
        values = body.get("values", [])
        results = []

        for record in values:
            record_id = record.get("recordId", "no-id")
            text = record.get("data", {}).get("text", "")

            if not text:
                results.append({
                    "recordId": record_id,
                    "errors": ["Missing 'text' field."]
                })
                continue

            try:
                client = openai.AzureOpenAI(
                    api_key=openai.api_key,
                    azure_endpoint=openai.api_base,
                    api_version=openai.api_version
                )

                chunks = chunk_text(text)
                embeddings = []
                for chunk in chunks:
                    response = client.embeddings.create(
                        input=chunk,
                        model=deployment_name
                    )
                    embeddings.append(response.data[0].embedding)

                final_embedding = average_embeddings(embeddings)

                results.append({
                    "recordId": record_id,
                    "data": {
                        "embedding": final_embedding
                    }
                })

            except Exception as inner_error:
                logging.exception("Error during embedding")
                results.append({
                    "recordId": record_id,
                    "errors": [str(inner_error)]
                })

        return func.HttpResponse(
            body=json.dumps({ "values": results }),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as outer_error:
        logging.exception("Top-level error")
        return func.HttpResponse(
            json.dumps({ "error": str(outer_error) }),
            status_code=500,
            mimetype="application/json"
        )
