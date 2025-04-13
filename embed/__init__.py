import logging
import json
import os
import openai
import azure.functions as func

openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = "2023-12-01-preview"
openai.api_key = os.getenv("AZURE_OPENAI_KEY")
deployment_name = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")

def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        body = req.get_json()
        values = body.get("values", [])
        results = []

        for record in values:
            record_id = record.get("recordId")
            text = record.get("data", {}).get("text", "")

            if not text:
                results.append({
                    "recordId": record_id,
                    "errors": ["Missing 'text' field."]
                })
                continue

            response = openai.AzureOpenAI().embeddings.create(
                input=text,
                model=deployment_name
            )
            embedding = response.data[0].embedding

            results.append({
                "recordId": record_id,
                "data": {
                    "embedding": embedding
                }
            })

        return func.HttpResponse(
            body=json.dumps({ "values": results }),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        logging.exception("Embedding function failed")
        return func.HttpResponse(
            json.dumps({ "error": str(e) }),
            status_code=500,
            mimetype="application/json"
        )