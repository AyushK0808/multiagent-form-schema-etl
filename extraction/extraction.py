from transformers import pipeline
import json

llm = pipeline(
    "text-generation",
    model="meta-llama/Llama-2-7b-chat-hf",
    temperature=0.0,
    max_new_tokens=256
)

def extract_field_value(field_name, text):
    prompt = f"""
You are filling a form field.

Field name: {field_name}

Document text:
\"\"\"{text}\"\"\"

Return ONLY valid JSON:
{{ "{field_name}": "<value or null>" }}
"""
    response = llm(prompt)[0]["generated_text"]

    try:
        return json.loads(response.strip())[field_name]
    except Exception:
        return None
