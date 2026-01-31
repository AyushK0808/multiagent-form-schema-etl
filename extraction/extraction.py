from transformers import pipeline
import json
import re

# We use TinyLlama because it is "ungated" (no login required) 
# and runs much faster on local CPUs than Llama-2.
llm = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
    temperature=0.1,
    max_new_tokens=256,
    device_map="auto" # requires 'accelerate', or remove this line to force CPU
)

def clean_json_string(text):
    """Extracts JSON object from a string containing other text."""
    try:
        # Look for the first opening brace and the last closing brace
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end != -1:
            json_str = text[start:end]
            return json.loads(json_str)
    except Exception:
        pass
    return None

def extract_field_value(field_name, text):
    # TinyLlama Chat uses a specific format: <|system|>, <|user|>, <|assistant|>
    prompt = f"""<|system|>
You are an API that outputs ONLY valid JSON.
Extract the value for "{field_name}" from the text.
If not found, return null.
</s>
<|user|>
Text:
\"\"\"{text}\"\"\"

Output format:
{{ "{field_name}": "<extracted_value>" }}
</s>
<|assistant|>
"""
    
    outputs = llm(prompt)
    response_text = outputs[0]["generated_text"]
    
    # Debug: uncomment to see what the LLM is actually generating
    # print(f"DEBUG LLM output: {response_text}")

    # Remove the prompt from the response to find the JSON
    if "<|assistant|>" in response_text:
        response_text = response_text.split("<|assistant|>")[-1]

    data = clean_json_string(response_text)
    
    if data and field_name in data:
        return data[field_name]
    return None