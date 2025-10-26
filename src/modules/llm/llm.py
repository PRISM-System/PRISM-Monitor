import json
import requests

def llm_generate_bimatrix(bimatrix_llm_url, prompt, model='/root/models/openai/gpt-oss-120b', is_json=True, **kwargs):
    messages = [
        {"role": "user", "content": prompt}
    ]
    data = {
        "model": model,
        "messages": messages,
        "max_tokens": kwargs.get("max_tokens", 1024),
        "temperature": kwargs.get("temperature", 0.7),
    }

    response = requests.post(bimatrix_llm_url, json=data, timeout=60)
    try:
        content = response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error parsing response: {e}")
        print(response.json())
        raise ValueError("Failed to parse LLM response")
    if not is_json:
        return str(content)
    return json.loads(content)

def llm_generate_openrouter(prompt, openrouter_api_key):
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": "Bearer "+openrouter_api_key,
            "Content-Type": "application/json",
        },
        data=json.dumps({
            "model": "qwen/qwen3-14b:free",
            "messages": [
            {
                "role": "user",
                "content": prompt
            }
            ],
            
        })
    )
    return json.loads(response.json()['choices'][0]['message']['content'])