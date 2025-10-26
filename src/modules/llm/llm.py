import json
import requests

def llm_generate_bimatrix(bimatrix_llm_url, prompt, model='/root/models/openai/gpt-oss-120b', **kwargs):
    messages = [
        {"role": "user", "content": prompt}
    ]
    data = {
        "model": model,
        "messages": messages,
        "max_tokens": kwargs.get("max_tokens", 512),
        "temperature": kwargs.get("temperature", 0.7),
    }

    response = requests.post(bimatrix_llm_url, json=data, timeout=60)
    return response.json()['choices'][0]['message']['content']

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