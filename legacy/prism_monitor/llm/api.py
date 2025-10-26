import os
import requests
import json

def llm_generate(url, prompt, model='/root/models/openai/gpt-oss-120b', max_tokens=512, temperature=0.7):
    messages = [
        {"role": "user", "content": prompt}
    ]
    data = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    res = requests.post(url, json=data)
    return res.json()

    
    try:
        response = requests.post(final_url, json=payload, timeout=60)
        
        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code}")
        
        result = response.json()
        
        if "choices" in result and len(result["choices"]) > 0:
            return {
                "text": result["choices"][0]["text"],
                "usage": result.get("usage", {}),
                "choices": result["choices"]
            }
        else:
            return result
            
    except Exception as e:
        raise Exception(f"LLM request failed: {str(e)}")
    
def temp_llm_call(prompt):
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": "Bearer "+os.environ['OPENROUTER_API_KEY'],
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