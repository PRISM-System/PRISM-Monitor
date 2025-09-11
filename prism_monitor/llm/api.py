import requests

def llm_generate(url, prompt, max_tokens=512, temperature=0.7, presence_penalty=1.5):
    # URL 정규화 수정
    if '/v1/completions' in url:
        # 이미 완전한 URL인 경우 그대로 사용
        final_url = url
    elif url.endswith('/v1'):
        # /v1로 끝나는 경우 /completions 추가
        final_url = url + '/completions'
    elif url.endswith('/'):
        # /로 끝나는 경우 v1/completions 추가
        final_url = url + 'v1/completions'
    else:
        # 베이스 URL인 경우 /v1/completions 추가
        final_url = url + '/v1/completions'
    
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": presence_penalty,
        "stop": None,
        "stream": False,
        "model": "Qwen/Qwen3-0.6B"
    }
    
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