import os
import json
import requests

class LLMCallManager:
    strategys = ['bimatrix_llm_agent_invoke', 'bimatrix_llm', 'openrouter']
    selected_strategy_index = 0
    bimatrix_llm_agent_invoke_url = os.environ.get('BIMATRIX_LLM_AGENT_INVOKE_URL', '')
    bimatrix_llm_url = os.environ.get('BIMATRIX_LLM_URL', '')
    openrouter_api_key = os.environ.get('OPENROUTER_API_KEY', '')
    print(f"LLMCallManager initialized with strategies: {strategys}")

    def _invoke(*args, **kwds):
        strategy = LLMCallManager.strategys[LLMCallManager.selected_strategy_index]
        if strategy == 'bimatrix_llm_agent_invoke':
            return LLMCallManager.llm_agent_invoke_bimatrix(*args, **kwds)
        elif strategy == 'bimatrix_llm':
            return LLMCallManager.llm_agent_bimatrix(*args, **kwds)
        elif strategy == 'openrouter':
            return LLMCallManager.llm_generate_openrouter(*args, **kwds)
        else:
            raise ValueError(f"Unknown LLM strategy: {strategy}")
        
    def invoke(*args, **kwds):
        for _ in range(len(LLMCallManager.strategys)):
            try:
                print(f"Trying LLM strategy: {LLMCallManager.strategys[LLMCallManager.selected_strategy_index]}")
                return LLMCallManager._invoke(*args, **kwds)
            except Exception as e:
                print(f"LLM call failed with strategy {LLMCallManager.strategys[LLMCallManager.selected_strategy_index]}: {e}")
                LLMCallManager.selected_strategy_index = (LLMCallManager.selected_strategy_index + 1) % len(LLMCallManager.strategys)
                continue
        raise ValueError("All LLM strategies failed")

    def llm_agent_invoke_bimatrix(prompt, is_json=True, **kwargs):
        # Disable thinking mode for structured JSON output
        extra_body = kwargs.get("extra_body", {})
        if "chat_template_kwargs" not in extra_body:
            extra_body["chat_template_kwargs"] = {}
        extra_body["chat_template_kwargs"]["enable_thinking"] = False

        data = {
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "temperature": kwargs.get("temperature", 0.7),
            "use_tools": kwargs.get("use_tools", False),
            "max_tool_calls": kwargs.get("max_tool_calls", 3),
            "extra_body": extra_body,
            "user_id": kwargs.get("user_id", "default_user"),
            "session_id": kwargs.get("session_id", "default_session"),
            "tool_for_use": kwargs.get("tool_for_use", []),
        }

        print(f"Calling agent invoke at URL: {LLMCallManager.bimatrix_llm_agent_invoke_url}")
        print(f"extra_body: {data.get('extra_body')}")
        response = requests.post(LLMCallManager.bimatrix_llm_agent_invoke_url, json=data, timeout=60)
        print(f"Response status: {response.status_code}")
        print(f"Response text: {response.text[:1000]}")
        try:
            content = response.json()['text']
        except Exception as e:
            print(f"Error parsing response: {e}")
            print(f"Full response: {response.text}")
            raise ValueError("Failed to parse LLM response")
        if is_json:
            if "```json\n" in content:
                content = content.replace("```json\n", "").replace("```", "")
            return json.loads(content)
        return content

    def llm_agent_bimatrix(prompt, is_json=True, **kwargs):
        # messages 형식으로 변환
        messages = [{"role": "user", "content": prompt}]

        # extra_body에 thinking 비활성화 추가
        extra_body = kwargs.get("extra_body", {})
        if "chat_template_kwargs" not in extra_body:
            extra_body["chat_template_kwargs"] = {}
        extra_body["chat_template_kwargs"]["enable_thinking"] = False

        data = {
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 1024),
            "temperature": kwargs.get("temperature", 0.7),
            "extra_body": extra_body
        }

        url = LLMCallManager.bimatrix_llm_url.rstrip('/') + '/api/generate'
        print(f"Calling LLM at URL: {url}")
        print(f"Sending messages format with thinking disabled")
        response = requests.post(url, json=data, timeout=60)
        print(f"Response status: {response.status_code}")
        print(f"Response content: {response.text[:500]}")
        try:
            content = response.json()['text']
        except Exception as e:
            print(f"Error parsing response: {e}")
            print(f"Response JSON: {response.text}")
            raise ValueError("Failed to parse LLM response")
        if is_json:
            if "```json\n" in content:
                content = content.replace("```json\n", "").replace("```", "")
            return json.loads(content)
        return content

    def llm_generate_openrouter(prompt, is_json=False, **kwargs):
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": "Bearer " + LLMCallManager.openrouter_api_key,
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": "openai/gpt-oss-20b:free",
                "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
                ],
                
            })
        )
        print('openrouter response', response.json())
        content = response.json()['choices'][0]['message']['content']
        if is_json:
            if "```json\n" in content:
                content = content.replace("```json\n", "").replace("```", "")
            return json.loads(content)
        return content