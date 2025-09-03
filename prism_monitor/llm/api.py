import requests

# def llm_generate(url, prompt, max_tokens=1024, temperature=0.7, presence_penalty=1.5):
#     payload = {
#         "prompt": prompt,
#         "max_tokens": max_tokens,
#         "temperature": temperature,
#         "presence_penalty": presence_penalty
#     }
#     response = requests.post(url, json=payload).json()
#     return response


def llm_generate(url, prompt, max_tokens=512, temperature=0.7, presence_penalty=1.5):
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "presence_penalty": presence_penalty,
        "use_tools": False,
        "extra_body": {
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        }
    }
    response = requests.post(url, json=payload).json()
    return response
