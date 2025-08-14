from openai import OpenAI

def get_chat_response(user_message, api_url="http://localhost:8001/v1", api_key='EMPTY', model="Qwen/Qwen3-0.6B"):
    client = OpenAI(
        api_key=api_key,
        base_url=api_url,
    )
    chat_response = client.chat.completions.create(
        model="Qwen/Qwen3-0.6B",
        messages=[
            {"role": "user", "content": user_message},
        ],
        max_tokens=8192,
        temperature=0.7,
        top_p=0.8,
        presence_penalty=1.5,
        extra_body={
            "top_k": 20, 
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    return chat_response.choices[0].message.content