from openai import OpenAI

client = OpenAI(
    api_key="sk-1234", # pass litellm proxy key, if you're using virtual keys
    base_url="http://0.0.0.0:4000/v1/" # point to litellm proxy
)

response = client.chat.completions.create(
    model="gemini-2.5-pro-preview-03-25",
    messages=[{"role": "user", "content": "Who won the world cup?"}]
)

print(response)