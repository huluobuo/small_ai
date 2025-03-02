from openai import OpenAI

client = OpenAI(api_key="sk************************e", base_url="https://api.deepseek.com")
response = client.chat.completions.create(
    model="deepseek-chat",  # 或 "deepseek-reasoner" 调用 R1 模型
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你好，DeepSeek！"}
    ],
    stream=False  # 设置为 True 可启用流式输出
)
print(response.choices[0].message.content)