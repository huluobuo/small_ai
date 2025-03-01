import openai
import tkinter as tk
from tkinter import scrolledtext

# 设置OpenAI API密钥
openai.api_key = 'sk-29b0439160364175be2098afd713e5af'

# 定义聊天机器人函数
def chat_with_bot():
    user_input = user_input_box.get("1.0", tk.END).strip()
    if user_input:
        response = openai.Completion.create(
            engine="deepseek-r1",  # 使用text-davinci-003模型
            prompt=user_input,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7
        )
        bot_response = response.choices[0].text.strip()
        chat_history.config(state=tk.NORMAL)
        chat_history.insert(tk.END, f"You: {user_input}\n")
        chat_history.insert(tk.END, f"Bot: {bot_response}\n\n")
        chat_history.config(state=tk.DISABLED)
        user_input_box.delete("1.0", tk.END)

# 创建主窗口
root = tk.Tk()
root.title("ChatBot")

# 创建聊天历史显示区域
chat_history = scrolledtext.ScrolledText(root, state='disabled', wrap=tk.WORD)
chat_history.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# 创建用户输入框
user_input_box = tk.Text(root, height=3, wrap=tk.WORD)
user_input_box.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# 创建发送按钮
send_button = tk.Button(root, text="Send", command=chat_with_bot)
send_button.pack(padx=10, pady=10)

# 运行主循环
root.mainloop()