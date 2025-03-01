from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import openai


# 加载环境变量
load_dotenv()
openai.api_base = "https://api.deepseek.com"
openai.api_key = "sk-3168fd3ac35d4a1ba91149d4bb9a02ce"

# 初始化 Flask
app = Flask(__name__)

@app.route("/")
def home():
    """渲染主页"""
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """处理聊天请求"""
    user_input = request.json.get("message")
    # 添加错误处理
    try:
        response = openai.Completion.create(
            engine="deepseek-r1",  # 确保引擎名称正确
            prompt=user_input,
            max_tokens=150,
            temperature=0.7
        )
        bot_response = response.choices[0].text.strip()
        return jsonify({"response": bot_response})
    except Exception as e:
        # 返回错误信息
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)