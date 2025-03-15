from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
from asgiref.wsgi import WsgiToAsgi
import os
import re

app = Flask(__name__)
CORS(app)

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("AI_TOKEN"),
)

def format_code_blocks(text):
    text = re.sub(r'```(\w+)?\n(.*?)\n```', r'```\1\n\2\n```', text, flags=re.DOTALL)
    text = re.sub(r'### (.*)', r'\n### \1\n', text)
    text = re.sub(r'## (.*)', r'\n## \1\n', text)
    text = re.sub(r'# (.*)', r'\n# \1\n', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'**\1**', text)
    text = re.sub(r'\*(.*?)\*', r'*\1*', text)
    return text

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    
    # Проверка наличия обязательных полей
    if not data.get('userInput') and not data.get('imageUrl'):
        return jsonify({"error": "Missing both userInput and imageUrl fields"}), 400
    
    user_message = data.get('userInput', '')
    image_url = data.get('imageUrl', '')
    
    try:
        # Формируем содержимое сообщения
        user_content = []
        if user_message:
            user_content.append({"type": "text", "text": user_message})
        if image_url:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })
        
        response = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://w1model.netlify.app/",
                "X-Title": "My AI Assistant",
            },
            model="google/gemma-3-27b-it:free",  # Используем новую модель
            messages=[
                {"role": "system", "content": "Вы очень полезный помощник отвечающий на русском языке!"},
                {
                    "role": "user", 
                    "content": user_content
                }
            ],
            max_tokens=1024,
            temperature=0.7,
        )

        content = response.choices[0].message.content
        formatted_content = format_code_blocks(content)
        return jsonify({"content": formatted_content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
asgi_app = WsgiToAsgi(app)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(asgi_app, host="0.0.0.0", port=3000)