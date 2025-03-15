from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
from asgiref.wsgi import WsgiToAsgi
import os
import re
import logging

# Инициализация приложения
app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# Настройка CORS
CORS(app, resources={
    r"/chat": {
        "origins": [
            "https://w1model.netlify.app",
            "http://localhost:*",
            "https://*.netlify.app"
        ],
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

load_dotenv()

# Инициализация клиента OpenAI
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("AI_TOKEN")
)

def format_code_blocks(text):
    """Форматирование Markdown-контента"""
    replacements = [
        (r'```(\w+)?\n(.*?)\n```', r'```\1\n\2\n```', re.DOTALL),
        (r'(#{1,3}) (.*)', r'\n\1 \2\n'),
        (r'\*\*(.*?)\*\*', r'**\1**'),
        (r'\*(.*?)\*', r'*\1*')
    ]
    
    for pattern, repl, flags in replacements:
        text = re.sub(pattern, repl, text, flags=flags or 0)
    return text

@app.route('/')
def health_check():
    """Эндпоинт для проверки работоспособности"""
    return jsonify({
        "status": "OK",
        "service": "AI Assistant",
        "version": "1.0"
    })

@app.route('/chat', methods=['POST'])
def chat_handler():
    """Основной обработчик запросов"""
    try:
        data = request.get_json()
        
        # Валидация входных данных
        if not data or ('userInput' not in data and 'imageUrl' not in data):
            app.logger.error('Invalid request data: %s', data)
            return jsonify({"error": "Требуется текст или изображение"}), 400

        # Формирование контента для OpenAI
        user_content = []
        if data.get('userInput'):
            user_content.append({"type": "text", "text": data['userInput']})
        if data.get('imageUrl'):
            user_content.append({
                "type": "image_url", 
                "image_url": {"url": data['imageUrl']}
            })

        # Запрос к OpenAI
        response = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://w1model.netlify.app/",
                "X-Title": "My AI Assistant"
            },
            model="google/gemma-3-27b-it:free",
            messages=[
                {"role": "system", "content": "Вы очень полезный помощник отвечающий на русском языке!"},
                {"role": "user", "content": user_content}
            ],
            max_tokens=1024,
            temperature=0.7
        )

        # Форматирование ответа
        content = response.choices[0].message.content
        return jsonify({
            "content": format_code_blocks(content)
        })

    except Exception as e:
        app.logger.exception("Ошибка обработки запроса")
        return jsonify({"error": str(e)}), 500

asgi_app = WsgiToAsgi(app)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        asgi_app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 3000)),
        log_level="info"
    )