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

# Настройка CORS для Render
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://w5model.netlify.app",
            "http://localhost:*",
            "https://*.netlify.app",
            "https://hdghs.onrender.com"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
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
        (r'(#{1,3}) (.*)', r'\n\1 \2\n', 0),  # Добавлен флаг 0
        (r'\*\*(.*?)\*\*', r'**\1**', 0),     # Добавлен флаг 0
        (r'\*(.*?)\*', r'*\1*', 0)            # Добавлен флаг 0
    ]
    
    for pattern, repl, flags in replacements:
        text = re.sub(pattern, repl, text, flags=flags)
    return text

@app.route('/')
def health_check():
    """Эндпоинт для проверки работоспособности"""
    return jsonify({
        "status": "OK",
        "service": "AI Assistant",
        "version": "1.1",
        "port": os.environ.get("PORT", "3000")
    })

@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat_handler():
    """Основной обработчик запросов"""
    try:
        if request.method == 'OPTIONS':
            return _build_cors_preflight_response()
            
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
                "HTTP-Referer": "https://w5model.netlify.app/",
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
        return _corsify_actual_response(jsonify({
            "content": format_code_blocks(content)
        }))

    except Exception as e:
        app.logger.exception("Ошибка обработки запроса")
        return _corsify_actual_response(jsonify({"error": str(e)})), 500

def _build_cors_preflight_response():
    response = jsonify()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "*")
    response.headers.add("Access-Control-Allow-Methods", "*")
    return response

def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "https://hdghs.onrender.com")
    response.headers.add("Access-Control-Allow-Credentials", "true")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    return response

asgi_app = WsgiToAsgi(app)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        "app:asgi_app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000)),  # Используем порт Render по умолчанию
        log_level="info",
        timeout_keep_alive=30
    )
