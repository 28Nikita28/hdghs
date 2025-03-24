from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import re
import logging
from pydantic import BaseModel
import uvicorn

load_dotenv()

app = FastAPI(title="AI Assistant", version="1.1")

origins = [
    "https://w5model.netlify.app",
    "http://localhost:*",
    "https://*.netlify.app",
    "https://hdghs.onrender.com",
    "http://localhost:5174"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("uvicorn")
logger.setLevel(logging.INFO)

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("AI_TOKEN")
)

class ChatRequest(BaseModel):
    userInput: str | None = None
    imageUrl: str | None = None

def format_code_blocks(text: str) -> str:
    """Форматирование Markdown-контента"""
    replacements = [
        (r'```(\w+)?\n(.*?)\n```', r'```\1\n\2\n```', re.DOTALL),
        (r'(#{1,3}) (.*)', r'\n\1 \2\n', 0),
        (r'\*\*(.*?)\*\*', r'**\1**', 0),
        (r'\*(.*?)\*', r'*\1*', 0)
    ]
    
    for pattern, repl, flags in replacements:
        text = re.sub(pattern, repl, text, flags=flags)
    return text

@app.get("/")
async def health_check():
    """Эндпоинт для проверки работоспособности"""
    return {
        "status": "OK",
        "service": "AI Assistant",
        "version": "1.1",
        "port": os.environ.get("PORT", "10000")
    }

@app.post("/chat")
async def chat_handler(request: Request, chat_data: ChatRequest):
    """Основной обработчик запросов"""
    try:
        logger.info(f"Incoming request headers: {request.headers}")
        
        # Валидация входных данных
        if not chat_data.userInput and not chat_data.imageUrl:
            logger.error('Invalid request data')
            raise HTTPException(status_code=400, detail="Требуется текст или изображение")

        # Формирование контента для OpenAI
        user_content = []
        if chat_data.userInput:
            user_content.append({"type": "text", "text": chat_data.userInput})
        if chat_data.imageUrl:
            user_content.append({
                "type": "image_url", 
                "image_url": {"url": chat_data.imageUrl}
            })

        # Асинхронный запрос к OpenAI
        response = await client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://w5model.netlify.app/",
                "X-Title": "My AI Assistant"
            },
            model="deepseek/deepseek-chat-v3-0324:free",
            messages=[
                {"role": "system", "content": "Вы очень полезный помощник отвечающий на русском языке!"},
                {"role": "user", "content": user_content}
            ],
            max_tokens=4096,
            temperature=0.5
        )

        # Форматирование ответа
        content = response.choices[0].message.content
        return {"content": format_code_blocks(content)}

    except Exception as e:
        logger.exception("Ошибка обработки запроса")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000)),
        log_level="info",
        timeout_keep_alive=30
    )