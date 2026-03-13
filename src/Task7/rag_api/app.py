# app.py
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uvicorn
import sys
import time
from loguru import logger
from contextlib import asynccontextmanager

from rag_core import RAGCore

# Инициализация RAG Core при старте
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Запуск RAG API сервиса")
    app.state.rag = RAGCore()
    yield
    # Shutdown
    logger.info("👋 Завершение работы RAG API сервиса")

# Создание FastAPI приложения
app = FastAPI(
    title="DeepSeek-R1 RAG API",
    description="Минимальный RAG сервис с DeepSeek-R1-1.5B без сторонних оберток",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Модели данных
class QueryRequest(BaseModel):
    query: str = Field(..., description="Текстовый запрос", min_length=1, max_length=1000)
    top_k: Optional[int] = Field(5, description="Количество чанков для поиска", ge=1, le=20)

class QueryResponse(BaseModel):
    query: str
    answer: str
    len_answer: int
    timestamp: float
    success_answer: bool
    sources: List[Dict[str, Any]]
    documents_found: int

class HealthResponse(BaseModel):
    status: str
    embedding_model: str
    deepseek_model: str
    device: str
    vector_store: Dict[str, Any]
    quantization: str

def setup_json_logger(log_file: str = "log.json", rotation: str = "100 MB", retention: str = "30 days"):
    
    logger.remove()
    
    logger.add(
        log_file,
        enqueue=True,
        serialize=True,
        backtrace=True,
        diagnose=True
    )
    
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="DEBUG"
    )
    
    return logger


# Эндпоинты
@app.get("/", tags=["Root"])
async def root():
    return {
        "name": "DeepSeek-R1 RAG API",
        "version": "1.0.0",
        "description": "Минимальный RAG сервис без сторонних оберток",
        "endpoints": [
            {"path": "/health", "method": "GET", "description": "Проверка состояния"},
            {"path": "/query", "method": "POST", "description": "Задать вопрос"},
            {"path": "/documents/count", "method": "GET", "description": "Количество документов"}
        ]
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Проверка состояния сервиса"""
    return app.state.rag.health()

@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(request: QueryRequest):
    """
    Получение ответа на вопрос с использованием RAG
    
    - **query**: текст вопроса
    - **top_k**: количество чанков для поиска (опционально)
    """
    try:
        result = app.state.rag.ask(request.query, request.top_k)
        logger.info(result)
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Запуск
if __name__ == "__main__":
    setup_json_logger("log.json")

    # Настройка логирования
    logger.add("logs/rag_api_{time}.log", rotation="500 MB")
    
    # Запуск сервера
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=10001,
        reload=False  # В продакшене лучше выключить
    )