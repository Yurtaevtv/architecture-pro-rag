# app.py
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uvicorn
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
    sources: List[Dict[str, Any]]
    documents_found: int
    processing_time: float
    model: str

class DocumentAddRequest(BaseModel):
    text: str = Field(..., description="Текст документа", min_length=1)
    source: str = Field(..., description="Название источника")
    chunk_size: Optional[int] = Field(500, description="Размер чанка", ge=100, le=2000)

class DocumentAddResponse(BaseModel):
    success: bool
    source: str
    chunks_added: int
    total_documents: int

class HealthResponse(BaseModel):
    status: str
    embedding_model: str
    deepseek_model: str
    device: str
    vector_store: Dict[str, Any]
    quantization: str

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
        logger.info(f"Ответ на запрос: {result}")
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Запуск
if __name__ == "__main__":
    # Настройка логирования
    logger.add("logs/rag_api_{time}.log", rotation="500 MB")
    
    # Запуск сервера
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=10001,
        reload=False  # В продакшене лучше выключить
    )