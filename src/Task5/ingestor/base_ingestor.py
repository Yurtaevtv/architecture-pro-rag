# app/ingestor.py
import os
import logging
from pathlib import Path
from typing import List, Dict, Any
import time
from tqdm import tqdm

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader
)
from dotenv import load_dotenv

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()

class DocumentIngestor:
    """Класс для индексации документов в ChromaDB"""
    
    def __init__(self):
        # Параметры подключения к ChromaDB
        self.chroma_host = os.getenv('CHROMA_HOST', 'localhost')
        self.chroma_port = int(os.getenv('CHROMA_PORT', 8000))
        self.collection_name = os.getenv('COLLECTION_NAME', 'documents')
        self.embedding_model_name = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        
        # Параметры чанков
        self.chunk_size = int(os.getenv('CHUNK_SIZE', 500))
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 50))
        
        # Путь к документам
        self.docs_path = Path('/app/documents')
        
        # Инициализация компонентов
        self.setup_chroma()
        self.setup_embedding_model()
        self.setup_text_splitter()
        
        logger.info(f"Инициализация завершена. Коллекция: {self.collection_name}")
        
    def setup_chroma(self):
        """Настройка подключения к ChromaDB"""
        try:
            # Ожидаем запуска ChromaDB
            self.wait_for_chroma()
            
            logger.info(f"Настройка клиента подключения к ChromaDB: {self.chroma_host}:{self.chroma_port}")
            # Создаем HTTP клиент
            self.chroma_client = chromadb.HttpClient(host=self.chroma_host, port=self.chroma_port,)
            
            heartbeat = self.chroma_client.heartbeat()
            logger.info(f"Heartbeat: {heartbeat}")

            logger.info(f"Получение коллекции ChromaDB: {self.collection_name}")
            # Получаем или создаем коллекцию
            try:
                # Пробуем получить коллекцию
                self.collection = self.chroma_client.get_collection(self.collection_name)
                logger.info(f"✅ Коллекция {self.collection_name} существует")
                logger.info(f"   В коллекции {self.collection.count()} документов")
            except Exception as e:
                # Если коллекции нет, создаем новую
                logger.info(f"Создание новой коллекции {self.collection_name}")
                self.collection = self.chroma_client.create_collection(self.collection_name)
                logger.info(f"✅ Коллекция {self.collection_name} создана")            
            logger.info(f"Подключение к ChromaDB установлено: {self.chroma_host}:{self.chroma_port}")
            
        except Exception as e:
            logger.error(f"Ошибка подключения к ChromaDB: {e}")
            raise
    
    def wait_for_chroma(self, max_retries=30, delay=2):
        """Ожидание готовности ChromaDB"""
        for i in range(max_retries):
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex((self.chroma_host, self.chroma_port))
                sock.close()
                
                if result == 0:
                    logger.info("ChromaDB готова к работе")
                    return True
                    
            except Exception:
                pass
            
            logger.info(f"Ожидание ChromaDB... попытка {i+1}/{max_retries}")
            time.sleep(delay)
        
        raise Exception("ChromaDB не доступна")
    
    def setup_embedding_model(self):
        """Загрузка модели эмбеддингов"""
        logger.info(f"Загрузка модели эмбеддингов: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
    def setup_text_splitter(self):
        """Настройка сплиттера текста"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_document(self, file_path: Path) -> List[Dict[str, Any]]:
        """Загрузка и чанкинг одного документа"""
        chunks = []
        
        try:
            # Определяем загрузчик по расширению
            ext = file_path.suffix.lower()
        
            loader = TextLoader(str(file_path), encoding='utf-8')
            documents = loader.load()
                        
            # Разбиваем на чанки
            for doc in documents:
                doc_chunks = self.text_splitter.split_text(doc.page_content)
                
                for i, chunk in enumerate(doc_chunks):
                    chunks.append({
                        'text': chunk,
                        'metadata': {
                            'source': str(file_path.name),
                            'chunk_id': i,
                            'file_type': ext,
                            'total_chunks': len(doc_chunks)
                        }
                    })
            
            logger.debug(f"Загружен {file_path.name}: {len(chunks)} чанков")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки {file_path}: {e}")
        
        return chunks
    
    def load_all_documents(self) -> List[Dict[str, Any]]:
        """Загрузка всех документов из директории"""
        if not self.docs_path.exists():
            logger.error(f"Директория не найдена: {self.docs_path}")
            return []
        
        all_chunks = []
        files = list(self.docs_path.glob('*'))
        
        logger.info(f"Найдено файлов: {len(files)}")
        
        for file_path in tqdm(files, desc="Загрузка документов"):
            if file_path.is_file():
                chunks = self.load_document(file_path)
                all_chunks.extend(chunks)
        
        logger.info(f"Всего загружено чанков: {len(all_chunks)}")
        return all_chunks
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Создание эмбеддингов для текстов"""
        logger.info(f"Создание эмбеддингов для {len(texts)} текстов")
        
        # Обработка батчами для экономии памяти
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            embeddings = self.embedding_model.encode(batch, show_progress_bar=False)
            all_embeddings.extend(embeddings.tolist())
            
            logger.debug(f"Обработан батч {i//batch_size + 1}")
        
        return all_embeddings
    
    def index_documents(self, chunks: List[Dict[str, Any]]):
        """Индексация документов в ChromaDB"""
        if not chunks:
            logger.warning("Нет документов для индексации")
            return
        
        # Подготовка данных
        texts = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        ids = [f"{meta['source']}_{meta['chunk_id']}" 
               for meta in metadatas]
        
        # Создание эмбеддингов
        embeddings = self.create_embeddings(texts)
        
        # Очистка существующей коллекции (опционально)
        logger.info("Очистка существующих данных...")
        try:
            existing_ids = self.collection.get()['ids']
            if existing_ids:
                self.collection.delete(ids=existing_ids)
        except:
            pass
        
        # Добавление в ChromaDB
        logger.info(f"Добавление {len(texts)} документов в ChromaDB...")
        
        # Добавляем батчами
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            end_idx = min(i + batch_size, len(texts))
            
            self.collection.add(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx],
                documents=texts[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )
            
            logger.info(f"Добавлен батч {i//batch_size + 1}: {end_idx-i} документов")
        
        logger.info(f"✅ Индексация завершена! Всего документов: {len(texts)}")
    
    def get_collection_stats(self):
        """Получение статистики коллекции"""
        try:
            collection_data = self.collection.get()
            count = len(collection_data['ids'])
            
            logger.info(f"\n{'='*50}")
            logger.info(f"СТАТИСТИКА КОЛЛЕКЦИИ: {self.collection_name}")
            logger.info(f"{'='*50}")
            logger.info(f"Всего документов: {count}")
            
            if count > 0:
                # Уникальные источники
                sources = set([m.get('source', 'unknown') 
                              for m in collection_data['metadatas']])
                logger.info(f"Уникальных файлов: {len(sources)}")
                
                # Пример метаданных
                logger.info(f"\nПример метаданных:")
                for i, meta in enumerate(collection_data['metadatas'][:3]):
                    logger.info(f"  {i+1}. {meta}")
            
            return count
            
        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}")
            return 0

def main():
    """Основная функция"""
    logger.info("🚀 Запуск индексации документов")
    
    try:
        # Создание ингестора
        ingestor = DocumentIngestor()
        
        # Загрузка документов
        chunks = ingestor.load_all_documents()
        
        if chunks:
            # Индексация в ChromaDB
            ingestor.index_documents(chunks)
            
            # Статистика
            ingestor.get_collection_stats()
        else:
            logger.warning("Документы не найдены или не удалось загрузить")
            
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        raise
    
    logger.info("✅ Процесс завершен")

if __name__ == "__main__":
    main()# app/ingestor.py