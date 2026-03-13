# app/ingestor.py
import os
import logging
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
import time
from datetime import datetime
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
        
        self.docs_path = Path(os.getenv('SOURCE_DIR', 'documents'))
        self.state_file = Path(os.getenv('STAT_FILE', 'documents.json'))
        
        # Инициализация компонентов
        self.setup_chroma()
        self.setup_embedding_model()
        self.setup_text_splitter()
        self.load_document_state()
        
        logger.info(f"Инициализация завершена. Коллекция: {self.collection_name}")
        
    def setup_chroma(self):
        """Настройка подключения к ChromaDB"""
        try:
            # Ожидаем запуска ChromaDB
            self.wait_for_chroma()
            
            logger.info(f"Настройка клиента подключения к ChromaDB: {self.chroma_host}:{self.chroma_port}")
            # Создаем HTTP клиент
            self.chroma_client = chromadb.HttpClient(host=self.chroma_host, port=self.chroma_port)
            
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
    
    def load_document_state(self):
        """Загрузка состояния документов из файла"""
        self.document_state = {}
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    self.document_state = json.load(f)
                logger.info(f"Загружено состояние для {len(self.document_state)} документов")
            except Exception as e:
                logger.error(f"Ошибка загрузки состояния: {e}")
                self.document_state = {}
    
    def save_document_state(self):
        """Сохранение состояния документов в файл"""
        try:
            # Создаем директорию, если её нет
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.document_state, f, ensure_ascii=False, indent=2)
            logger.info(f"Сохранено состояние для {len(self.document_state)} документов")
        except Exception as e:
            logger.error(f"Ошибка сохранения состояния: {e}")
    
    def get_file_hash(self, file_path: Path) -> str:
        """Вычисление хеша файла для отслеживания изменений"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read(65536)  # Читаем по 64KB
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()
    
    def get_file_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Получение метаданных файла"""
        stat = file_path.stat()
        return {
            'size': stat.st_size,
            'mtime': stat.st_mtime,
            'hash': self.get_file_hash(file_path)
        }
    
    def get_changed_files(self) -> Tuple[List[Path], List[Path]]:
        """Определение новых и измененных файлов"""
        if not self.docs_path.exists():
            logger.error(f"Директория не найдена: {self.docs_path}")
            return [], []
        
        current_files = list(self.docs_path.glob('*'))
        new_files = []
        changed_files = []
        
        for file_path in current_files:
            if not file_path.is_file():
                continue
                
            file_name = file_path.name
            current_meta = self.get_file_metadata(file_path)
            
            if file_name not in self.document_state:
                # Новый файл
                new_files.append(file_path)
                logger.info(f"Новый файл: {file_name}")
            else:
                # Проверяем изменения
                saved_meta = self.document_state[file_name]
                if (current_meta['size'] != saved_meta['size'] or 
                    current_meta['mtime'] != saved_meta['mtime'] or
                    current_meta['hash'] != saved_meta['hash']):
                    changed_files.append(file_path)
                    logger.info(f"Изменен файл: {file_name}")
        
        return new_files, changed_files
    
    def get_deleted_files(self) -> List[str]:
        """Определение удаленных файлов"""
        if not self.docs_path.exists():
            return list(self.document_state.keys())
        
        current_files = {f.name for f in self.docs_path.glob('*') if f.is_file()}
        deleted_files = [f for f in self.document_state.keys() if f not in current_files]
        
        if deleted_files:
            logger.info(f"Удалено файлов: {len(deleted_files)}")
            for f in deleted_files:
                logger.info(f"  - {f}")
        
        return deleted_files
    
    def delete_document_chunks(self, file_name: str):
        """Удаление чанков документа из коллекции"""
        try:
            # Получаем все чанки для этого файла
            results = self.collection.get(
                where={"source": file_name}
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"Удалено {len(results['ids'])} чанков для {file_name}")
                
                # Удаляем из состояния
                if file_name in self.document_state:
                    del self.document_state[file_name]
                    
        except Exception as e:
            logger.error(f"Ошибка удаления чанков для {file_name}: {e}")
    
    def load_document(self, file_path: Path) -> List[Dict[str, Any]]:
        """Загрузка и чанкинг одного документа"""
        chunks = []
        
        try:
            # Определяем загрузчик по расширению
            ext = file_path.suffix.lower()
            
            # Здесь можно добавить разные загрузчики для разных типов файлов
            # Пока используем TextLoader для всех
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
                            'total_chunks': len(doc_chunks),
                            'indexed_at': datetime.now().isoformat()
                        }
                    })
            
            logger.debug(f"Загружен {file_path.name}: {len(chunks)} чанков")
            
        except Exception as e:
            logger.error(f"Ошибка загрузки {file_path}: {e}")
        
        return chunks
    
    def process_files(self, files: List[Path]) -> List[Dict[str, Any]]:
        """Обработка списка файлов"""
        all_chunks = []
        
        for file_path in tqdm(files, desc="Обработка файлов"):
            chunks = self.load_document(file_path)
            all_chunks.extend(chunks)
            
            # Обновляем состояние документа
            self.document_state[file_path.name] = self.get_file_metadata(file_path)
        
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
        
        logger.info(f"✅ Индексация завершена! Добавлено документов: {len(texts)}")
    
    def sync_documents(self):
        """Синхронизация документов с базой данных"""
        logger.info("🔄 Начало синхронизации документов")
        
        # Обработка удаленных файлов
        deleted_files = self.get_deleted_files()
        for file_name in deleted_files:
            self.delete_document_chunks(file_name)
        
        # Получение новых и измененных файлов
        new_files, changed_files = self.get_changed_files()
        
        # Удаляем старые чанки для измененных файлов
        for file_path in changed_files:
            self.delete_document_chunks(file_path.name)
        
        # Обрабатываем все файлы для индексации
        files_to_process = new_files + changed_files
        
        if not files_to_process:
            logger.info("Нет новых или измененных документов")
            return
        
        logger.info(f"Найдено файлов для обработки: {len(files_to_process)}")
        logger.info(f"  - Новых: {len(new_files)}")
        logger.info(f"  - Измененных: {len(changed_files)}")
        logger.info(f"  - Удаленных: {len(deleted_files)}")
        
        # Обработка файлов
        chunks = self.process_files(files_to_process)
        
        if chunks:
            # Индексация в ChromaDB
            self.index_documents(chunks)
            
            # Сохраняем состояние
            self.save_document_state()
        else:
            logger.warning("Не удалось загрузить чанки из файлов")
    
    def get_collection_stats(self):
        """Получение статистики коллекции"""
        try:
            collection_data = self.collection.get()
            count = len(collection_data['ids'])
            
            logger.info(f"\n{'='*50}")
            logger.info(f"СТАТИСТИКА КОЛЛЕКЦИИ: {self.collection_name}")
            logger.info(f"{'='*50}")
            logger.info(f"Всего документов (чанков): {count}")
            
            if count > 0:
                # Уникальные источники
                sources = set([m.get('source', 'unknown') 
                              for m in collection_data['metadatas']])
                logger.info(f"Уникальных файлов: {len(sources)}")
                
                # Сравнение с состоянием
                logger.info(f"\nСОСТОЯНИЕ ФАЙЛОВ:")
                logger.info(f"  В коллекции: {len(sources)} файлов")
                logger.info(f"  В состоянии: {len(self.document_state)} файлов")
                
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
    logger.info("🚀 Запуск синхронизации документов")
    
    try:
        # Создание ингестора
        ingestor = DocumentIngestor()
        
        # Синхронизация документов
        ingestor.sync_documents()
        
        # Статистика
        ingestor.get_collection_stats()
            
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        raise
    
    logger.info("✅ Процесс завершен")

if __name__ == "__main__":
    main()