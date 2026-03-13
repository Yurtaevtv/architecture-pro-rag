# test_index_completeness.py
"""
Скрипт для автоматического тестирования обновлённой версии индекса RAG системы
с оценкой полноты ответов на наборе "золотых" вопросов.
"""

import asyncio
import aiohttp
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from loguru import logger
import argparse
from tqdm import tqdm
import hashlib
from collections import defaultdict
import warnings
import re

warnings.filterwarnings('ignore')

# Настройка логирования
logger.add("logs/test_index_{time}.log", rotation="100 MB", retention="10 days")

@dataclass
class GoldQuestion:
    id: str
    question: str
    expected_answer: str
    
@dataclass
class TestResult:
    question_id: str
    question: str
    answer: str
    expected_answer: str
    response_time: float
    completeness_score: float
    answer_length: int
    error: Optional[str] = None

class CompletenessCalculator:
    """Класс для вычисления полноты ответа на стороне теста"""
    
    def normalize_text(self, text: str) -> str:
        # Приводим к нижнему регистру
        text = text.lower()
        # Удаляем пунктуацию
        text = re.sub(r'[^\w\s]', ' ', text)
        # Убираем лишние пробелы
        text = ' '.join(text.split())
        return text
    
    def calculate_rouge_l(self, answer: str, expected: str) -> float:
        """
        Упрощенная ROUGE-L метрика (длина наибольшей общей подпоследовательности)
        """
        answer_words = self.normalize_text(answer).split()
        expected_words = self.normalize_text(expected).split()
        
        if not expected_words:
            return 1.0
        
        # Простое совпадение униграмм
        common = set(answer_words) & set(expected_words)
        
        return len(common) / len(set(expected_words))
    
   

class CompletenessTester:
    """Класс для тестирования полноты ответов"""
    
    def __init__(self, api_url: str = "http://localhost:10001"):
        self.api_url = api_url
        self.results_dir = Path("test_results")
        self.results_dir.mkdir(exist_ok=True)
        self.reports_dir = Path("test_reports")
        self.reports_dir.mkdir(exist_ok=True)
        
    def load_gold_questions(self, file_path: str) -> List[GoldQuestion]:
        """
        Загрузка золотых вопросов из JSON файла
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        questions = []
        for item in data:
            questions.append(GoldQuestion(
                id=item.get('id', hashlib.md5(item['question'].encode()).hexdigest()[:8]),
                question=item['question'],
                expected_answer=item['expected_answer'],
            ))
        
        logger.info(f"Загружено {len(questions)} золотых вопросов")
        return questions
    
    async def query_api(self, session: aiohttp.ClientSession, 
                       question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Отправка запроса к API
        """
        try:
            async with session.post(
                f"{self.api_url}/query",
                json={
                    "query": question,
                    "top_k": top_k
                },
                timeout=aiohttp.ClientTimeout(total=3000)
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    return {"error": f"HTTP {response.status}: {error_text}"}
        except Exception as e:
            return {"error": str(e)}
    
    async def evaluate_single(self, session: aiohttp.ClientSession,
                             question: GoldQuestion, top_k: int = 5) -> TestResult:
        """
        Оценка одного вопроса
        """
        start_time = datetime.now()
        
        try:
            # Получаем ответ от API
            result = await self.query_api(session, question.question, top_k)

            if "error" in result:
                return TestResult(
                    question_id=question.id,
                    question=question.question,
                    answer="",
                    expected_answer=question.expected_answer,
                    completeness_score=0.0,
                    response_time=(datetime.now() - start_time).total_seconds(),
                    answer_length=0,
                    error=result["error"]
                )
            
            # Используем встроенную оценку полноты из API
            completeness_details = result.get("completeness_details", {})
            calculator = CompletenessCalculator()
            return TestResult(
                question_id=question.id,
                question=question.question,
                answer=result.get("answer", ""),
                expected_answer=question.expected_answer,
                completeness_score=calculator.calculate_rouge_l(result.get("answer", ""), question.expected_answer),
                response_time=(datetime.now() - start_time).total_seconds(),
                answer_length=result.get("len_answer", 0)
            )
            
        except Exception as e:
            logger.error(f"Ошибка при оценке вопроса {question.id}: {e}")
            return TestResult(
                question_id=question.id,
                question=question.question,
                answer="",
                expected_answer=question.expected_answer,
                completeness_score=0.0,
                response_time=(datetime.now() - start_time).total_seconds(),
                answer_length=0,
                error=str(e)
            )
    
    async def run_tests(self, questions: List[GoldQuestion], 
                       top_k: int = 5, 
                       concurrent: int = 5) -> List[TestResult]:
        """
        Запуск тестирования на всех вопросах
        """
        results = []
        
        # Создаем сессию с ограничением на количество одновременных запросов
        connector = aiohttp.TCPConnector(limit=concurrent)
        async with aiohttp.ClientSession(connector=connector) as session:
            # Создаем задачи с ограничением параллелизма
            semaphore = asyncio.Semaphore(concurrent)
            
            async def bounded_evaluate(question):
                async with semaphore:
                    return await self.evaluate_single(session, question, top_k)
            
            # Запускаем все задачи с прогресс-баром
            tasks = [bounded_evaluate(q) for q in questions]
            for task in tqdm(asyncio.as_completed(tasks), 
                           total=len(tasks), 
                           desc="Тестирование вопросов"):
                result = await task
                results.append(result)
        
        return results


    def analyze_results(self, results: List[TestResult]) -> Dict[str, Any]:
        """
        Анализ результатов тестирования
        """
        if not results:
            return {}
        
        # Общая статистика
        completeness_scores = [r.completeness_score for r in results if r.error is None]
        
        analysis = {
            "total_questions": len(results),
            "successful_queries": len([r for r in results if r.error is None]),
            "failed_queries": len([r for r in results if r.error is not None]),
            "average_completeness": np.mean(completeness_scores) if completeness_scores else 0,
            "median_completeness": np.median(completeness_scores) if completeness_scores else 0,
            "std_completeness": np.std(completeness_scores) if completeness_scores else 0,
            "min_completeness": min(completeness_scores) if completeness_scores else 0,
            "max_completeness": max(completeness_scores) if completeness_scores else 0,
            "average_response_time": np.mean([r.response_time for r in results]),
        }
        
        return analysis
    
async def main():
    parser = argparse.ArgumentParser(description='Тестирование полноты ответов RAG системы')
    parser.add_argument('--gold-questions', type=str, required=True,
                       help='./gold_quest.json')
    parser.add_argument('--api-url', type=str, default='http://localhost:10001',
                       help='URL API сервиса')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Количество чанков для поиска')
    parser.add_argument('--concurrent', type=int, default=5,
                       help='Количество одновременных запросов')
    
    args = parser.parse_args()
    
    # Инициализация тестера
    tester = CompletenessTester(args.api_url)
    
    # Загрузка золотых вопросов
    questions = tester.load_gold_questions(args.gold_questions)
    
    # Запуск тестирования
    logger.info("Запуск тестирования...")
    results = await tester.run_tests(questions, args.top_k, args.concurrent)
   
    logger.info("ИТОГИ ТЕСТИРОВАНИЯ:")
    logger.info(f"{results}")
    
    analys_info = tester.analyze_results(results)

    logger.info("Данные анализа:")
    logger.info(f"{analys_info}")



if __name__ == "__main__":
    asyncio.run(main())