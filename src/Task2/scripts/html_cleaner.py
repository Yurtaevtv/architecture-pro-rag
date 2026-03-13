import os
from bs4 import BeautifulSoup
from html_to_markdown import convert
import re

def batch_convert_html_to_md(input_dir, output_dir):
    """Конвертирует все HTML файлы в папке в Markdown"""
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(('.html', '.htm')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename.replace('.html', '.md').replace('.htm', '.md'))
            
            with open(input_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            markdown = convert(html_content)
            
            # Удаляем markdown-ссылки вида [текст](url)
            # Оставляем только текст в квадратных скобках
            markdown = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', markdown)
            
            # Удаляем URL в тексте
            markdown = re.sub(r'https?://[^\s]+', '', markdown)
            markdown = re.sub(r'www\.[^\s]+', '', markdown)
            
            # Удаляем пустые ссылки [](url)
            markdown = re.sub(r'\[\]\([^\)]+\)', '', markdown)
            
            # Очищаем лишние пробелы
            # markdown = re.sub(r'\s+', ' ', markdown)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown)
            
            print(f"✅ {filename} → {os.path.basename(output_path)}")

def clean_html_from_file(input_file_path):
    """Читает файл и возвращает текст без HTML‑тегов."""
    with open(input_file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Удаляем скрипты и стили
    for script in soup(["script", "style"]):
        script.decompose()
    
    # Получаем текст
    clean_text = soup.get_text()
    
    # Разбиваем на строки и удаляем лишние пробелы
    lines = (line.strip() for line in clean_text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    clean_text = '\n'.join(chunk for chunk in chunks if chunk)

    return clean_text

def process_directory(input_dir, output_dir):
    """Обрабатывает все файлы в директории и сохраняет очищенные версии."""
    # Создаём выходную директорию, если её нет
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Проходим по всем файлам в входной директории
    for filename in os.listdir(input_dir):
        input_file_path = os.path.join(input_dir, filename)
        
        # Пропускаем поддиректории, обрабатываем только файлы
        if os.path.isfile(input_file_path):
            try:
                # Очищаем файл от HTML
                cleaned_text = batch_convert_html_to_md(input_file_path)
                
                # Формируем путь для выходного файла
                output_file_path = os.path.join(output_dir, filename)
                
                # Сохраняем очищенный текст
                with open(output_file_path, 'w', encoding='utf-8') as output_file:
                    output_file.write(cleaned_text)
                
                print(f"Обработан: {filename}")
            except Exception as e:
                print(f"Ошибка при обработке файла {filename}: {e}")

# Настройки
input_directory = "./../original_knowledge"  # Директория с исходными файлами
output_directory = "./../processed_knowledge_md"  # Директория для очищенных файлов

batch_convert_html_to_md(input_directory, output_directory)

# Запуск обработки
#process_directory(input_directory, output_directory)
print("Обработка завершена!")