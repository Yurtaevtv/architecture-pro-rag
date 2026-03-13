import json
import re
import os
from pathlib import Path

def load_replacement_dict(json_file):
    """Загружает словарь замен из JSON файла"""
    try:
        with open(json_file, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Ошибка: Файл {json_file} не найден!")
        return None
    except json.JSONDecodeError:
        print(f"Ошибка: Файл {json_file} содержит некорректный JSON!")
        return None

def replace_words(text, replacement_dict):
    """
    Заменяет слова в тексте на основе словаря замен
    Учитывает границы слов и регистр
    """
    if not replacement_dict:
        return text
    
    # Сортируем ключи по убыванию длины для корректной замены
    sorted_keys = sorted(replacement_dict.keys(), key=len, reverse=True)
    
    for old_word in sorted_keys:
        new_word = replacement_dict[old_word]
        
        # Создаем паттерн для поиска слова с учетом границ
        pattern = r'\b' + re.escape(old_word) + r'\b'
        
        # Функция для замены с сохранением регистра
        def replace_case(match):
            matched = match.group(0)
            if matched.isupper():
                return new_word.upper()
            elif matched[0].isupper():
                return new_word.capitalize()
            else:
                return new_word
        
        text = re.sub(pattern, replace_case, text, flags=re.IGNORECASE)
    
    return text

def process_text(text, replacement_dict):
    """Обрабатывает текст, применяя все замены"""
    # Разбиваем текст на предложения для более точной обработки
    sentences = re.split(r'([.!?]+)', text)
    result = []
    
    for i in range(0, len(sentences), 2):
        sentence = sentences[i]
        if i + 1 < len(sentences):
            punctuation = sentences[i + 1]
        else:
            punctuation = ''
        
        # Заменяем слова в предложении
        modified_sentence = replace_words(sentence, replacement_dict)
        result.append(modified_sentence + punctuation)
    
    return ''.join(result)

def get_supported_files(directory, extensions=None):
    """Возвращает список файлов с поддерживаемыми расширениями"""
    if extensions is None:
        extensions = ['.txt', '.md', '.html', '.xml', '.json', '.csv', '.ini', '.cfg', '.log']
    
    files = []
    for ext in extensions:
        files.extend(Path(directory).glob(f'*{ext}'))
    return files

def process_directory(input_dir, output_dir, json_file, extensions=None):
    """
    Обрабатывает все файлы в директории и сохраняет результаты в другую директорию
    
    Args:
        input_dir (str): путь к входной директории
        output_dir (str): путь к выходной директории
        json_file (str): путь к JSON файлу со словарем замен
        extensions (list): список расширений файлов для обработки
    """
    
    # Загружаем словарь замен
    replacement_dict = load_replacement_dict(json_file)
    if replacement_dict is None:
        return
    
    # Создаем выходную директорию, если она не существует
    os.makedirs(output_dir, exist_ok=True)
    
    # Получаем список файлов для обработки
    files_to_process = get_supported_files(input_dir, extensions)
    
    if not files_to_process:
        print(f"В директории {input_dir} не найдено файлов с поддерживаемыми расширениями")
        return
    
    print(f"Найдено {len(files_to_process)} файлов для обработки")
    print(f"Загружено {len(replacement_dict)} замен из файла {json_file}")
    print("-" * 50)
    
    processed_count = 0
    error_count = 0
    
    for input_file in files_to_process:
        try:
            # Формируем путь к выходному файлу
            relative_path = input_file.relative_to(input_dir) if hasattr(input_file, 'relative_to') else Path(input_file).name
            output_file = Path(output_dir) / relative_path
            
            # Создаем поддиректории в выходной папке, если нужно
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Читаем входной файл
            with open(input_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Обрабатываем текст
            processed_text = process_text(text, replacement_dict)
            
            # Записываем результат
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(processed_text)
            
            print(f"✓ Обработан: {input_file.name} -> {output_file}")
            processed_count += 1
            
        except UnicodeDecodeError:
            print(f"✗ Ошибка кодировки в файле: {input_file.name} (пропущен)")
            error_count += 1
        except PermissionError:
            print(f"✗ Нет доступа к файлу: {input_file.name}")
            error_count += 1
        except Exception as e:
            print(f"✗ Ошибка при обработке {input_file.name}: {e}")
            error_count += 1
    
    print("-" * 50)
    print(f"Обработка завершена!")
    print(f"✓ Успешно обработано: {processed_count}")
    if error_count > 0:
        print(f"✗ Ошибок: {error_count}")

def main():
    # Настройки
    input_directory = "./../processed_knowledge_md"      # Папка с исходными файлами
    output_directory = "./../result_knowledge_md"    # Папка для обработанных файлов
    json_file = "../gen_terms.json"  # Файл со словарем замен
    
    # Расширения файлов для обработки (можно изменить)
    supported_extensions = [
        '.txt', '.md', '.html', '.xml', 
        '.json', '.csv', '.ini', '.cfg', 
        '.log', '.conf', '.yaml', '.yml'
    ]
    
    # Запускаем обработку
    process_directory(
        input_dir=input_directory,
        output_dir=output_directory,
        json_file=json_file,
        extensions=supported_extensions
    )

if __name__ == "__main__":
    main()