# Описание проекта

Проект позволяет распознавать номерные знаки машин.


# Запуск

1. Скопируйте репозиторий

   ```
   git clone https://github.com/AsmodaiP/plates
   ```
2. Создайте  и активируйте виртуальное окружение

   ```
   python -m venv .venv
   .venv/Scripts/Activate
   ```
3. Установите зависимости

   ```
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. Запустите скрипт, указав путь до изображения

   ```
   python ml.py 'image.webp'
   ```
