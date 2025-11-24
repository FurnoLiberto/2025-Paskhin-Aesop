#!/bin/bash
FILE=$1
NAME=$2

# URL для скачивания текстовой версии (Plain Text UTF-8)
URL="https://www.gutenberg.org/cache/epub/$FILE/pg$FILE.txt"
TARGET_DIR="./data/$NAME/"

echo "Создаем директорию: $TARGET_DIR"
mkdir -p "$TARGET_DIR"

TXT_FILE="$TARGET_DIR/data.txt"

echo "Скачиваем файл с $URL..."
wget -N "$URL" -O "$TXT_FILE"

if [ -f "$TXT_FILE" ]; then
    echo "Загрузка завершена успешно. Файл сохранен в $TXT_FILE"
    # Показать первые 10 строк для проверки
    head -n 10 "$TXT_FILE"
else
    echo "Ошибка загрузки."
    exit 1
fi
