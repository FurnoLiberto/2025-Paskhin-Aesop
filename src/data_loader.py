import torch
from torch.utils.data import Dataset, DataLoader
import re
from collections import Counter

class AesopDataset(Dataset):
    def __init__(self, file_path, sequence_length=10):
        self.sequence_length = sequence_length
        self.raw_text = self.load_and_clean_text(file_path)
        
        # Токенизация (простая, по словам)
        self.words = self.raw_text.split()
        
        # Построение словаря
        self.word_counts = Counter(self.words)
        # Сортируем слова по частоте
        self.sorted_words = sorted(self.word_counts, key=self.word_counts.get, reverse=True)
        
        self.int_to_word = {k: w for k, w in enumerate(self.sorted_words)}
        self.word_to_int = {w: k for k, w in enumerate(self.sorted_words)}
        
        self.vocab_size = len(self.int_to_word)
        
        # Преобразование всего текста в числа
        self.encoded_text = [self.word_to_int[w] for w in self.words]
        
        print(f"Dataset loaded. Total words: {len(self.words)}. Vocab size: {self.vocab_size}")

    def load_and_clean_text(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # 1. Нижний регистр
        text = text.lower()

        # 2. Очистка от мусора Project Gutenberg
        # Обычно текст начинается после заголовка и заканчивается перед лицензией
        # Эвристика для ID 21
        start_marker = "*** start of the project gutenberg ebook"
        end_marker = "*** end of the project gutenberg ebook"
        
        start_idx = text.find(start_marker)
        end_idx = text.find(end_marker)
        
        if start_idx != -1:
            text = text[start_idx + len(start_marker):]
        if end_idx != -1:
            text = text[:end_idx]

        # 3. Разделение рассказов
        # В этом файле басни часто разделены пустыми строками. 
        # Заменяем двойные переносы строк на спец-токен разделителя.
        separator = " ||||||||||||||||||| "
        text = re.sub(r'\n\s*\n', separator, text)
        
        # Убираем одиночные переносы строк (делаем текст сплошным внутри абзаца)
        text = text.replace('\n', ' ')
        
        # Отделяем знаки препинания пробелами, чтобы они стали отдельными токенами
        text = re.sub(r'([.,!?;:"\(\)])', r' \1 ', text)
        
        # Убираем лишние пробелы
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def __len__(self):
        return len(self.encoded_text) - self.sequence_length

    def __getitem__(self, index):
        # Вход: последовательность слов
        # Выход: следующее слово
        x = torch.tensor(self.encoded_text[index:index+self.sequence_length])
        y = torch.tensor(self.encoded_text[index+self.sequence_length])
        return x, y

def get_data_loader(file_path, batch_size=64, sequence_length=10):
    dataset = AesopDataset(file_path, sequence_length)
    # drop_last=True важно, чтобы все батчи были одного размера для LSTM state
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader, dataset
