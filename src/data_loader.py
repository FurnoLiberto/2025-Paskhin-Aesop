import torch
from torch.utils.data import Dataset, DataLoader, random_split
import re
from collections import Counter

class AesopDataset(Dataset):
    def __init__(self, file_path, sequence_length=10):
        self.sequence_length = sequence_length
        self.raw_text = self.load_and_clean_text(file_path)
        
        self.words = self.raw_text.split()
        self.word_counts = Counter(self.words)
        self.sorted_words = sorted(self.word_counts, key=self.word_counts.get, reverse=True)
        
        self.int_to_word = {k: w for k, w in enumerate(self.sorted_words)}
        self.word_to_int = {w: k for k, w in enumerate(self.sorted_words)}
        
        self.vocab_size = len(self.int_to_word)
        self.encoded_text = [self.word_to_int[w] for w in self.words]
        
        print(f"Dataset loaded. Total words: {len(self.words)}. Vocab size: {self.vocab_size}")

    def load_and_clean_text(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        text = text.lower()
        
        start_marker = "*** start of the project gutenberg ebook"
        end_marker = "*** end of the project gutenberg ebook"
        
        start_idx = text.find(start_marker)
        end_idx = text.find(end_marker)
        
        if start_idx != -1:
            text = text[start_idx + len(start_marker):]
        if end_idx != -1:
            text = text[:end_idx]

        separator = " ||||||||||||||||||| "
        text = re.sub(r'\n\s*\n', separator, text)
        text = text.replace('\n', ' ')
        text = re.sub(r'([.,!?;:"\(\)])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def __len__(self):
        return len(self.encoded_text) - self.sequence_length

    def __getitem__(self, index):
        x = torch.tensor(self.encoded_text[index:index+self.sequence_length])
        y = torch.tensor(self.encoded_text[index+self.sequence_length])
        return x, y

def get_data_loaders(file_path, batch_size=64, sequence_length=10, val_split=0.2):
    # Создаем полный датасет
    dataset = AesopDataset(file_path, sequence_length)
    
    # Вычисляем размеры сплитов
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    # Случайное разделение
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    print(f"Train size: {len(train_ds)}, Validation size: {len(val_ds)}")
    
    # Создаем DataLoader'ы
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    # Для валидации shuffle не обязателен, drop_last=True оставим для стабильности размеров батча
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    
    return train_loader, val_loader, dataset
