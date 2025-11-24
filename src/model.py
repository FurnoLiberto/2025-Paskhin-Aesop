import torch
import torch.nn as nn

class AesopLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=256):
        super(AesopLSTM, self).__init__()
        
        # 1. Embedding Layer
        # Input: Индексы слов
        # Output: Векторы размерности 100
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 2. LSTM Layer
        # Input size: 100 (от эмбеддинга)
        # Hidden size: 256
        # batch_first=True означает, что входные данные имеют форму (batch, seq, features)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
        
        # 3. Dense Layer (Linear)
        # Input: 256 (от LSTM)
        # Output: Размер словаря (для предсказания вероятности каждого слова)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        # x shape: (batch_size, sequence_length)
        
        # Embeddings
        # out: (batch_size, sequence_length, embedding_dim)
        embeds = self.embedding(x)
        
        # LSTM
        # lstm_out: (batch_size, sequence_length, hidden_dim)
        # hidden: (h_n, c_n) - состояния
        lstm_out, hidden = self.lstm(embeds, hidden)
        
        # Нам нужно предсказание только на основе последнего шага последовательности
        # Берем последний временной шаг для каждого элемента батча
        # last_step: (batch_size, hidden_dim)
        last_step = lstm_out[:, -1, :]
        
        # Fully Connected
        # out: (batch_size, vocab_size)
        out = self.fc(last_step)
        
        return out, hidden
