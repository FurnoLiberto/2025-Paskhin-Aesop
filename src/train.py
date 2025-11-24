import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from data_loader import get_data_loader
from model import AesopLSTM

# --- Конфигурация ---
DATA_FILE = "./data/aesop/data.txt"
SEQ_LENGTH = 20      # Длина контекста
BATCH_SIZE = 512
EMBEDDING_DIM = 100  # Как на скриншоте
HIDDEN_DIM = 256     # Как на скриншоте
EPOCHS = 5
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    print(f"Запуск обучения на устройстве: {DEVICE}")
    
    # 1. Загрузка данных
    dataloader, dataset = get_data_loader(DATA_FILE, BATCH_SIZE, SEQ_LENGTH)
    vocab_size = dataset.vocab_size
    
    # 2. Модель
    model = AesopLSTM(vocab_size, EMBEDDING_DIM, HIDDEN_DIM).to(DEVICE)
    
    # 3. Лосс и Оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 4. Логгер
    writer = SummaryWriter('runs/aesop_experiment')
    
    # Цикл обучения
    step = 0
    for epoch in range(EPOCHS):
        model.train()
        total_epoch_loss = 0
        
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass
            # hidden state не передаем явно между батчами, так как у нас shuffle=True,
            # каждый батч независим
            outputs, _ = model(x) 
            
            loss = criterion(outputs, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_epoch_loss += loss.item()
            
            # Логирование в TensorBoard каждые 50 шагов
            if step % 50 == 0:
                writer.add_scalar('Loss/train', loss.item(), step)
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{step}], Loss: {loss.item():.4f}")
            
            step += 1
            
        avg_loss = total_epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] завершена. Средний Loss: {avg_loss:.4f}")
        
        # Сохранение чекпоинта каждую эпоху
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"aesop_model_epoch_{epoch+1}.pth")

    # Сохранение финальной модели и метаданных для инференса
    print("Сохранение финальной модели...")
    torch.save(model.state_dict(), "aesop_final.pth")
    
    # Сохраняем словари, чтобы использовать их при инференсе
    torch.save({
        'word_to_int': dataset.word_to_int,
        'int_to_word': dataset.int_to_word,
        'seq_length': SEQ_LENGTH,
        'embedding_dim': EMBEDDING_DIM,
        'hidden_dim': HIDDEN_DIM
    }, "vocab.pth")
    
    writer.close()
    print("Обучение завершено.")

if __name__ == "__main__":
    train()
