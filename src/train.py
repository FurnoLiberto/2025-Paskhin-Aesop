%%writefile src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from data_loader import get_data_loaders # Обратите внимание: функция теперь называется get_data_loaders (мн.ч.)
from model import AesopLSTM

# Конфигурация
DATA_FILE = "./data/aesop/data.txt"
SEQ_LENGTH = 20
BATCH_SIZE = 128
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    print(f"Запуск обучения на устройстве: {DEVICE}")
    
    # Загрузка данных (теперь получаем два лоадера)
    train_loader, val_loader, dataset = get_data_loaders(DATA_FILE, BATCH_SIZE, SEQ_LENGTH, val_split=0.2)
    vocab_size = dataset.vocab_size
    
    # Модель
    model = AesopLSTM(vocab_size, EMBEDDING_DIM, HIDDEN_DIM).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    writer = SummaryWriter('runs/aesop_experiment_with_val')
    
    step = 0
    best_val_loss = float('inf') # Для сохранения лучшей модели

    for epoch in range(EPOCHS):
        model.train() # Включаем режим обучения (Dropout работает, батч-норм обновляется)
        total_train_loss = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs, _ = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            if step % 50 == 0:
                writer.add_scalar('Loss/train_step', loss.item(), step)
            
            step += 1
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        model.eval() # Включаем режим оценки (выключаем Dropout и т.д.)
        total_val_loss = 0
        
        with torch.no_grad(): # Отключаем расчет градиентов для ускорения и экономии памяти
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                outputs, _ = model(x)
                loss = criterion(outputs, y)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        # Логируем средние значения за эпоху
        writer.add_scalars('Loss/epoch', {'train': avg_train_loss, 'val': avg_val_loss}, epoch)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Сохраняем лучшую модель (по валидации)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "aesop_best_model.pth")
            # print("  -> New best model saved!")

    print("Сохранение финальной модели...")
    torch.save(model.state_dict(), "aesop_final.pth")
    
    # Сохраняем словари
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
