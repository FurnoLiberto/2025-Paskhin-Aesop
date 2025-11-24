import torch
import torch.nn.functional as F
from model import AesopLSTM
import re

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate(model, start_text, word_to_int, int_to_word, seq_length, max_words=100, temperature=1.0):
    model.eval()
    
    # Предобработка начального текста (как при обучении)
    text = start_text.lower()
    text = re.sub(r'([.,!?;:"\(\)])', r' \1 ', text)
    words = text.split()
    
    generated = words[:]
    
    with torch.no_grad():
        for _ in range(max_words):
            # Подготовка входной последовательности
            # Берем последние seq_length слов
            if len(words) >= seq_length:
                input_seq = words[-seq_length:]
            else:
                # Если слов меньше чем длина окна, используем то что есть (модель обработает)
                # Или можно дополнить паддингами (но для простоты берем срез)
                input_seq = words
            
            # Конвертация в индексы (0, если слово неизвестно)
            input_ids = [word_to_int.get(w, 0) for w in input_seq]
            
            # Добавляем размерность батча (1, len)
            x = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)
            
            # Прямой проход
            output, _ = model(x)
            
            # Применение температуры (чем выше, тем случайнее текст)
            logits = output / temperature
            probabilities = F.softmax(logits, dim=1)
            
            # Выбор следующего слова (сэмплирование)
            next_word_idx = torch.multinomial(probabilities, 1).item()
            next_word = int_to_word[next_word_idx]
            
            # Добавляем в список
            words.append(next_word)
            generated.append(next_word)
            
            # Если сгенерирован разделитель - можно остановиться или заменить на перенос
            if "|||" in next_word:
                generated[-1] = "\n\n--- КОНЕЦ ИСТОРИИ ---\n"
                break

    # Сборка текста обратно
    result = ' '.join(generated)
    # Убираем пробелы перед знаками препинания (обратная операция)
    result = re.sub(r'\s+([.,!?;:"\(\)])', r'\1', result)
    
    return result

if __name__ == "__main__":
    # Загрузка словаря и параметров
    checkpoint = torch.load("vocab.pth")
    word_to_int = checkpoint['word_to_int']
    int_to_word = checkpoint['int_to_word']
    seq_length = checkpoint['seq_length']
    embedding_dim = checkpoint['embedding_dim']
    hidden_dim = checkpoint['hidden_dim']
    
    vocab_size = len(word_to_int)
    
    # Инициализация и загрузка модели
    model = AesopLSTM(vocab_size, embedding_dim, hidden_dim).to(DEVICE)
    model.load_state_dict(torch.load("aesop_final.pth", map_location=DEVICE))
    
    print("Модель загружена. Генерация текста...\n")
    
    # Примеры генерации
    seeds = [
        "the wolf and the lamb",
        "a fox saw some grapes",
        "the lion went to sleep"
    ]
    
    for seed in seeds:
        print(f"SEED: {seed}")
        print("-" * 40)
        print(generate(model, seed, word_to_int, int_to_word, seq_length, temperature=0.8))
        print("\n" + "=" * 40 + "\n")
