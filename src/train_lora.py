import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from peft import LoraConfig, get_peft_model
from vectorize import get_prepared_data
from model import TextClassifier


def evaluate_model(model: nn.Module, data_loader: DataLoader, loss_fn: nn.Module) -> tuple[float, float, float]:
    """
    Оценивает модель на тестовых данных.
    
    Returns:
        tuple: (avg_loss, accuracy, f1_score)
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            logits = model(batch_X)
            loss = loss_fn(logits, batch_y)
            total_loss += loss.item()
            
            _, predicted = torch.max(logits, dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return avg_loss, accuracy, f1


def create_dataloaders(X: torch.Tensor, y: torch.Tensor, test_size: float = 0.2, 
                       batch_size: int = 32, random_state: int = 42) -> tuple[DataLoader, DataLoader]:
    """
    Разделяет данные на train/test и создает DataLoader'ы.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X.numpy(), y.numpy(), 
        test_size=test_size, 
        random_state=random_state
    )
    
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def run_lora_experiment(rank_value: int, num_epochs: int = 10, save_model: bool = False) -> dict:
    """
    Запускает эксперимент с LoRA для заданного ранга.
    
    Args:
        rank_value: Ранг LoRA-адаптера (4, 8, 16)
        num_epochs: Количество эпох обучения
        save_model: Сохранять ли обученные адаптеры
    
    Returns:
        dict: Метрики (accuracy, f1, baseline_acc, baseline_f1)
    """
    print(f"\n{'='*60}")
    print(f"Запуск эксперимента с LoRA (Ранг r={rank_value})")
    print(f"{'='*60}")
    
    # 1. Загрузка данных
    X, y = get_prepared_data()
    in_features = X.shape[1]
    num_classes = int(y.max()) + 1
    
    print(f"Размерность признаков: {in_features}")
    print(f"Количество классов: {num_classes}")
    
    # 2. Создание DataLoader'ов
    train_loader, test_loader = create_dataloaders(X, y)
    
    # 3. Baseline оценка
    base_model = TextClassifier(in_features=in_features, num_classes=num_classes)
    loss_fn = nn.CrossEntropyLoss()
    
    _, baseline_acc, baseline_f1 = evaluate_model(base_model, test_loader, loss_fn)
    print(f"\n[Baseline] Необученная модель:")
    print(f"  Accuracy: {baseline_acc:.4f}")
    print(f"  F1-score: {baseline_f1:.4f}")
    
    # 4. Настройка LoRA
    lora_config = LoraConfig(
        r=rank_value,
        lora_alpha=2 * rank_value,  # Стандартное правило: alpha = 2 * r
        target_modules=["0", "2"],  # Оба Linear слоя в Sequential
        lora_dropout=0.1,
        bias="none"
    )
    
    # 5. Применение LoRA
    model = get_peft_model(base_model, lora_config)
    print(f"\n[LoRA Configuration]")
    model.print_trainable_parameters()
    
    # 6. Обучение
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"\n[Training] Начало обучения на {num_epochs} эпохах...")
    best_acc = 0.0
    best_f1 = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Оценка после каждой эпохи
        avg_loss, acc, f1 = evaluate_model(model, test_loader, loss_fn)
        
        print(f"  Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")
        
        # Сохранение лучшей модели
        if acc > best_acc:
            best_acc = acc
            best_f1 = f1
            
            if save_model:
                output_dir = f"models/lora_adapter_r{rank_value}"
                os.makedirs(output_dir, exist_ok=True)
                model.save_pretrained(output_dir)
                print(f"  ✓ Сохранена лучшая модель в {output_dir}")
    
    # 7. Итоговые метрики
    print(f"\n[Results] Ранг r={rank_value}:")
    print(f"  Лучшая Accuracy: {best_acc:.4f} (улучшение: +{best_acc - baseline_acc:.4f})")
    print(f"  Лучшая F1-score: {best_f1:.4f} (улучшение: +{best_f1 - baseline_f1:.4f})")
    
    return {
        'rank': rank_value,
        'baseline_acc': baseline_acc,
        'baseline_f1': baseline_f1,
        'best_acc': best_acc,
        'best_f1': best_f1,
        'improvement_acc': best_acc - baseline_acc,
        'improvement_f1': best_f1 - baseline_f1
    }


def compare_experiments(results: list[dict]) -> None:
    """
    Выводит сравнительную таблицу результатов экспериментов.
    """
    print(f"\n{'='*60}")
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ ЭКСПЕРИМЕНТОВ")
    print(f"{'='*60}")
    print(f"{'Ранг':<8} {'Baseline Acc':<15} {'Best Acc':<12} {'Улучшение':<12} {'F1-score':<12}")
    print(f"{'-'*60}")
    
    for r in results:
        print(f"r={r['rank']:<5} {r['baseline_acc']:<15.4f} {r['best_acc']:<12.4f} "
              f"+{r['improvement_acc']:<10.4f} {r['best_f1']:<12.4f}")
    
    # Находим лучший ранг
    best_result = max(results, key=lambda x: x['best_acc'])
    print(f"\n🏆 Лучший результат: Ранг r={best_result['rank']} "
          f"с Accuracy={best_result['best_acc']:.4f}")


def main():
    """
    Главная функция: запускает серию экспериментов с разными рангами LoRA.
    """
    print("🚀 Запуск серии экспериментов с LoRA")
    print(f"Дата: {torch.__version__}")
    
    # Параметры экспериментов
    ranks_to_test = [4, 8, 16]
    num_epochs = 10
    
    results = []
    
    # Запуск экспериментов
    for rank in ranks_to_test:
        save_model = (rank == 8)  # Сохраняем только модель с рангом 8
        result = run_lora_experiment(
            rank_value=rank,
            num_epochs=num_epochs,
            save_model=save_model
        )
        results.append(result)
    
    # Сравнение результатов
    compare_experiments(results)
    
    print(f"\n✅ Все эксперименты завершены успешно!")


if __name__ == "__main__":
    main()