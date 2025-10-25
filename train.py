# 1. Импорты
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import timm
import os
import random
import numpy as np
from dataclasses import dataclass

# 2. Дата-класс для хранения конфигурации
@dataclass
class Config:
    SEED: int = 42
    DATA_PATH: str = 'data/raw'
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE: int = 16
    IMG_SIZE: int = 224
    LEARNING_RATE: float = 0.001
    NUM_EPOCHS: int = 5
    MODEL_NAME: str = 'efficientnet_b0'
    NUM_CLASSES: int = 3
    SAVE_PATH: str = 'efficientnet_b0_best.pth'
    ONNX_PATH: str = 'model.onnx'  # Путь для ONNX-модели

# 3. Функция для фиксации случайных чисел
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 4. Конвертация модели в ONNX (после обучения)
def export_to_onnx(model, config: Config):
    model.eval()
    dummy_input = torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE).to(config.DEVICE)
    torch.onnx.export(
        model,
        dummy_input,
        config.ONNX_PATH,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=13
    )
    print(f"Модель экспортирована в ONNX: {config.ONNX_PATH}")

# 5. Основная функция обучения
def train(config: Config):
    print(f"Используется устройство: {config.DEVICE}")
    set_seed(config.SEED)

    # Трансформации
    train_transforms = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Загрузка данных
    full_dataset = datasets.ImageFolder(config.DATA_PATH, transform=train_transforms)
    print(f"Найдены классы (в порядке ImageFolder): {full_dataset.classes}")
    assert len(full_dataset.classes) == config.NUM_CLASSES, "Количество классов не совпадает!"

    # Разделение с фиксированным генератором
    generator = torch.Generator().manual_seed(config.SEED)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    # Применяем правильные трансформации для валидации
    val_dataset.dataset.transform = val_transforms

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Модель
    model = timm.create_model(config.MODEL_NAME, pretrained=True, num_classes=config.NUM_CLASSES)
    model.to(config.DEVICE)
    print(f"Модель {config.MODEL_NAME} загружена.")

    # Оптимизатор и функция потерь
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Обучение
    best_val_acc = 0.0
    print("Начало обучения...")

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        print(f"Эпоха {epoch+1}/{config.NUM_EPOCHS}, "
              f"Train Loss: {running_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.SAVE_PATH)
            print(f"Новая лучшая модель сохранена с точностью {best_val_acc:.2f}% в файл {config.SAVE_PATH}")

    # Загружаем лучшую модель и экспортируем в ONNX
    model.load_state_dict(torch.load(config.SAVE_PATH, map_location=config.DEVICE))
    export_to_onnx(model, config)

    print("Обучение и экспорт в ONNX завершены.")

# 6. Точка входа
if __name__ == '__main__':
    config = Config()
    train(config)