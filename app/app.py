# 1. Импортируем необходимые библиотеки
import gradio as gr
import onnxruntime as rt
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
import os

# 2. Определяем константы и переменные
# Путь к нашей ONNX модели
MODEL_PATH = "app\model.onnx"
# Названия наших классов (важно, чтобы порядок был тот же, что и при обучении!)
CLASS_NAMES = ['puma', 'leopard', 'tiger']
# Размер изображений, на котором обучалась модель
IMG_SIZE = 224

# 3. Создаем сессию для инференса (предсказания) с помощью ONNX Runtime
# Это наш "движок", который будет выполнять модель
sess = rt.InferenceSession(MODEL_PATH)
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

# 4. Определяем трансформации для входного изображения
# Они должны быть АБСОЛЮТНО ТАКИМИ ЖЕ, как val_transforms в нашем ноутбуке
val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 5. Создаем главную функцию, которая будет делать предсказание
def predict(image):
    """
    Принимает на вход изображение (от интерфейса Gradio),
    обрабатывает его и возвращает словарь с вероятностями классов.
    """
    # Конвертируем изображение в RGB, если оно имеет альфа-канал (прозрачность)
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    # Применяем к изображению наши трансформации
    img_tensor = val_transforms(image)
    
    # Добавляем "батчевое" измерение, так как модель ожидает на вход (batch, channels, h, w)
    # Наш тензор сейчас (3, 224, 224), а должен быть (1, 3, 224, 224)
    img_tensor = img_tensor.unsqueeze(0)

    # Делаем предсказание с помощью ONNX-модели
    # Результат (ort_outs) будет в формате numpy array
    ort_outs = sess.run([output_name], {input_name: img_tensor.numpy()})
    
    # Получаем вероятности, применяя Softmax
    # Softmax превращает "сырые" выходы модели (логиты) в вероятности от 0 до 1
    probabilities = torch.nn.functional.softmax(torch.tensor(ort_outs[0]), dim=1)[0]
    
    # Создаем красивый словарь для вывода: { 'класс': вероятность }
    confidences = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))}
    
    return confidences

# 6. Создаем и настраиваем интерфейс Gradio
# gr.Interface - это главный класс для создания UI
# fn - какую функцию вызывать
# inputs - какой компонент использовать для входа (в нашем случае - загрузка изображения)
# outputs - какой компонент использовать для выхода (метка с вероятностями)
# title, description - заголовки для нашего веб-интерфейса
# examples - список путей к картинкам, которые появятся как примеры
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Загрузите изображение"),
    outputs=gr.Label(num_top_classes=3, label="Вероятности классов"),
    title="Классификатор транспорта",
    description="Загрузите изображение легкового автомобиля, велосипеда или мотоцикла, чтобы модель определила его тип.",
)

# 7. Запускаем приложение!
if __name__ == "__main__":
    iface.launch(share=True)