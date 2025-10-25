# 1. Импорты
import gradio as gr
import onnxruntime as rt
from torchvision import transforms
from PIL import Image
import torch
import numpy as np

# 2. Константы — ДОЛЖНЫ СОВПАДАТЬ с порядком из ImageFolder!
# После запуска train.py вы увидите: "Найдены классы: [...]"
# Скопируйте этот порядок сюда!
CLASS_NAMES = ['leopard', 'puma', 'tiger']  # ←←← ОБЯЗАТЕЛЬНО ПРОВЕРЬТЕ ЭТОТ ПОРЯДОК!
IMG_SIZE = 224
MODEL_PATH = "model.onnx"  # Должен быть создан train.py

# 3. Загрузка ONNX-модели
sess = rt.InferenceSession(MODEL_PATH)
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

# 4. Трансформации (точно как val_transforms)
val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 5. Функция предсказания
def predict(image):
    if image is None:
        return {cls: 0.0 for cls in CLASS_NAMES}
    if image.mode != "RGB":
        image = image.convert("RGB")
    img_tensor = val_transforms(image).unsqueeze(0)  # (1, 3, 224, 224)
    ort_outs = sess.run([output_name], {input_name: img_tensor.numpy()})
    probabilities = torch.nn.functional.softmax(torch.tensor(ort_outs[0]), dim=1)[0]
    confidences = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))}
    return confidences

# 6. Gradio-интерфейс
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Загрузите изображение хищной кошки"),
    outputs=gr.Label(num_top_classes=3, label="Вероятности классов"),
    title="Классификатор хищных кошек",
    description="Загрузите изображение пумы, леопарда или тигра. Модель определит вид.",
    examples=[]  # можно добавить примеры, если есть: ["examples/leopard.jpg", ...]
)

# 7. Запуск
if __name__ == "__main__":
    iface.launch(share=True)