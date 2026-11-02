import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from dataset import CLASSES
from model import CIFAR10CNN

def predict_image(image_path, model_path='best_model.pth', output_path='prediction_result.png'):
    # Detectar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Verificar que el modelo y la imagen existan
    if not os.path.exists(model_path):
        print(f"Error: El modelo '{model_path}' no existe. Por favor, entrena la red primero.")
        return
        
    if not os.path.exists(image_path):
        print(f"Error: La imagen '{image_path}' no existe.")
        return
        
    # Inicializar y cargar modelo
    model = CIFAR10CNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Preprocesamiento de la imagen de entrada
    # CIFAR-10 usa media y std por canal:
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    
    preprocess = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Cargar y preprocesar la imagen
    try:
        raw_image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error al abrir la imagen: {e}")
        return
        
    input_tensor = preprocess(raw_image).unsqueeze(0).to(device) # Añadir dimensión de lote
    
    # Ejecutar inferencia
    print(f"Realizando predicción para la imagen: {image_path}")
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        
    # Obtener clase con mayor probabilidad
    pred_idx = np.argmax(probabilities)
    pred_class = CLASSES[pred_idx]
    pred_prob = probabilities[pred_idx] * 100
    
    # Imprimir resultados
    print(f"\nResultado de la predicción:")
    print(f" - Clase más probable: {pred_class.upper()} ({pred_prob:.2f}% de confianza)")
    print(f"\nDistribución de probabilidades por clase:")
    for cls_name, prob in zip(CLASSES, probabilities):
        print(f" - {cls_name:12s}: {prob * 100:6.2f}%")
        
    # Crear gráfica con la predicción y el gráfico de barras
    visualize_prediction(raw_image, probabilities, pred_class, pred_prob, output_path)
    print(f"\nVisualización de predicción guardada como '{output_path}'")

def visualize_prediction(raw_image, probabilities, pred_class, pred_prob, output_path):
    plt.figure(figsize=(10, 5))
    
    # Subplot 1: Mostrar la imagen de entrada original
    plt.subplot(1, 2, 1)
    plt.imshow(raw_image)
    plt.title(f"Imagen Original\nPred: {pred_class.upper()} ({pred_prob:.1f}%)")
    plt.axis('off')
    
    # Subplot 2: Mostrar el gráfico de barras de probabilidades
    plt.subplot(1, 2, 2)
    y_pos = np.arange(len(CLASSES))
    colors = ['gray'] * len(CLASSES)
    # Resaltar la clase ganadora en azul
    colors[CLASSES.index(pred_class)] = '#1f77b4'
    
    plt.barh(y_pos, probabilities * 100, align='center', color=colors, alpha=0.8)
    plt.yticks(y_pos, [cls.capitalize() for cls in CLASSES])
    plt.gca().invert_yaxis()  # Poner la clase con índice menor arriba
    plt.xlabel('Probabilidad (%)')
    plt.title('Distribución de Probabilidad')
    plt.xlim(0, 100)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inferencia con CNN en CIFAR-10')
    parser.add_argument('--image_path', type=str, required=True, help='Ruta a la imagen local a clasificar')
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='Ruta al archivo del modelo guardado')
    parser.add_argument('--output_path', type=str, default='prediction_result.png', help='Ruta para guardar la gráfica de salida')
    
    args = parser.parse_args()
    predict_image(image_path=args.image_path, model_path=args.model_path, output_path=args.output_path)
