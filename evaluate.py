import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from dataset import get_dataloaders, CLASSES
from model import CIFAR10CNN

def evaluate_model(model_path='best_model.pth'):
    # Detectar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluando en dispositivo: {device}")
    
    # Cargar los dataloaders
    _, _, test_loader = get_dataloaders(batch_size=128)
    
    # Inicializar modelo y cargar pesos
    if not os.path.exists(model_path):
        print(f"Error: El archivo de pesos '{model_path}' no existe. Por favor, entrena el modelo primero.")
        return
        
    model = CIFAR10CNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_targets = []
    
    print("Realizando inferencia en el conjunto de prueba...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calcular precisión global
    accuracy = np.mean(all_preds == all_targets) * 100
    print(f"\n==========================================")
    print(f"Precisión global (Accuracy) en Test: {accuracy:.2f}%")
    print(f"==========================================\n")
    
    # Reporte de clasificación detallado por clase
    print("Reporte de clasificación detallado por clase:")
    report = classification_report(all_targets, all_preds, target_names=CLASSES)
    print(report)
    
    # Matriz de confusión
    cm = confusion_matrix(all_targets, all_preds)
    plot_confusion_matrix(cm, CLASSES)
    print("Matriz de confusión guardada como 'confusion_matrix.png'")

def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(10, 8))
    # Normalizar la matriz por filas (por clase real)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(
        cm_normalized, annot=cm, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes
    )
    plt.title('Matriz de Confusión Normalizada (CIFAR-10)')
    plt.ylabel('Clase Real')
    plt.xlabel('Clase Predicha')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluación de CNN en CIFAR-10')
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='Ruta al archivo del modelo guardado')
    
    args = parser.parse_args()
    evaluate_model(model_path=args.model_path)
