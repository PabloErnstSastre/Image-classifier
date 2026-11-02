import argparse
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from dataset import get_dataloaders
from model import CIFAR10CNN

def train_model(epochs=15, batch_size=64, lr=0.001, save_path='best_model.pth'):
    # Detectar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Entrenando en el dispositivo: {device}")
    
    # Obtener dataloaders
    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size)
    
    # Inicializar modelo
    model = CIFAR10CNN().to(device)
    
    # Pérdida y optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Scheduler para ajustar la tasa de aprendizaje si la pérdida de val se estanca
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5
    )
    
    # Listas para guardar el historial y graficar
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_loss = float('inf')
    
    print("\nComenzando el entrenamiento...")
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # --- Fase de Entrenamiento ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Resetear gradientes
            optimizer.zero_grad()
            
            # Forward + Backward + Optimizar
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Calcular estadísticas
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = 100.0 * correct_train / total_train
        
        # --- Fase de Validación ---
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = 100.0 * correct_val / total_val
        
        # Ajustar lr de acuerdo con la pérdida de validación
        scheduler.step(epoch_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Guardar historial
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        # Guardar el mejor modelo si mejora la pérdida de validación
        saved_status = ""
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), save_path)
            saved_status = "[MEJOR MODELO GUARDADO]"
            
        epoch_time = time.time() - epoch_start
        print(f"Época {epoch+1:02d}/{epochs:02d} | "
              f"Pérdida Train: {epoch_train_loss:.4f} - Acc Train: {epoch_train_acc:.2f}% | "
              f"Pérdida Val: {epoch_val_loss:.4f} - Acc Val: {epoch_val_acc:.2f}% | "
              f"LR: {current_lr:.6f} | {epoch_time:.1f}s {saved_status}")
              
    total_time = time.time() - start_time
    print(f"\nEntrenamiento completado en {total_time/60:.2f} minutos.")
    print(f"Mejor pérdida de validación alcanzada: {best_val_loss:.4f}")
    
    # Graficar curvas de pérdida y precisión
    plot_curves(history)
    print("Gráficas guardadas como 'loss_accuracy_curves.png'")

def plot_curves(history):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Gráfica de Pérdida (Loss)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-o', label='Entrenamiento')
    plt.plot(epochs, history['val_loss'], 'r-s', label='Validación')
    plt.title('Curvas de Pérdida (Loss)')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida (CrossEntropy)')
    plt.legend()
    plt.grid(True)
    
    # Gráfica de Precisión (Accuracy)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-o', label='Entrenamiento')
    plt.plot(epochs, history['val_acc'], 'r-s', label='Validación')
    plt.title('Curvas de Precisión (Accuracy)')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('loss_accuracy_curves.png', dpi=300)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entrenamiento de CNN en CIFAR-10')
    parser.add_argument('--epochs', type=int, default=15, help='Número de épocas de entrenamiento')
    parser.add_argument('--batch_size', type=int, default=64, help='Tamaño del lote')
    parser.add_argument('--lr', type=float, default=0.001, help='Tasa de aprendizaje inicial')
    parser.add_argument('--save_path', type=str, default='best_model.pth', help='Ruta para guardar el modelo')
    
    args = parser.parse_args()
    train_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, save_path=args.save_path)
