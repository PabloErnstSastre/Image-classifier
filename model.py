import torch
import torch.nn as nn

class CIFAR10CNN(nn.Module):
    """
    Una red neuronal convolucional (CNN) personalizada para clasificar CIFAR-10.
    Inspirada en la arquitectura VGG, con Batch Normalization y Dropout para 
    evitar el sobreajuste (overfitting).
    """
    def __init__(self, num_classes=10):
        super(CIFAR10CNN, self).__init__()
        
        # Bloque Convolucional 1
        # Entrada: 3 x 32 x 32 -> Salida: 32 x 16 x 16
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2)
        )
        
        # Bloque Convolucional 2
        # Entrada: 32 x 16 x 16 -> Salida: 64 x 8 x 8
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3)
        )
        
        # Bloque Convolucional 3
        # Entrada: 64 x 8 x 8 -> Salida: 128 x 4 x 4
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.4)
        )
        
        # Capas Completamente Conectadas (Clasificador)
        # Entrada: 128 * 4 * 4 = 2048 -> Salida: num_classes
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        logits = self.classifier(x)
        return logits

if __name__ == '__main__':
    # Pequeño test para verificar que el tensor fluye correctamente por la red
    model = CIFAR10CNN()
    
    # Simular un lote de 8 imágenes de 3 canales y 32x32 píxeles
    x = torch.randn(8, 3, 32, 32)
    output = model(x)
    
    print("Test de la arquitectura CNN:")
    print(f" - Dimensiones de entrada: {x.shape}")
    print(f" - Dimensiones de salida (logits): {output.shape}")
    
    # Calcular el número de parámetros
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" - Número total de parámetros entrenables: {total_params:,}")
