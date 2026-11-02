import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Clases de CIFAR-10 en orden
CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def get_dataloaders(data_dir='./data', batch_size=64, val_split=0.1, num_workers=0):
    """
    Descarga (si es necesario) y prepara los DataLoaders de CIFAR-10.
    
    Args:
        data_dir (str): Directorio donde se guardarán los datos.
        batch_size (int): Tamaño del lote.
        val_split (float): Proporción del conjunto de entrenamiento asignada a validación.
        num_workers (int): Número de hilos para cargar datos.
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Estadísticas de CIFAR-10 para normalización (media y desviación estándar por canal)
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    
    # Transformaciones para entrenamiento (con aumento de datos)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Transformaciones para validación/prueba (sin aumentos espaciales)
    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Descargar y cargar el dataset completo de entrenamiento
    full_train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    
    # Cargar el dataset de prueba
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=val_test_transform
    )
    
    # Crear la división de validación
    num_train = len(full_train_dataset)
    num_val = int(num_train * val_split)
    num_train_subset = num_train - num_val
    
    # Usamos un generador con semilla fija para reproducibilidad
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(
        full_train_dataset, [num_train_subset, num_val], generator=generator
    )
    
    # IMPORTANTE: Aplicar la transformación de validación (sin aumentos) al subconjunto de validación.
    # Por defecto, random_split hereda las transformaciones del dataset base. 
    # Para cambiar la transformación del subconjunto de validación de forma limpia:
    class MapDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, map_fn):
            self.dataset = dataset
            self.map_fn = map_fn
            
        def __getitem__(self, index):
            x, y = self.dataset[index]
            return self.map_fn(x), y
            
        def __len__(self):
            return len(self.dataset)
            
    # Para cambiar la transformación de validación de manera limpia en PyTorch,
    # podemos extraer los índices del split de validación y aplicarlos a un dataset con val_test_transform.
    val_dataset_clean = datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=val_test_transform
    )
    val_subset_clean = torch.utils.data.Subset(val_dataset_clean, val_subset.indices)
    
    # Crear los cargadores de datos (DataLoaders)
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset_clean, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    print(f"Dataset cargado correctamente:")
    print(f" - Imágenes de Entrenamiento: {len(train_subset)}")
    print(f" - Imágenes de Validación: {len(val_subset_clean)}")
    print(f" - Imágenes de Prueba: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    # Pequeño script de prueba de carga de datos
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=64)
    images, labels = next(iter(train_loader))
    print(f"Dimensiones de un lote de imágenes de entrenamiento: {images.shape}")
    print(f"Dimensiones de un lote de etiquetas de entrenamiento: {labels.shape}")
