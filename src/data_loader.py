import zipfile

import kagglehub
from sympy.printing.pytorch import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def download_dataset():
    dataset = "atulanandjha/lfwpeople"
    zip_path = kagglehub.dataset_download(dataset)
    print(f"Dataset baixado em: {zip_path}")

    # Caminho da pasta onde vai extrair (mesma pasta do zip, sem extensão)
    extract_path = zip_path.replace(".zip", "") + "\\lfw-funneled"

    print(f"Dataset baixado em: {extract_path}")

    # Se já não foi extraído, extrai agora
    if not os.path.exists(extract_path):
        print(f"Extraindo {zip_path} para {extract_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("Extração concluída.")
    else:
        print("Dataset já extraído.")

    # Retorna o caminho da pasta extraída
    return extract_path + "\\lfw_funneled"

def create_dataloaders(dataset_path, img_size=96, batch_size=32):
    """
    Cria DataLoaders para treino e validação com data augmentation.
    """

    # Transformações para os dados de treino (aumentam variedade e robustez)
    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),         # Redimensiona imagens
        transforms.RandomHorizontalFlip(),               # Inverte horizontalmente aleatoriamente
        transforms.RandomRotation(10),                   # Rotaciona levemente
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Varia iluminação e contraste
        transforms.ToTensor(),                           # Converte para tensor (PyTorch)
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normaliza valores (-1 a 1)
    ])

    # Transformações para validação (sem data augmentation, apenas normalização)
    transform_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 1. Carrega uma vez
    full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform_train)

    # 2. Split consistente
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # fixar o split
    )

    # 3. Aplique transformações ao acessar os dados
    train_dataset.dataset.transform = transform_train
    val_dataset.dataset.transform = transform_val

    # Cria DataLoaders para carregar os dados em batches
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,  # usa threads para paralelizar carregamento
        pin_memory=True  # otimiza transferência CPU->GPU
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    # Retorna os DataLoaders e a quantidade de classes
    return train_loader, val_loader, len(train_dataset.dataset.classes)
