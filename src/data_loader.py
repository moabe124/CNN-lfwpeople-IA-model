import tarfile

import kagglehub
from sympy.printing.pytorch import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def download_dataset():
    # dataset = "atulanandjha/lfwpeople"
    # zip_path = kagglehub.dataset_download(dataset)
    # print(f"Dataset baixado em: {zip_path}")
    #
    # # Pasta onde o zip está
    # base_path = os.path.dirname(zip_path)
    # zip_file_name = os.path.join(zip_path, "lfw-funneled.tgz")
    # extract_path = os.path.join(base_path, "3")
    # extract_path = os.path.join(extract_path, "lfw_funneled")
    # extract_path = os.path.join(extract_path, "lfw_funneled")

    import kagglehub

    # Download latest version
    zip_path = kagglehub.dataset_download("yakhyokhuja/ms1m-arcface-dataset")

    print("Path to dataset files:", zip_path)

    base_path = os.path.dirname(zip_path)

    extract_path = os.path.join("/home/moabe/.cache/kagglehub/datasets/yakhyokhuja/ms1m-arcface-dataset/versions/1/ms1m-arcface", "")

    # Extrai se ainda não existir
    # if not os.path.exists(extract_path):
    #     print(f"Extraindo {zip_file_name} para {base_path}...")
    #     with tarfile.open(zip_file_name, "r:gz") as tar:
    #         tar.extractall(path=extract_path)
    #     print("Extração concluída.")
    # else:
    #     print("Dataset já extraído.")

    # Retorna o caminho da pasta com as imagens
    return extract_path

def create_dataloaders(dataset_path, img_size=112, batch_size=32):
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

    # Lê imagens organizadas por pastas (cada pasta = uma classe)
    train_dataset = datasets.ImageFolder(root=dataset_path, transform=transform_train)
    train_dataset.samples = [(path, target) for path, target in train_dataset.samples]

    val_dataset = datasets.ImageFolder(root=dataset_path, transform=transform_val)


    # Divide dataset em treino e validação (80% / 20%)
    val_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

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
