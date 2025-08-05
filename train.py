import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from time import time
from src.data_loader import download_dataset, create_dataloaders
from src.model import FaceRecognitionCNN
from src.visualization import plot_training_curves, plot_confusion_matrix
import os

def train_model(epochs=20, lr=0.001, batch_size=128, device="cuda" if torch.version.hip else "cpu"):
    print("=" * 50)
    print("Iniciando treinamento do modelo de reconhecimento facial")
    print(f"Dispositivo: {device}")
    print(f"Épocas: {epochs} | Taxa de aprendizado: {lr} | Batch size: {batch_size}")
    print("=" * 50)

    print(f"Usa Cuda?: {device}")

    print(torch.version.hip)  # Deve retornar a versão, como '6.3.0'
    print(torch.cuda.is_available())  # Falso (normal em AMD)

    # Baixa e carrega dataset
    dataset_path = download_dataset()
    print(f"Dataset carregado de: {dataset_path}")
    train_loader, val_loader, num_classes = create_dataloaders(dataset_path, batch_size=batch_size)
    print(f"Total de classes detectadas: {num_classes}")
    print(f"Tamanho do treino: {len(train_loader.dataset)} | Validação: {len(val_loader.dataset)}")

    # Inicializa modelo
    model = FaceRecognitionCNN(num_classes).to(device)
    print(f"Modelo criado: {model}")

    # Define função de perda e otimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Histórico
    train_loss_history, val_loss_history = [], []
    train_acc_history, val_acc_history = [], []

    # Loop de treinamento
    for epoch in range(epochs):
        start_time = time()
        print(f"\n--- Época {epoch+1}/{epochs} ---")
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        batch_loss_values = []
        batch_acc_values = []

        # Treinamento por batch
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Treinando Época {epoch+1}")):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Print parcial a cada 50 batches
            if (batch_idx + 1) % batch_size == 0 or (batch_idx + 1) == len(train_loader):
                acc_batch = (preds == labels).sum().item() / labels.size(0)
                print(f"[Treino - Época {epoch + 1}] Batch {batch_idx + 1}/{len(train_loader)} "
                      f"| Loss: {loss.item():.4f} | Acurácia do batch: {acc_batch:.4f}")
                avg_loss = sum(batch_loss_values[-batch_size:]) / len(batch_loss_values[-batch_size:])
                avg_acc = sum(batch_acc_values[-batch_size:]) / len(batch_acc_values[-batch_size:])
                print(f"[Média dos ultimos {batch_size} batches] Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")

                print(f"Epoca até agora, loss {running_loss / len(train_loader):.4f} | acc {correct/total:.4f}")

            batch_loss_values.append(loss.item())
            batch_acc_values.append((preds == labels).sum().item() / labels.size(0))

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        # Validação
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        # Tempo gasto
        epoch_time = time() - start_time
        print(f"[Época {epoch+1}] Treino -> Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"[Época {epoch+1}] Validação -> Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
        print(f"Tempo da época: {epoch_time:.2f} segundos")

        # Guarda histórico
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

    # Salva modelo treinado
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/face_recognition.pth")
    print("\nTreinamento concluído!")
    print("Modelo salvo em: models/face_recognition.pth")

    # Visualizações
    plot_training_curves(train_loss_history, val_loss_history, train_acc_history, val_acc_history)
    plot_confusion_matrix(model, val_loader, device)

if __name__ == "__main__":
    train_model()
