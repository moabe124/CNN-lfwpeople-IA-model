import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.data_loader import download_dataset, create_dataloaders
from src.model import FaceRecognitionCNN
from src.visualization import plot_training_curves, plot_confusion_matrix
import os

def train_model(epochs=20, lr=0.001, device="cuda" if torch.cuda.is_available() else "cpu"):
    # Baixa o dataset
    dataset_path = download_dataset()

    # Cria DataLoaders
    train_loader, val_loader, num_classes = create_dataloaders(dataset_path)

    # Cria o modelo e envia para GPU (se disponível)
    model = FaceRecognitionCNN(num_classes).to(device)

    # Define a função de perda e otimizador
    criterion = nn.CrossEntropyLoss()  # Perda para classificação multiclasse
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Listas para guardar histórico
    train_loss_history, val_loss_history = [], []
    train_acc_history, val_acc_history = [], []

    # Loop de treinamento
    for epoch in range(epochs):
        model.train()  # Coloca modelo em modo treino
        running_loss, correct, total = 0.0, 0, 0

        # Loop por batch no treino
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Treino]"):
            images, labels = images.to(device), labels.to(device)

            # Zera gradientes
            optimizer.zero_grad()

            # Forward
            outputs = model(images)

            # Calcula perda
            loss = criterion(outputs, labels)

            # Backpropagation
            loss.backward()

            # Atualiza pesos
            optimizer.step()

            # Acumula métricas
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        # Calcula métricas de treino
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        # Validação
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        # Exibe métricas da época
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Salva histórico
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

    # Salva modelo treinado
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/face_recognition.pth")
    print("Modelo salvo em models/face_recognition.pth")

    # Plota gráficos e matriz de confusão
    plot_training_curves(train_loss_history, val_loss_history, train_acc_history, val_acc_history)
    plot_confusion_matrix(model, val_loader, device)
