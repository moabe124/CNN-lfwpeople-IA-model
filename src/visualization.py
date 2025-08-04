import matplotlib.pyplot as plt
import torch
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def plot_training_curves(train_loss, val_loss, train_acc, val_acc):
    """
    Plota curvas de perda e acurácia do treino e validação.
    """
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(14, 5))

    # Acurácia
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc, 'b', label='Treino')
    plt.plot(epochs, val_acc, 'r', label='Validação')
    plt.title('Acurácia')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()

    # Perda
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, 'b', label='Treino')
    plt.plot(epochs, val_loss, 'r', label='Validação')
    plt.title('Perda')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.legend()
    plt.show()

def plot_confusion_matrix(model, data_loader, device):
    """
    Plota a matriz de confusão e imprime relatório de classificação.
    """
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    # Calcula matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues", annot=False)
    plt.title("Matriz de Confusão")
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.show()

    # Relatório detalhado
    print("Relatório de classificação:")
    print(classification_report(y_true, y_pred))
