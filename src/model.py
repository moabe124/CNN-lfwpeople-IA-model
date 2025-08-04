import torch.nn as nn
import torch.nn.functional as F

class FaceRecognitionCNN(nn.Module):
    def __init__(self, num_classes):
        super(FaceRecognitionCNN, self).__init__()
        # Camadas convolucionais para extrair features das imagens
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)      # 3 canais (RGB) -> 32 filtros
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)     # 32 -> 64 filtros
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)    # 64 -> 128 filtros
        self.pool = nn.MaxPool2d(2, 2)                   # Reduz tamanho da imagem pela metade
        self.dropout = nn.Dropout(0.5)                   # Evita overfitting zerando neurônios aleatórios
        self.fc1 = nn.Linear(128 * 12 * 12, 128)         # Camada totalmente conectada
        self.fc2 = nn.Linear(128, num_classes)           # Saída com número de classes

    def forward(self, x):
        # Passa pelos blocos Conv + ReLU + Pool
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv3(x)))  # Conv3 -> ReLU -> Pool
        x = x.view(x.size(0), -1)             # "Flatten" (transforma em vetor)
        x = F.relu(self.fc1(x))               # Primeira fully connected
        x = self.dropout(x)                   # Dropout
        x = self.fc2(x)                       # Saída (logits)
        return x
