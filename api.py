from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torchvision.transforms as transforms
from src.model import FaceRecognitionCNN
import io
from src.data_loader import download_dataset, load_class_names
from src.model import FaceRecognitionCNN
from torchvision import transforms
import torch

# Define transform para manter consistência com treino/validação
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Dataset e classes
dataset_path = download_dataset()
class_names = load_class_names(dataset_path)
num_classes = len(class_names)

print(f"numero de classes: {num_classes}")

# Modelo
device = torch.device(device="cuda" if torch.version.hip else "cpu")
model = FaceRecognitionCNN(num_classes)
model.load_state_dict(torch.load("models/face_recognition.pth", map_location=device))
model.eval().to(device)

app = FastAPI()

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        top5_probs, top5_indices = torch.topk(probs, 5)

    top5 = [
        {"class": class_names[idx], "probability": float(prob)}
        for prob, idx in zip(top5_probs[0], top5_indices[0])
    ]

    return JSONResponse(content={"top5_predictions": top5})
