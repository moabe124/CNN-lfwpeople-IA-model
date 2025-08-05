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
from src.classNameList import class_names
import torch
from fastapi.middleware.cors import CORSMiddleware

# Define transform para manter consistência com treino/validação
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

print(f"numero de classes: {84585}")

# Modelo
device = torch.device(device="cuda" if torch.version.hip else "cpu")
model = FaceRecognitionCNN(84585)
model.load_state_dict(torch.load("models/face_recognition.pth", map_location=device))
model.eval().to(device)

app = FastAPI()

# Permite qualquer origem, qualquer método, qualquer header
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas as origens
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos os métodos (GET, POST, etc)
    allow_headers=["*"],  # Permite todos os headers
)

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
