

import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from torchvision import transforms
from PIL import Image
from sys import argv
import sys
import os
from index2bird import index2bird
import io
from saliency_map import saliency_map_generator as SMG

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.torch_model import Flower_model

device = torch.device("cuda" if torch.cuda.is_available else "cpu")
model_path = argv[1]
model = Flower_model(525).to(device)

try:
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

app = FastAPI()


model.eval()
smg = SMG(model)

@app.post("/predict/")
async def predict(file: UploadFile = File()):
    image = Image.open(file.file)
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)

    saliency = smg(image)
    prediction = torch.max(output, 1)



    return {"prediction": index2bird[prediction.indices.item()]}

@app.get("/imagetest/")
async def gen_img():
    from PIL import Image
    image = Image.new("RGB", (200,200), color=(200,0,0))
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format ="png")
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
