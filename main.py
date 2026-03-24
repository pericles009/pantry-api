from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from PIL import Image
import io

# Iniciando a aplicação FastAPI
app = FastAPI(title="Fridge API", description="API for detecting food in the fridge")

# Carregando o modelo YOLO
model = YOLO("best.onnx", task="detect")


# Criar o endpoint
@app.post("/predict")
async def predict_food(file:UploadFile = File(...)):

  # Ler a imagem na memoria
  image_bytes = await file.read()
  image = Image.open(io.BytesIO(image_bytes))

  # Rodar a imagem no modelo
  results = model.predict(image, conf=0.4)

  # Formatar os resultados em formato JSON
  predictions = []
  for result in results:
    for box in result.boxes:
      predictions.append({
        "class": model.names[int(box.cls)],
        "confidence": round(float(box.conf), 2),
        "bounding_box": {
          "x1": round(float(box.xyxy[0][0]), 1),
          "y1": round(float(box.xyxy[0][1]), 1),
          "x2": round(float(box.xyxy[0][2]), 1),
          "y2": round(float(box.xyxy[0][3]), 1)
        }
      }) 


  # Retornar os dados requisitados
  return {
    "status": "success",
    "item_found": len(predictions),
    "predictions": predictions
  }