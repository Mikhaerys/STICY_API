from typing import Any
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from ultralytics import YOLO
from database import init_db, save_detection, get_all_detections
import torch

# Type hints for OpenCV
cv2.imdecode: Any
cv2.IMREAD_COLOR: Any

app = FastAPI(title="STICY API")

# Initialize the database
init_db()

# Load the YOLO model
model = YOLO('Models/best.pt')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print(f"Model running on: {model.device}")

# Class names
clsName = ['Metal', 'Glass', 'Plastic', 'Carton', 'Medical']


@app.post("/detect-waste/")
async def detect_waste(file: UploadFile = File(...)):
    try:
        # Read the image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Perform detection
        results = model(img, stream=True, verbose=False)

        detections = []
        for res in results:
            boxes = res.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if conf > 0.5:  # Only consider detections with confidence > 50%
                    waste_type = clsName[cls]
                    detections.append({
                        "waste_type": waste_type,
                        "confidence": conf
                    })
                    # Save to database
                    save_detection(waste_type, conf)

        if not detections:
            return JSONResponse(
                content={"message": "No waste detected in the image"},
                status_code=200
            )

        return {
            "detections": detections
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/detections/")
async def get_detections():
    try:
        detections = get_all_detections()
        return {
            "detections": [
                {
                    "id": d[0],
                    "waste_type": d[1],
                    "confidence": d[2],
                    "detection_date": d[3]
                }
                for d in detections
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        access_log=True
    )
