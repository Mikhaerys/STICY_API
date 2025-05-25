# Waste Detection API

This simple API service uses YOLO object detection to identify different types of waste in images and stores the detection results in a SQLite database.

## Setup

1. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Make sure you have the YOLO model file (`best.pt`) in the `Models` directory.

## Running the Server

Start the server with:
```bash
uvicorn main:app --reload
```

The server will be available at `http://localhost:8000`

## API Endpoints

### 1. Detect Waste
- **URL**: `/detect-waste/`
- **Method**: POST
- **Content-Type**: multipart/form-data
- **Body**: 
  - `file`: Image file to analyze
- **Response**: JSON with detected waste types and confidence scores

### 2. Get All Detections
- **URL**: `/detections/`
- **Method**: GET
- **Response**: JSON with all stored detections including waste type, confidence, and timestamp

## Example Usage

Using curl to detect waste in an image:
```bash
curl -X POST "http://localhost:8000/detect-waste/" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@path/to/your/image.jpg"
```

Get all detections:
```bash
curl -X GET "http://localhost:8000/detections/" -H "accept: application/json"
``` 