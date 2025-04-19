from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi import Request
from fastapi import Form
from firebase_admin import credentials, storage, firestore

from utils import read_image, predict_img

import firebase_admin
import os
import logging
import uuid
import requests

UPLOAD_FOLDER = 'all/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.mount("/all", StaticFiles(directory="all"), name="all")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def render_opening(request: Request): 
    for file in os.listdir('all/uploads'):
        os.remove(os.path.join('all/uploads', file))

    logger.info("Cleared upload folder")
    return templates.TemplateResponse("start.html", {"request": request})

@app.get("/predict_ui", response_class=HTMLResponse)
async def render_action(request: Request):
    return templates.TemplateResponse("predict.html", {"request": request})

@app.get("/result", response_class=HTMLResponse)
async def result_page(request: Request):
    return templates.TemplateResponse("result.html", {"request": request})
    
try:
    cred = credentials.Certificate("credential/credentials.json")
    firebase_admin.initialize_app(cred) if not firebase_admin._apps else firebase_admin.get_app()

    db = firestore.client(database_id='planktonsee-database')
    bucket = storage.bucket(name='planktosee-temp-file')
    logger.info("Connected to Firestore successfully.")

except Exception as e:
    logger.error(f"Error initializing Firebase: {e}")
    raise e

@app.post("/predict_action")
async def predict_action(
    request: Request, 
    img_path: UploadFile = File(...), 
    model_option: str = Form("yolov8-detect"), 
    llm_option: str = Form("gemini")
):
    try:
        image_data = await img_path.read()

        filename = f"{uuid.uuid4()}.jpg"
        blob = bucket.blob(f"images/{filename}")
        blob.upload_from_string(image_data, content_type=img_path.content_type)
        blob.make_public()

        image = read_image(image_data)
        predict_image = await predict_img(model_option, llm_option, image)

        predictions = [
            {"class": (" ".join(c.split("_"))).title(), "confidence": f'{float(p):.6f}'}
            for c, p in zip(predict_image["detected_classes"], predict_image["confidences"])
        ]

        doc_id = str(uuid.uuid4())
        db.collection("predictions").document(doc_id).set({
            "img_url": blob.public_url,
            "img_predict_url": predict_image.get("image_public_url"),
            "predictions": predictions,
            "response": predict_image.get("response_public_url") or "Tidak terdeteksi"
        })

        return JSONResponse(content={
            "doc_id": doc_id
        }, status_code=200)

    except Exception as e:
        return JSONResponse(content={
            "error": str(e)
        }, status_code=500)
    
@app.get("/result/{doc_id}", response_class=HTMLResponse)
async def result_page(request: Request, doc_id: str):
    doc = db.collection("predictions").document(doc_id).get()
    if not doc.exists:
        return HTMLResponse("Hasil tidak ditemukan.", status_code=404)
    
    data = doc.to_dict()

    response = requests.get(data['response'])
    if response.status_code == 200:
        text_encoder = response.text

    logger.info(f"Text link: {data['response']}")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "img_path": data["img_predict_url"],
        "predictions": data["predictions"],
        "response": text_encoder
    })

if __name__ == "__main__":
    import uvicorn
    import os
    uvicorn.run(port=int(os.environ.get("PORT", 8000)), host='0.0.0.0', debug=True)