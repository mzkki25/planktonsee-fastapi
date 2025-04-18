from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi import Request
from fastapi import Form

from utils import read_image, predict_img

import os
import logging
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
    return templates.TemplateResponse("opening.html", {"request": request})

@app.get("/predict_ui", response_class=HTMLResponse)
async def render_action(request: Request):
    return templates.TemplateResponse("predict.html", {"request": request})

@app.post("/upload_action")
async def upload_image(file: UploadFile = File(...)):
    if not file:
        logger.error("No file part in the request")
        raise HTTPException(status_code=400, detail="No file part")
    
    filename = file.filename
    if filename == '':
        logger.error("No selected file")
        raise HTTPException(status_code=400, detail="No selected file")

    try:
        file_location = os.path.join(UPLOAD_FOLDER, filename)
        with open(file_location, "wb") as f:
            f.write(await file.read())
            
        logger.info(f"File saved to {file_location}")
        return JSONResponse(content={
            "img_path": file_location
        }, status_code=200)
    
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    
@app.post("/predict_action")
async def predict_action(
    request: Request,
    img_path: UploadFile = File(...),
    model_option: str = Form("yolov8-detect"),
    llm_option: str = Form("gemini")
):
    try:
        logger.info("Received image for prediction")
        image = read_image(await img_path.read()) 

        logger.info("Image read successfully")
        predict_image = await predict_img(model_option, llm_option, image)

        logger.info("Prediction completed")
        actual_classes = predict_image["detected_classes"] or ["Tidak terdeteksi"]
        probability_classes = predict_image["confidences"] or [99999]

        predictions = list(zip(
            [(" ".join(c.split("_"))).title() for c in actual_classes],
            [f'{float(p):.6f}' for p in probability_classes]
        ))

        text_encoder = "Plankton tidak terdeteksi"

        if actual_classes[0] != "Tidak terdeteksi":
            text_link = predict_image["response_public_url"]

            response = requests.get(text_link)
            if response.status_code == 200:
                text_encoder = response.text

            logger.info(f"Text link: {text_link}")

        return JSONResponse(content={
            "img_path": predict_image["image_public_url"],
            "predictions": predictions,
            "response": text_encoder
        }, status_code=200)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return JSONResponse(content={
            "error": str(e)
        }, status_code=500)
    
@app.get("/result", response_class=HTMLResponse)
async def result_page(request: Request):
    return templates.TemplateResponse("result.html", {"request": request})

@app.get("/error", response_class=HTMLResponse)
async def error_page(request: Request):
    return templates.TemplateResponse("error.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    import os
    uvicorn.run(port=int(os.environ.get("PORT", 8080)), host='0.0.0.0', debug=True)