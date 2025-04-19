import cv2
import os
import logging
import firebase_admin
import tempfile
import os
import uuid
import google.generativeai as genai
import dotenv
import json
import numpy as np

dotenv.load_dotenv()

from io import BytesIO
from PIL import Image
from firebase_admin import credentials, storage
from ultralytics import YOLO
from gradio_client import Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def gemini(message):
    try:
        API_KEY = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=API_KEY)

        chatbot = genai.GenerativeModel("gemini-1.5-flash-002")
        system = f"""
            Saya ingin mendapatkan penjelasan mendalam tentang taksonomi dan klasifikasi plankton {message}. Berikut adalah beberapa aspek utama yang perlu dijelaskan:

            1. *Hierarki Taksonomi dalam Plankton:*  
            2. *Kelompok Utama Plankton Berdasarkan Taksonomi:*  
            3. *Klasifikasi Plankton Berdasarkan Kemampuan Bergerak:*  
            4. *Klasifikasi Berdasarkan Habitat:*  
            5. *Klasifikasi Berdasarkan Siklus Hidup:*  
            6. *Klasifikasi Berdasarkan Fungsi dalam Ekosistem:*  
            7. *Klasifikasi Berdasarkan Ukuran:*  
        """

        result = chatbot.generate_content(system).text

        return result

    except Exception as e:
        logger.error(f"Error di Gemini: {e}")
        return "Terjadi kesalahan saat menghubungi Gemini API."
    
def qwen2(message):
    client = Client("Qwen/Qwen2-57b-a14b-instruct-demo")
    
    try:
        result = client.predict(
            query=message,
            history=[],
            system=
                f"""
                Saya ingin mendapatkan penjelasan mendalam tentang taksonomi dan klasifikasi plankton {message}. Berikut adalah beberapa aspek utama yang perlu dijelaskan:  
                1. *Hierarki Taksonomi dalam Plankton:*
                2. *Kelompok Utama Plankton Berdasarkan Taksonomi:*  
                3. *Klasifikasi Plankton Berdasarkan Kemampuan Bergerak:*  
                4. *Klasifikasi Berdasarkan Habitat:*  
                5. *Klasifikasi Berdasarkan Siklus Hidup:*  
                6. *Klasifikasi Berdasarkan Fungsi dalam Ekosistem:*  
                7. *Klasifikasi Berdasarkan Ukuran:*  
                """,
            api_name="/model_chat"
        )
        
        return result[1][0][1]

    except Exception as e:
        logger.error(f"Error di Qwen2: {e}")
        return "Terjadi kesalahan saat menghubungi Qwen2 API."
    
def deepseek(message):
    client = Client("Abubekersiraj/Deepseek")
    
    try:
        result = client.predict(
            message=message,
            system_message=f"""
                Saya ingin mendapatkan penjelasan mendalam tentang taksonomi dan klasifikasi plankton {message}. Berikut adalah beberapa aspek utama yang perlu dijelaskan:  
                1. *Hierarki Taksonomi dalam Plankton:*
                2. *Kelompok Utama Plankton Berdasarkan Taksonomi:*  
                3. *Klasifikasi Plankton Berdasarkan Kemampuan Bergerak:*  
                4. *Klasifikasi Berdasarkan Habitat:*  
                5. *Klasifikasi Berdasarkan Siklus Hidup:*  
                6. *Klasifikasi Berdasarkan Fungsi dalam Ekosistem:*  
                7. *Klasifikasi Berdasarkan Ukuran:*  
                """,
            max_tokens=2048,
            temperature=0.1,
            top_p=0.95,
            api_name="/chat"
        )
        
        return result

    except Exception as e:
        logger.error(f"Error di DeepSeek: {e}")
        return "Terjadi kesalahan saat menghubungi DeepSeek"

try:
    cred = credentials.Certificate("credential/credentials.json")
    firebase_admin.initialize_app(cred) if not firebase_admin._apps else firebase_admin.get_app()

    bucket = storage.bucket(name='planktosee-temp-file')
    logger.info("Connected to Firestore successfully.")

except Exception as e:
    logger.error(f"Error initializing Firebase: {e}")
    raise e

def upload_image_to_firebase(image_path):
    try:
        blob = bucket.blob('images/' + os.path.basename(image_path))
        blob.upload_from_filename(image_path)
        blob.make_public()
        return blob.public_url
    except Exception as e:
        logger.error(f"Error uploading image to Firebase: {e}")
        return None
    
def upload_txt_to_firebase(txt_path):
    try:
        blob = bucket.blob('texts/' + os.path.basename(txt_path))
        blob.upload_from_filename(txt_path)
        blob.make_public()
        return blob.public_url
    
    except Exception as e:
        logger.error(f"Error uploading text to Firebase: {e}")
        return None

def read_image(file: bytes) -> np.ndarray:
    try:
        pil_image = Image.open(BytesIO(file))
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    except Exception as e:
        logger.error(f"Error reading image: {e}")
        raise e
    
async def predict_img(model_option: str, llm_option: str, img_path: np.ndarray) -> dict:
    if model_option == "yolov8-detect":
        model = YOLO("model/yolov8-detect.pt")
    elif model_option == "yolov8-acvit":
        model = YOLO("model/yolov8-acvit.pt")
    else:
        return "Model tidak ditemukan"

    img = cv2.resize(img_path, (864, 576))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = model(img)
    img_result = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)

    logger.info("Image predict successfully")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        temp_filename_img = uuid.uuid4().hex + ".jpg"
        cv2.imwrite(temp_filename_img, img_result)
    
    try:
        image_public_url = upload_image_to_firebase(temp_filename_img)
    except Exception as e:
        logger.error(f"Error uploading image to Firebase: {e}")

    os.remove(temp_filename_img)

    detected_classes = [model.names[int(box.cls)] for box in results[0].boxes]
    confidences = [float(box.conf) for box in results[0].boxes]

    if llm_option == "qwen":
        response = qwen2(detected_classes)
    elif llm_option == "deepseek":
        response = deepseek(detected_classes)
    elif llm_option == "gemini":
        response = gemini(detected_classes)

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
        temp_filename_text = uuid.uuid4().hex + ".txt"

        with open(temp_filename_text, 'w', encoding='utf-8') as f:
            f.write(response)

    logger.info("Text predict successfully")
    
    try:
        response_public_url = upload_txt_to_firebase(temp_filename_text)
    except Exception as e:
        logger.error(f"Error uploading text to Firebase: {e}")

    os.remove(temp_filename_text)

    return {
        "detected_classes": detected_classes,
        "confidences": confidences,
        "image_public_url": image_public_url,
        "response_public_url": response_public_url
    }