
import os
import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def generate_analysis_prompt(base64_image: str):
    return [
        {"type": "text", "text": "Analiza detalladamente esta imagen de un medidor eléctrico. Devuelve el análisis estructurado con estos módulos:"},
        {"type": "text", "text": "**Tipo de medidor**
**Cantidad de cables conectados**
**Estado general visible**
**Anomalías o intervenciones**
**Conclusión**"},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": "high"
            }
        }
    ]

@app.post("/api/gemini-report")
async def generate_report(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        base64_image = base64.b64encode(image_data).decode("utf-8")
        model = genai.GenerativeModel("gemini-pro-vision")
        response = model.generate_content(generate_analysis_prompt(base64_image))
        return {
            "filename": file.filename,
            "size_kb": round(len(image_data) / 1024, 1),
            "message": response.text
        }
    except Exception as e:
        return {"error": str(e)}
