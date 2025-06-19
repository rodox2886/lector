from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pydantic import BaseModel
import base64
import requests
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("GEMINI_API_KEY")

class GeminiResponse(BaseModel):
    filename: str
    size_kb: float
    message: str

@app.post("/api/gemini-report", response_model=List[GeminiResponse])
async def generate_report(files: List[UploadFile] = File(...)):
    responses = []
    for file in files:
        content = await file.read()
        image_base64 = base64.b64encode(content).decode("utf-8")
        prompt = (
            "Analiza detalladamente esta imagen de un medidor eléctrico. "
            "Devuelve el análisis dividido en secciones con subtítulos en negrita: "
            "**Tipo de medidor**, **Cantidad de cables conectados**, **Estado general visible**, "
            "**Anomalías o intervenciones**, **Conclusión**."
        )
        gemini_payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/jpeg", "data": image_base64}},
                    ]
                }
            ]
        }
        try:
            res = requests.post(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-vision:generateContent",
                headers={"Content-Type": "application/json"},
                params={"key": API_KEY},
                json=gemini_payload,
                timeout=30,
            )
            res.raise_for_status()
            message = res.json()["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            message = f"Error al procesar con Gemini: {str(e)}"
        responses.append(
            GeminiResponse(
                filename=file.filename,
                size_kb=round(len(content) / 1024, 1),
                message=message,
            )
        )
    return responses
