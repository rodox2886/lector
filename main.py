from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os
import base64
import httpx

load_dotenv()

app = FastAPI()

# Habilitar CORS para permitir peticiones desde GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Podés restringir esto a tu dominio GitHub si querés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

@app.post("/api/gemini-report")
async def generate_gemini_report(file: UploadFile = File(...)):
    try:
        if not GEMINI_API_KEY:
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY no configurada")

        image_bytes = await file.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": file.content_type,
                                "data": image_base64
                            }
                        },
                        {
                            "text": "Analiza esta imagen de un medidor eléctrico y genera un resumen con estos campos: Estado general, Observaciones visuales, Anomalías detectadas, Acciones observadas y Resumen del informe."
                        }
                    ]
                }
            ]
        }

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-vision:generateContent",
                headers=headers,
                json=payload
            )

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)

        gemini_data = response.json()
        text_parts = [part.get("text", "") for part in gemini_data.get("candidates", [])[0].get("content", {}).get("parts", [])]
        result_text = "\n".join(text_parts)

        return {"result": result_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
