
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import httpx
import base64
import os

app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Obtener API Key de variable de entorno
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=" + GEMINI_API_KEY

@app.post("/api/gemini-report")
async def generate_report(file: UploadFile = File(...)):
    try:
        # Leer y codificar la imagen en base64
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode("utf-8")
        image_part = {
            "inlineData": {
                "mimeType": file.content_type,
                "data": base64_image
            }
        }

        # Enviar solicitud a Gemini
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{
                "parts": [
                    {"text": "Analiza esta imagen de un medidor eléctrico y proporciona los siguientes datos: tipo de medidor, cantidad de cables conectados, estado general visible y cualquier anomalía o intervención."},
                    image_part
                ]
            }]
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(GEMINI_URL, headers=headers, json=data)
            response.raise_for_status()
            gemini_data = response.json()

        # Extraer texto del resultado de Gemini
        output_text = gemini_data["candidates"][0]["content"]["parts"][0]["text"]
        size_kb = len(contents) / 1024

        return {
            "filename": file.filename,
            "size_kb": round(size_kb, 1),
            "message": output_text
        }

    except Exception as e:
        return {"error": str(e)}
