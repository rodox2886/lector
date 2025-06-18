from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from dotenv import load_dotenv
import os
import base64
import httpx

load_dotenv()

app = FastAPI()

# Middleware para límite de tamaño
class LimitUploadSizeMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_upload_size: int):
        super().__init__(app)
        self.max_upload_size = max_upload_size

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get('content-length')
        if content_length and int(content_length) > self.max_upload_size:
            return Response("Payload too large", status_code=413)
        return await call_next(request)

app.add_middleware(LimitUploadSizeMiddleware, max_upload_size=10 * 1024 * 1024)  # 10MB

# CORS para conectar con GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoint principal
@app.post("/api/gemini-report")
async def gemini_report(files: list[UploadFile] = File(...)):
    api_key = os.getenv("GEMINI_API_KEY")
    endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-vision:generateContent?key=" + api_key

    # Crear payload de solicitud a Gemini
    images_base64 = []
    for file in files:
        content = await file.read()
        encoded = base64.b64encode(content).decode("utf-8")
        images_base64.append({
            "inlineData": {
                "mimeType": file.content_type,
                "data": encoded
            }
        })

    prompt = {
        "contents": [{
            "parts": [
                {"text": "Analiza estas imágenes de un medidor de energía. Extrae los datos visibles como número de cliente, lectura, tipo de medidor, estado físico, anomalías o manipulaciones."},
                *images_base64
            ]
        }]
    }

    # Enviar solicitud a Gemini
    async with httpx.AsyncClient() as client:
        response = await client.post(endpoint, json=prompt)

    if response.status_code != 200:
        return {"error": f"Gemini API error: {response.status_code}", "detail": response.text}

    data = response.json()
    try:
        output = data['candidates'][0]['content']['parts'][0]['text']
    except Exception:
        return {"error": "Respuesta inesperada de Gemini", "raw": data}

    return {"respuesta": output}
