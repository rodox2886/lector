from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from typing import List
import base64
import os
import httpx
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Middleware para archivos grandes (hasta 50 MB)
class LimitUploadSizeMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_upload_size: int):
        super().__init__(app)
        self.max_upload_size = max_upload_size

    async def dispatch(self, request: Request, call_next):
        if request.headers.get("content-length"):
            if int(request.headers["content-length"]) > self.max_upload_size:
                return Response("Payload too large", status_code=413)
        return await call_next(request)

app.add_middleware(LimitUploadSizeMiddleware, max_upload_size=50 * 1024 * 1024)

# CORS para permitir peticiones desde GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://rodox2886.github.io"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ruta principal para procesar las imágenes
@app.post("/api/gemini-report")
async def generate_gemini_report(files: List[UploadFile] = File(...)):
    try:
        images_base64 = []
        for file in files:
            content = await file.read()
            encoded = base64.b64encode(content).decode("utf-8")
            images_base64.append(f"data:{file.content_type};base64,{encoded}")

        prompt = (
            "Analiza estas imágenes de un medidor eléctrico. Genera un informe con los siguientes campos:\n"
            "1. Estado general del medidor\n"
            "2. Observaciones visuales (si hay anomalías)\n"
            "3. Anomalías detectadas (ejemplo: medidor roto, faltante de tapa, cables expuestos)\n"
            "4. Acciones observadas del operario (por ejemplo si intervino, fotografió, manipuló, etc)\n"
            "5. Resumen general.\n\n"
            "Si hay varias imágenes, correlaciona los datos entre ellas. Sé específico. El resultado debe estar en formato JSON."
        )

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="Falta la API key de Gemini")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        *[{"inline_data": {"mime_type": "image/jpeg", "data": img}} for img in images_base64]
                    ]
                }
            ]
        }

        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-vision:generateContent"

        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload)

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Error de Gemini: {response.text}")

        data = response.json()
        output = data["candidates"][0]["content"]["parts"][0]["text"]
        return {"resultado": output}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
