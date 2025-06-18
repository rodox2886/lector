from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import base64
import httpx
import os

app = FastAPI()

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://rodox2886.github.io"],  # Dominio del frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Gemini Meter Reader Backend is running!"}

@app.post("/api/gemini-report")
async def gemini_report(files: List[UploadFile] = File(...)):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="API key not configured")

    # Leer y codificar las imágenes
    contents = []
    for file in files:
        data = await file.read()
        b64 = base64.b64encode(data).decode("utf-8")
        contents.append(f"<img src='data:image/jpeg;base64,{b64}'>")

    prompt = """
Analiza las siguientes imágenes de un medidor de energía y genera un informe completo.
Indica el tipo de medidor, cantidad de medidores, estado físico, números legibles,
acciones observadas (por ejemplo: tapas abiertas, elementos manipulados) y una
conclusión breve.

Responde con etiquetas y formato claro, sin explicaciones adicionales.
    """

    full_content = prompt + "\n".join(contents)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    body = {
        "contents": [
            {
                "parts": [{"text": full_content}]
            }
        ]
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-vision:generateContent",
                headers=headers,
                json=body
            )
        response.raise_for_status()
        result = response.json()
        output = result["candidates"][0]["content"]["parts"][0]["text"]
        return {"resultado": output}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar con Gemini: {str(e)}")
