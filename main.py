
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from dotenv import load_dotenv
import os
import httpx

load_dotenv()

app = FastAPI()

# Permitir CORS desde cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/gemini-report")
async def generate_report(files: list[UploadFile] = File(...)):
    import base64
    import json

    try:
        encoded_images = []
        for file in files:
            content = await file.read()
            encoded = base64.b64encode(content).decode("utf-8")
            encoded_images.append(f"data:{file.content_type};base64,{encoded}")

        prompt = (
            "Analiza las siguientes imágenes de un medidor eléctrico. "
            "Devuelve un informe con los siguientes campos separados por secciones claras:\n"
            "- Estado general\n- Observaciones visuales\n- Anomalías detectadas\n- Acciones observadas\n- Resumen final\n"
            "Responde en español de forma clara y profesional.\n"
        )

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="API Key no configurada")

        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        body = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    *[
                        {"inline_data": {"mime_type": file.content_type, "data": base64.b64encode(await file.read()).decode()}}
                        for file in files
                    ]
                ]
            }]
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://generativelanguage.googleapis.com/v1/models/gemini-pro-vision:generateContent",
                headers=headers,
                json=body
            )

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Error al llamar a la API de Gemini")

        result = response.json()
        text = result["candidates"][0]["content"]["parts"][0]["text"]
        return {"response": text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar: {str(e)}")
