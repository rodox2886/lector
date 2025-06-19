from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
import base64
import os
from typing import List

app = FastAPI()

# Configurar CORS para permitir solicitudes desde cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todos los orígenes
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permite todos los encabezados
)

# Obtener API Key de la variable de entorno. Asegúrate de que GEMINI_API_KEY esté configurada en tu entorno de Render.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("La variable de entorno GEMINI_API_KEY no está configurada.")

# URL del endpoint de Gemini 1.5 Flash
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=" + GEMINI_API_KEY

@app.post("/api/gemini-report")
async def generate_report(files: List[UploadFile] = File(...)):
    """
    Endpoint para recibir múltiples imágenes de un medidor, enviarlas a Gemini
    y retornar un informe consolidado.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No se han proporcionado archivos.")

    image_parts = []
    total_size_kb = 0
    filenames = []

    # Procesar cada archivo de imagen
    for file in files:
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode("utf-8")
        image_parts.append(
            {
                "inlineData": {
                    "mimeType": file.content_type,
                    "data": base64_image
                }
            }
        )
        total_size_kb += len(contents) / 1024
        filenames.append(file.filename)

    # El prompt se ajusta para indicar que hay múltiples imágenes del mismo medidor
    prompt_text = "Analiza estas imágenes de un medidor eléctrico (pueden ser múltiples vistas del mismo medidor) y proporciona los siguientes datos en un informe conciso: tipo de medidor, cantidad de cables conectados, estado general visible, cualquier anomalía o intervención detectada, y una conclusión sobre la lectura o el estado general del medidor basándose en todas las imágenes. Enumera los hallazgos claramente."

    # Preparar el contenido para la solicitud a Gemini
    # La primera parte es el texto del prompt, seguido de todas las partes de la imagen.
    gemini_contents = [{"text": prompt_text}] + image_parts

    # Cuerpo de la solicitud a la API de Gemini
    data = {
        "contents": [{
            "parts": gemini_contents
        }]
    }

    headers = {"Content-Type": "application/json"}

    try:
        # Enviar la solicitud a Gemini usando httpx para solicitudes asincrónicas
        async with httpx.AsyncClient(timeout=60.0) as client: # Aumentar el tiempo de espera si es necesario
            response = await client.post(GEMINI_URL, headers=headers, json=data)
            response.raise_for_status()  # Lanza una excepción para respuestas 4xx/5xx

            gemini_data = response.json()

        # Extraer el texto generado por Gemini
        if gemini_data and "candidates" in gemini_data and gemini_data["candidates"]:
            output_text = gemini_data["candidates"][0]["content"]["parts"][0]["text"]
        else:
            output_text = "Gemini no pudo generar un informe con las imágenes proporcionadas."

        return {
            "filenames": filenames,
            "total_size_kb": round(total_size_kb, 1),
            "message": output_text
        }

    except httpx.RequestError as e:
        # Manejo de errores de conexión o solicitud
        raise HTTPException(status_code=500, detail=f"Error de conexión con la API de Gemini: {e}")
    except httpx.HTTPStatusError as e:
        # Manejo de errores de estado HTTP (ej. 400, 500 del lado de Gemini)
        error_detail = e.response.json() if e.response.content else str(e)
        raise HTTPException(status_code=e.response.status_code, detail=f"Error de la API de Gemini: {error_detail}")
    except Exception as e:
        # Manejo de cualquier otra excepción inesperada
        raise HTTPException(status_code=500, detail=f"Ocurrió un error inesperado: {e}")
