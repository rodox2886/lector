import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import httpx

# Cargar variables de entorno del archivo .env
load_dotenv()

# Inicializar la aplicación FastAPI
app = FastAPI(
    title="Gemini Meter Reader Backend",
    description="Proxy para la API de Gemini Vision para leer medidores, protegiendo la clave de API."
)

# Configurar CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todos los orígenes para desarrollo. ¡Cambiar en producción!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Obtener la clave de API de Gemini de las variables de entorno
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
print(f"DEBUG: Valor de GEMINI_API_KEY leído: {GEMINI_API_KEY}")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

if not GEMINI_API_KEY:
    raise ValueError("La variable de entorno GEMINI_API_KEY no está configurada. Por favor, créala en tu archivo .env.")

# Modelos Pydantic para validar la estructura del cuerpo de la petición entrante
class PartInlineData(BaseModel):
    mimeType: str
    data: str

class Part(BaseModel):
    text: Optional[str] = None
    inlineData: Optional[PartInlineData] = None

class GeminiContent(BaseModel):
    role: str
    parts: List[Part]

class GeminiRequest(BaseModel):
    contents: List[GeminiContent]


# Endpoint para procesar la lectura del medidor con Gemini
@app.post("/api/gemini-report")
async def generate_gemini_report(gemini_request: GeminiRequest):
    """
    Envía las imágenes y el prompt a la API de Gemini para generar un informe.
    """
    print(f"Received request with {len(gemini_request.contents)} content blocks.")
    if gemini_request.contents:
        print(f"First content block has {len(gemini_request.contents[0].parts)} parts.")

    # Definir la estructura de la respuesta JSON que esperamos de Gemini
    generation_config_schema = {
        "type": "OBJECT",
        "properties": {
            "reportTitle": {"type": "STRING"},
            "meterId": {"type": "STRING", "nullable": True},
            "inspectionDate": {"type": "STRING", "nullable": True},
            "overallStatus": {"type": "STRING"},
            "visualObservations": {
                "type": "ARRAY",
                "items": {"type": "STRING"}
            },
            "readings": {
                "type": "OBJECT",
                "properties": {
                    "meterReading": {"type": "STRING", "nullable": True},
                    "unit": {"type": "STRING", "nullable": True},
                    "loadAtInspection": {"type": "STRING", "nullable": True}
                }
            },
            "anomaliesDetected": {
                "type": "ARRAY",
                "items": {"type": "STRING"}
            },
            "actionsTakenObserved": {
                "type": "ARRAY",
                "items": {"type": "STRING"}
            },
            "summary": {"type": "STRING"}
        },
        "propertyOrdering": [
            "reportTitle",
            "meterId",
            "inspectionDate",
            "overallStatus",
            "visualObservations",
            "readings",
            "anomaliesDetected",
            "actionsTakenObserved",
            "summary"
        ]
    }

    # === CORRECCIÓN AQUÍ: Convertir los modelos Pydantic a diccionarios ===
    # Usamos .model_dump() para convertir cada objeto GeminiContent a un diccionario
    # Esto es necesario para que httpx pueda serializarlo correctamente a JSON.
    serialized_contents = [content.model_dump(mode='json') for content in gemini_request.contents]

    # Payload completo para la API de Gemini
    gemini_payload = {
        "contents": serialized_contents, # Ahora esto es una lista de diccionarios
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": generation_config_schema
        }
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        # Usar httpx para hacer la petición HTTP asíncrona a la API de Gemini
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
                json=gemini_payload, # httpx ahora puede serializar esto correctamente
                headers=headers
            )
            response.raise_for_status()

            gemini_response = response.json()
            return gemini_response

    except httpx.HTTPStatusError as e:
        print(f"Error HTTP al llamar a Gemini: {e.response.status_code} - {e.response.text}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Error de la API de Gemini: {e.response.text}"
        )
    except httpx.RequestError as e:
        print(f"Error de red al llamar a Gemini: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error de red al conectar con la API de Gemini: {e}"
        )
    except Exception as e:
        print(f"Un error inesperado ocurrió: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error interno del servidor: {e}"
        )

# Ruta de prueba simple para verificar que el backend está funcionando
@app.get("/")
async def read_root():
    return {"message": "Gemini Meter Reader Backend is running!"}
