from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
import base64
import os
import json # Importar la librería json
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
    y retornar un informe consolidado en un formato estructurado (JSON).
    """
    if not files:
        return {"error": "No se han proporcionado archivos para el análisis.", "message": "No se han proporcionado archivos para el análisis."}

    image_parts = []
    total_size_kb = 0
    filenames = []

    for file in files:
        try:
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
        except Exception as e:
            return {"error": f"Error al procesar el archivo {file.filename}: {e}", "message": f"Error al procesar el archivo {file.filename}: {e}"}

    # El prompt se ajusta para solicitar un informe estructurado en JSON
    # Se especifica que la respuesta debe ser SOLO JSON y se incluye un ejemplo
    prompt_text = (
        "Analiza estas imágenes de un medidor eléctrico (pueden ser múltiples vistas del mismo medidor) "
        "y proporciona un informe detallado en formato JSON. Tu respuesta debe ser SOLO el objeto JSON, "
        "sin preámbulos ni texto adicional. El informe debe incluir los siguientes campos: "
        "'tipoMedidor', 'cablesConectados', 'estadoGeneralVisible', 'anomaliasDetectadas' (un array de strings, vacío si no hay), "
        "y 'conclusionGeneral'.\n\n"
        "Ejemplo de formato JSON esperado:\n"
        "```json\n"
        "{\n"
        "  \"tipoMedidor\": \"Digital\",\n"
        "  \"cablesConectados\": 3,\n"
        "  \"estadoGeneralVisible\": \"Buen estado\",\n"
        "  \"anomaliasDetectadas\": [\"Cable suelto\", \"Sellos rotos\"],\n"
        "  \"conclusionGeneral\": \"El medidor parece funcionar correctamente, pero requiere una inspección de cables.\"\n"
        "}\n"
        "```\n"
        "Genera el informe JSON ahora:"
    )

    # Definición del esquema JSON esperado para la respuesta de Gemini (esto es para la guía interna de Gemini,
    # pero la extracción robusta es nuestra principal defensa contra respuestas no conformes).
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "tipoMedidor": {"type": "STRING", "description": "Tipo de medidor eléctrico, ej. 'Monofásico', 'Trifásico', 'Digital', 'Analógico'."},
            "cablesConectados": {"type": "INTEGER", "description": "Cantidad de cables conectados al medidor."},
            "estadoGeneralVisible": {"type": "STRING", "description": "Descripción general del estado visible del medidor, ej. 'Buen estado', 'Desgaste leve', 'Deterioro visible'."},
            "anomaliasDetectadas": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
                "description": "Lista de cualquier anomalía, daño, manipulación o intervención detectada."
            },
            "conclusionGeneral": {"type": "STRING", "description": "Conclusión concisa sobre la lectura o el estado general del medidor."}
        },
        "required": ["tipoMedidor", "cablesConectados", "estadoGeneralVisible", "anomaliasDetectadas", "conclusionGeneral"]
    }

    # Preparar el contenido para la solicitud a Gemini, incluyendo el prompt y las imágenes
    gemini_contents = [{"text": prompt_text}] + image_parts

    # Cuerpo de la solicitud a la API de Gemini, con la configuración de respuesta estructurada
    data = {
        "contents": [{
            "parts": gemini_contents
        }],
        "generationConfig": {
            "responseMimeType": "application/json", # Intentamos forzar JSON
            "responseSchema": response_schema
        }
    }

    headers = {"Content-Type": "application/json"}

    try:
        # Aumentar el tiempo de espera para solicitudes a Gemini
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(GEMINI_URL, headers=headers, json=data)
            response.raise_for_status()

            gemini_data = response.json()

        # Extraer el texto generado por Gemini
        if gemini_data and "candidates" in gemini_data and gemini_data["candidates"]:
            raw_gemini_text = gemini_data["candidates"][0]["content"]["parts"][0]["text"]
            
            # Intentar extraer el JSON de la cadena completa
            # Buscar el primer '{' y el último '}'
            start_idx = raw_gemini_text.find('{')
            end_idx = raw_gemini_text.rfind('}')

            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                json_string = raw_gemini_text[start_idx : end_idx + 1]
                output_report = json.loads(json_string) # Parsear la cadena JSON a un objeto Python
            else:
                # Si no se encuentra un JSON válido, reportar un error
                error_msg = "La respuesta de Gemini no contiene un JSON válido. Respuesta cruda: " + raw_gemini_text
                return {"error": error_msg, "message": error_msg}
        else:
            output_report = {"error": "Gemini no pudo generar un informe o la respuesta fue inesperada.", "message": "Gemini no pudo generar un informe o la respuesta fue inesperada."}

        return {
            "filenames": filenames,
            "total_size_kb": round(total_size_kb, 1),
            "message": output_report # Ahora 'message' contiene el objeto JSON del informe
        }

    except httpx.RequestError as e:
        error_msg = f"Error de conexión con la API de Gemini: {e}"
        return {"error": error_msg, "message": error_msg}
    except httpx.HTTPStatusError as e:
        try:
            error_detail = e.response.json()
            error_msg = f"Error de la API de Gemini ({e.response.status_code}): {error_detail.get('error', {}).get('message', 'Mensaje de error no disponible')}"
        except ValueError:
            error_msg = f"Error de la API de Gemini ({e.response.status_code}): {e.response.text}"
        return {"error": error_msg, "message": error_msg}
    except json.JSONDecodeError as e:
        # Este error ahora es menos probable si la extracción robusta funciona,
        # pero es bueno mantenerlo.
        error_msg = f"Error al parsear el JSON extraído de Gemini: {e}. Asegúrate de que el formato JSON sea válido."
        # Aquí sería útil loggear `raw_gemini_text` o `json_string` para depuración
        return {"error": error_msg, "message": error_msg}
    except Exception as e:
        error_msg = f"Ocurrió un error inesperado en el servidor: {e}"
        return {"error": error_msg, "message": error_msg}
