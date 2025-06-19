from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()

# CORS para permitir acceso desde GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # podés restringir a tu GitHub Pages si querés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/gemini-report")
async def generate_report(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        # Simulación del análisis con Gemini
        return {
            "filename": file.filename,
            "size_kb": round(len(contents) / 1024, 2),
            "message": "Imagen procesada correctamente."
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

