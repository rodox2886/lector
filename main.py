from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/gemini-report")
async def gemini_report(file: UploadFile = File(...)):
    contents = await file.read()
    size_kb = round(len(contents) / 1024, 1)
    return {
        "filename": file.filename,
        "size_kb": size_kb,
        "message": "Imagen procesada correctamente."
    }
