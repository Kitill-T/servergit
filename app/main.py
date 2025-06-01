from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from pathlib import Path
import time

app = FastAPI(title="Audio Processing API")

# CORS настройки
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    print("Server starting...")


@app.post("/api/process_audio")
async def process_audio(file: UploadFile):
    try:
        # Проверка размера файла (макс 5MB)
        if file.size > 5 * 1024 * 1024:
            raise HTTPException(413, "File too large (max 5MB)")

        # Сохраняем временный файл
        temp_dir = Path("/tmp/audio_uploads")
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / f"audio_{int(time.time())}.wav"

        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # Ленивая загрузка модулей для экономии памяти
        from app.asr.whisper import transcribe_audio
        from app.llm.openai_client import generate_response

        text = transcribe_audio(str(temp_path))
        response = generate_response(text)

        # Удаляем временный файл
        temp_path.unlink()

        return {
            "status": "success",
            "transcription": text,
            "ai_response": response
        }

    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/health")
async def health_check():
    return {"status": "ok", "environment": "codespaces"}