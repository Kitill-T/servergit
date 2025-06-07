import whisper
model = whisper.load_model("tiny")
import os
from typing import Optional

def load_model():
    global model
    if model is None:
        model_name = os.getenv("WHISPER_MODEL", "tiny")
        print(f"Loading Whisper model: {model_name}")
        model = whisper.load_model(model_name)
    return model

def transcribe_audio(audio_path: str) -> Optional[str]:
    try:
        model = load_model()
        result = model.transcribe(
            audio_path,
            language="ru",
            fp16=False,  # Для совместимости с CPU
            verbose=False
        )
        return result["text"]
    except Exception as e:
        print(f"Transcription error: {str(e)}")
        return None