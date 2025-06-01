import whisper
import os
from typing import Optional

_model = None

def load_model():
    global _model
    if _model is None:
        model_name = os.getenv("WHISPER_MODEL", "tiny")
        print(f"Loading Whisper model: {model_name}")
        _model = whisper.load_model(model_name)
    return _model

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