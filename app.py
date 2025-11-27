import os
import tempfile
import logging
import subprocess
import shlex
import uuid
import shutil
import threading
from typing import List, Dict, Union, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import librosa

# ------------------ Logging ------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("speech-analyzer")

# ------------------ Config ------------------
FFMPEG_PATH = shutil.which("ffmpeg")
if FFMPEG_PATH:
    logger.info(f"ffmpeg found at: {FFMPEG_PATH}")
else:
    logger.warning("ffmpeg not found on PATH. Audio conversion will fail until ffmpeg is installed.")

# ------------------ Simple vocab lists ------------------
FILLERS = ["um", "uh", "like", "you know", "so", "actually", "basically"]
POSITIVE_WORDS = ["good", "great", "amazing", "happy", "love", "excellent", "strong", "positive", "best"]
NEGATIVE_WORDS = ["bad", "sad", "angry", "hate", "terrible", "poor", "worst", "negative", "awful"]

# ------------------ FastAPI app ------------------
app = FastAPI(
    title="Interview Coach AI - Speech Analyzer",
    description="Upload an audio file to get a full speech transcription and analysis report."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in production restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Serve static frontend ------------------
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
INDEX_AT_ROOT = os.path.join(os.path.dirname(__file__), "index.html")

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    logger.info(f"Mounted static files from: {STATIC_DIR}")
else:
    logger.warning(f"Static directory not found at {STATIC_DIR}.")

@app.get("/", include_in_schema=False)
async def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    if os.path.exists(INDEX_AT_ROOT):
        return FileResponse(INDEX_AT_ROOT)
    return RedirectResponse(url="/docs")

@app.get("/ping")
async def ping():
    return JSONResponse({"status": "ok"})

# ------------------ Lazy Whisper loader ------------------
_model_lock = threading.Lock()
_model: Optional[object] = None
_MODEL_NAME = os.environ.get("MODEL_NAME", "tiny")  # default to tiny for reliability

def get_model():
    """
    Lazily import whisper and load the model on first call.
    Thread-safe: only one thread will load the model.
    """
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:
            return _model
        try:
            import whisper as _whisper_mod
        except Exception as e:
            logger.exception("Failed to import whisper module.")
            raise RuntimeError("Whisper import failed") from e

        logger.info(f"Lazy-loading Whisper model: {_MODEL_NAME}")
        try:
            _model = _whisper_mod.load_model(_MODEL_NAME)
            logger.info(f"✅ Whisper '{_MODEL_NAME}' model loaded successfully (lazy).")
        except Exception:
            logger.exception("Failed to load Whisper model lazily.")
            _model = None
            raise

    return _model

# ------------------ Pydantic models ------------------
class FillerAnalysis(BaseModel):
    fillers_found: List[str]
    filler_count: int

class ToneAnalysis(BaseModel):
    tone: str
    positive_score: int
    negative_score: int

class PaceAnalysis(BaseModel):
    duration_sec: float
    words: int
    pace_wpm: float

class AnalysisResponse(BaseModel):
    transcription: str
    filler_analysis: FillerAnalysis
    tone_analysis: ToneAnalysis
    pace_analysis: Union[PaceAnalysis, Dict]

# ------------------ Helper functions ------------------
def speech_to_text(audio_path: str) -> str:
    model_local = get_model()
    if not model_local:
        raise RuntimeError("Whisper model is not loaded.")
    result = model_local.transcribe(audio_path, fp16=False)
    return result.get("text", "").strip()

def filler_detector(text: str) -> FillerAnalysis:
    text_lower = text.lower()
    detected = [word for word in FILLERS if word in text_lower]
    count = sum(text_lower.count(f) for f in FILLERS)
    return FillerAnalysis(fillers_found=detected, filler_count=count)

def tone_analyzer(text: str) -> ToneAnalysis:
    text_lower = text.lower()
    pos_score = sum(text_lower.count(w) for w in POSITIVE_WORDS)
    neg_score = sum(text_lower.count(w) for w in NEGATIVE_WORDS)
    tone = "Neutral"
    if pos_score > neg_score:
        tone = "Positive"
    elif neg_score > pos_score:
        tone = "Negative"
    return ToneAnalysis(tone=tone, positive_score=pos_score, negative_score=neg_score)

def pace_calculator(audio_path: str, text: str) -> Union[PaceAnalysis, Dict]:
    try:
        duration = librosa.get_duration(path=audio_path)
        words = len(text.split())
        pace_wpm = round(words / (duration / 60), 2) if duration > 0 else 0.0
        return PaceAnalysis(duration_sec=round(duration, 2), words=words, pace_wpm=pace_wpm)
    except Exception:
        logger.exception("Pace calculation failed")
        return {"error": "Failed to compute pace"}

# ------------------ Endpoints ------------------
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_audio(file: UploadFile = File(...)):
    # Ensure ffmpeg available
    if not FFMPEG_PATH:
        raise HTTPException(
            status_code=500,
            detail=(
                "ffmpeg executable not found on the server. "
                "Install ffmpeg and add it to PATH (see server logs)."
            ),
        )

    # Try to ensure import will work — do NOT eagerly load model here.
    try:
        # test import of whisper only; actual load happens inside speech_to_text/get_model
        import importlib
        importlib.import_module("whisper")
    except Exception:
        logger.exception("Whisper import failed at request time.")
        raise HTTPException(status_code=503, detail="Whisper import failed on server; check logs.")

    orig_ext = os.path.splitext(file.filename)[1] or ".webm"
    tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=orig_ext)
    tmp_in_path = tmp_in.name
    tmp_out_path = os.path.join(tempfile.gettempdir(), f"conv_{uuid.uuid4().hex}.wav")

    try:
        contents = await file.read()
        tmp_in.write(contents)
        tmp_in.flush()
        tmp_in.close()
        logger.info(f"Saved upload to {tmp_in_path}")

        cmd = [
            FFMPEG_PATH,
            "-y",
            "-i", tmp_in_path,
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            tmp_out_path,
        ]
        logger.info("Running ffmpeg: " + " ".join(shlex.quote(p) for p in cmd))
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if proc.returncode != 0 or not os.path.exists(tmp_out_path):
            logger.error(f"ffmpeg failed; stderr: {proc.stderr}")
            last_line = proc.stderr.splitlines()[-1] if proc.stderr else "ffmpeg error"
            raise HTTPException(status_code=500, detail=f"Audio conversion failed: {last_line}")

        logger.info(f"Converted audio saved to {tmp_out_path}")

        # This will call get_model() and may trigger the lazy load (download + load).
        try:
            text = speech_to_text(tmp_out_path)
        except Exception as e:
            logger.exception("Transcription failed (model load or inference).")
            raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

        fillers = filler_detector(text)
        tone = tone_analyzer(text)
        pace = pace_calculator(tmp_out_path, text)

        return AnalysisResponse(
            transcription=text,
            filler_analysis=fillers,
            tone_analysis=tone,
            pace_analysis=pace,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error during analysis")
        raise HTTPException(status_code=500, detail=f"Analysis Failed: {str(e)}")
    finally:
        for p in (tmp_in_path, tmp_out_path):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
                    logger.info(f"Removed temp file {p}")
            except Exception as cleanup_err:
                logger.warning(f"Failed to remove {p}: {cleanup_err}")

# ------------------ Run server (local dev) ------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=(os.environ.get("DEV","false").lower()=="true"))
