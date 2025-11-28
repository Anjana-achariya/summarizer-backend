from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from utils import multimodal

app = FastAPI()
@app.get("/")
def home():
    return {
        "status": "OK",
        "message": "GlowBrief Summarizer Backend is running 🚀"
    }


app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://summarizer-frontend-fw54.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/summarize/pdf")
async def summarize_pdf(
    file: UploadFile = File(...),
    tone: str = Form("neutral"),
    limit: int = Form(None),
    to_english: bool = Form(False, convert_underscores=False)
):
    content = await file.read()
    r = await asyncio.to_thread(multimodal, "pdf", content, tone, limit, to_english)
    return JSONResponse(r)                      # <-- FIXED

@app.post("/summarize/audio")
async def summarize_audio(
    file: UploadFile = File(...),
    tone: str = Form("neutral"),
    limit: int = Form(None),
    to_english: bool = Form(False, convert_underscores=False)
):
    ALLOWED_AUDIO_TYPES = [
        "audio/mpeg",   # .mp3
        "audio/mp3",
        "audio/wav",
        "audio/x-wav",
        "audio/ogg",
        "audio/webm",
        "audio/aac",
        "audio/m4a",
        "audio/x-m4a",
        "audio/mp4",    
    ]

    if file.content_type not in ALLOWED_AUDIO_TYPES:
        return JSONResponse(
            {"error": f"Unsupported audio type: {file.content_type}"},
            status_code=400
        )

    content = await file.read()
    r = await asyncio.to_thread(multimodal, "audio", content, tone, limit, to_english)
    return JSONResponse(r)

@app.post("/summarize/youtube")
async def summarize_youtube(
    url: str = Form(...),
    tone: str = Form("neutral"),
    limit: int = Form(None),
    to_english: bool = Form(False, convert_underscores=False)
):
    r = await asyncio.to_thread(multimodal, "youtube", url, tone, limit, to_english)
    return JSONResponse(r)                      # <-- FIXED

@app.post("/summarize/text")
async def summarize_text(
    text: str = Form(...),
    tone: str = Form("neutral"),
    limit: int = Form(None),
    to_english: bool = Form(False, convert_underscores=False)
):
    r = await asyncio.to_thread(multimodal, "text", text, tone, limit, to_english)
    return JSONResponse(r)                      # <-- FIXED


