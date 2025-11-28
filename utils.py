import fitz
import tempfile
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from youtube_transcript_api import YouTubeTranscriptApi
from openai import OpenAI

client = OpenAI()

from config import (
    OPENAI_API_KEY,
    OPENAI_MODEL_SUMMARIZER,
    OPENAI_MODEL_EMBEDDING,
    OPENAI_MODEL_WHISPER,
)


def ef(file: Union[str, bytes]) -> Dict[str, Any]:
    if isinstance(file, str):
        doc = fitz.open(file)
    else:
        doc = fitz.open(stream=file, filetype="pdf")

    pages = []
    full = []
    for i, p in enumerate(doc):
        t = p.get_text("text")
        full.append(t)
        pages.append({"page": i + 1, "text": t})

    return {"text": "\n".join(full), "pages": pages, "metadata": {"page_count": len(doc)}}


def transcribe_audio(path=None, data=None, model=OPENAI_MODEL_WHISPER):
    if not path and not data:
        raise ValueError("Need path or data")

    tmp = None
    if data:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp.write(data)
        tmp.close()
        path = tmp.name

    with open(path, "rb") as f:
        r = client.audio.transcriptions.create(
            model=model,
            file=f
        )

    if tmp:
        Path(tmp.name).unlink(missing_ok=True)

    return r.text


# -----------------------------
# NEW YOUTUBE TRANSCRIPT FUNCTION
# -----------------------------
def transcribe_youtube(url: str):
    try:
        video_id = url.split("v=")[-1].split("&")[0]
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join(entry["text"] for entry in transcript_list)
        return {"transcript": text}

    except Exception:
        return {"transcript": "Could not fetch YouTube transcript."}


def translate_to_english(text: str):
    response = client.chat.completions.create(
        model=OPENAI_MODEL_SUMMARIZER,
        messages=[{
            "role": "user",
            "content": f"Translate this to English:\n\n{text}"
        }]
    )
    return response.choices[0].message.content.strip()


def chunk_text(text: str, size=1500, overlap=150):
    words = text.split()
    chunks = []
    step = size - overlap
    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i:i+size]))
    return chunks


def summarize_chunk(text: str, tone="neutral", limit=None, model=OPENAI_MODEL_SUMMARIZER):
    prompt = f"""
    Summarize the following text.

    Tone: {tone}
    Word limit: {limit if limit else "no limit"}

    Format EXACTLY like this (no JSON):

    Title: <title>

    • bullet 1
    • bullet 2
    • bullet 3

    Key Insights:
    - insight 1
    - insight 2
    - insight 3

    TLDR:
    <one-line summary>

    Text to summarize:
    {text}
    """

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()


def summarize_pipeline(text: str, tone="neutral", limit=None, to_english=False):

    if str(to_english).lower() == "true":
        text = translate_to_english(text)

    chunks = chunk_text(text)

    if len(chunks) == 1:
        single = summarize_chunk(chunks[0], tone=tone, limit=limit)
        return {"result": single}

    partial = []
    for chunk in chunks:
        partial.append(summarize_chunk(chunk, tone=tone, limit=limit))

    combined = "\n\n".join(partial)
    final_summary = summarize_chunk(combined, tone=tone, limit=limit)

    return {"result": final_summary}


def multimodal(source: str, data, tone="neutral", limit=None, to_english=False):
    if source == "pdf":
        t = ef(data)["text"]
        return summarize_pipeline(t, tone, limit, to_english)

    if source == "audio":
        t = transcribe_audio(data=data)
        return summarize_pipeline(t, tone, limit, to_english)

    if source == "youtube":
        t = transcribe_youtube(url=data)["transcript"]
        return summarize_pipeline(t, tone, limit, to_english)

    if source == "text":
        return summarize_pipeline(data, tone, limit, to_english)

    raise ValueError("Unsupported source")
