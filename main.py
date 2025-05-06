from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import whisper
import tempfile
import requests
import base64
import os
from dotenv import load_dotenv
from pathlib import Path

# Load env vars
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = FastAPI(
    title="Groq Video Summarizer API",
    description="Summarize and analyze video content using Whisper + Groq LLaMA3",
    version="1.0"
)

# Enable CORS if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Whisper model once
whisper_model = whisper.load_model("base")

def transcribe_audio(video_path):
    result = whisper_model.transcribe(video_path)
    return result["text"]

def query_groq(prompt, model="llama3-8b-8192"):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant for summarizing video content."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"Groq API Error: {response.status_code} - {response.text}"

# --- AGENT LOGIC ---

def web_search(query):
    return f"https://www.duckduckgo.com/?q={query}"

def fact_checking(query):
    return f"https://www.duckduckgo.com/?q={query}"

class VideoAgent:
    def __init__(self, video_path, user_query):
        self.video_path = video_path
        self.user_query = user_query
        self.transcript = transcribe_audio(video_path)

    def summarize_video(self):
        prompt = (
            f"Here is the transcript of a video:\n\n{self.transcript}\n\n"
            f"Now respond to the following request:\n{self.user_query}"
        )
        return query_groq(prompt)

    def search_web(self):
        return web_search(self.user_query)

    def fact_check(self):
        return fact_checking(self.user_query)

    def process_query(self):
        user_query_lower = self.user_query.lower()
        if any(keyword in user_query_lower for keyword in ["summarize", "key points", "summary", "main ideas"]):
            return {"type": "summary", "result": self.summarize_video()}
        elif any(keyword in user_query_lower for keyword in ["search", "find more", "look up", "additional info"]):
            return {"type": "web_search", "result": self.search_web()}
        elif any(keyword in user_query_lower for keyword in ["fact-check", "verify", "is this true", "check"]):
            return {"type": "fact_check", "result": self.fact_check()}
        else:
            return {"type": "unknown", "result": "Sorry, I couldn't understand the query."}

# --- FastAPI Endpoints ---

@app.post("/transcribe")
async def transcribe_video(video: UploadFile):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
            temp.write(await video.read())
            temp_path = temp.name

        transcript = transcribe_audio(temp_path)
        Path(temp_path).unlink(missing_ok=True)

        return {"transcript": transcript}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/analyze")
async def analyze_video(video: UploadFile, user_query: str = Form(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
            temp.write(await video.read())
            video_path = temp.name

        agent = VideoAgent(video_path, user_query)
        result = agent.process_query()

        Path(video_path).unlink(missing_ok=True)

        return {"analysis": result}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
