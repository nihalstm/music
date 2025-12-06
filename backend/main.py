import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
# Add at the top after your imports
from fastapi.middleware.cors import CORSMiddleware

# After initializing FastAPI
app = FastAPI(title="GitHub GPT Mood Classifier")
from fastapi.staticfiles import StaticFiles

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

# Allow your frontend to call the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:5500"] if you want to restrict
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load .env
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise RuntimeError("GITHUB_TOKEN not set in environment")

# GitHub-hosted GPT endpoint
GITHUB_GPT_ENDPOINT = "https://models.github.ai/inference"


# Request / Response models
class UserText(BaseModel):
    text: str

class Recommendation(BaseModel):
    mood: str
    track_name: str = None
    artist: str = None
    preview_url: str = None

@app.get("/")
def root():
    return {"message": "Backend running"}

# ------------------------
# GitHub GPT call
# ------------------------
def classify_mood_github_gpt(text: str) -> str:
    """
    Send a prompt to GitHub-hosted GPT to classify mood.
    """
    prompt = f"""
You are a mood classifier. Map the following user input to exactly one of these moods:
Calm, Energizing, Focus, Uplifting.
User input: '{text}'
Respond ONLY with the mood in one word (Calm, Energizing, Focus, or Uplifting). No punctuation or extra words.
"""

    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "openai/gpt-4o",
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "max_tokens": 50
    }

    try:
        response = requests.post(f"{GITHUB_GPT_ENDPOINT}/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        # Extract text
        output_text = result['choices'][0]['message']['content'].strip().split()[0].capitalize()
        if output_text in ["Calm", "Energizing", "Focus", "Uplifting"]:
            return output_text
        return "Uplifting"
    except Exception as e:
        print("GitHub GPT API error:", e)
        return "Uplifting"

# ------------------------
# Analyze endpoint
# ------------------------
@app.post("/analyze", response_model=Recommendation)
def analyze_text(user_text: UserText):
    text = user_text.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    mood = classify_mood_github_gpt(text)

    # Map moods to dummy songs
    song_recommendation = {
        "Calm": {"track_name": "Weightless", "artist": "Marconi Union", "preview_url": "https://p.scdn.co/mp3-preview/..."},
        "Energizing": {"track_name": "Great", "artist": "Pharrell Williams", "preview_url": "https://p.scdn.co/mp3-preview/..."},
        "Focus": {"track_name": "Breathe", "artist": "The Cinematic Orchestra", "preview_url": "https://p.scdn.co/mp3-preview/..."},
        "Uplifting": {"track_name": "Best Day of My Life", "artist": "American Authors", "preview_url": "https://p.scdn.co/mp3-preview/..."}
    }

    track_info = song_recommendation.get(mood, {})

    return Recommendation(
        mood=mood,
        track_name=track_info.get("track_name"),
        artist=track_info.get("artist"),
        preview_url=track_info.get("preview_url")
    )
