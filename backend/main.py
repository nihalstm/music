import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# -----------------------------
# FastAPI App Setup
# -----------------------------
app = FastAPI(title="GitHub GPT Mood Classifier")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # allow all for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Load ENV Variables
# -----------------------------
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise RuntimeError("GITHUB_TOKEN not set in environment")

# GitHub GPT endpoint
GITHUB_GPT_ENDPOINT = "https://models.github.ai/inference"


# -----------------------------
# Pydantic Models
# -----------------------------
class UserText(BaseModel):
    text: str

class Recommendation(BaseModel):
    mood: str
    track_name: str = None
    artist: str = None
    preview_url: str = None


# -----------------------------
# Root Route
# -----------------------------
@app.get("/")
def home():
    return {"message": "Backend running"}


# -----------------------------
# GitHub GPT Mood Classifier
# -----------------------------
def classify_mood_github_gpt(text: str) -> str:
    prompt = f"""
You are a mood classifier. Map the following user input to exactly one of these moods:
Calm, Energizing, Focus, Uplifting.
User input: '{text}'
Respond ONLY with the mood in one word (Calm, Energizing, Focus, or Uplifting). No punctuation.
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
        response = requests.post(
            f"{GITHUB_GPT_ENDPOINT}/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()

        result = response.json()
        output_text = result['choices'][0]['message']['content'].strip().split()[0].capitalize()

        if output_text in ["Calm", "Energizing", "Focus", "Uplifting"]:
            return output_text
        return "Uplifting"

    except Exception as e:
        print("GitHub GPT API error:", e)
        return "Uplifting"


# -----------------------------
# Analyze Endpoint
# -----------------------------
@app.post("/analyze", response_model=Recommendation)
def analyze_text(user_text: UserText):
    text = user_text.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    mood = classify_mood_github_gpt(text)

    # Dummy music recommendations
    song_recommendation = {
        "Calm": {
            "track_name": "Weightless",
            "artist": "Marconi Union",
            "preview_url": "https://p.scdn.co/mp3-preview/..."
        },
        "Energizing": {
            "track_name": "Great",
            "artist": "Pharrell Williams",
            "preview_url": "https://p.scdn.co/mp3-preview/..."
        },
        "Focus": {
            "track_name": "Breathe",
            "artist": "The Cinematic Orchestra",
            "preview_url": "https://p.scdn.co/mp3-preview/..."
        },
        "Uplifting": {
            "track_name": "Best Day of My Life",
            "artist": "American Authors",
            "preview_url": "https://p.scdn.co/mp3-preview/..."
        }
    }

    track_info = song_recommendation.get(mood, {})

    return Recommendation(
        mood=mood,
        track_name=track_info.get("track_name"),
        artist=track_info.get("artist"),
        preview_url=track_info.get("preview_url")
    )


# -----------------------------
# Railway / Render Entry Point
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port)
