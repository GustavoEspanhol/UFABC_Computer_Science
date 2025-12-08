# backend/app.py
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from spotify_client import SpotifyClient
from wikipedia_client import fetch_wikipedia_summary
from langchain_oracle import OraclePipeline

app = FastAPI(title="Oráculo Estocástico - Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load credentials from env
SPOTIFY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # or other LLM key used by LangChain

if not OPENAI_API_KEY:
    # allow running with other LLM configurations but warn
    print("WARNING: OPENAI_API_KEY not set. Configure your LLM provider or set OPENAI_API_KEY.")

spotify = SpotifyClient(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET)
oracle = OraclePipeline()

class UserInput(BaseModel):
    nome: str
    idade: int
    signo: str
    genero_musical: str
    artista_favorito: str
    time_futebol: str
    cidade: str

@app.post("/generate")
async def generate(user: UserInput):
    try:
        # 1) Fetch Wikipedia summaries
        wiki_signo = fetch_wikipedia_summary(user.signo)
        wiki_time = fetch_wikipedia_summary(user.time_futebol)
        wiki_cidade = fetch_wikipedia_summary(user.cidade)

        # 2) Fetch Spotify artist info
        artist_info = spotify.get_artist_info(user.artista_favorito)  # returns dict with name, genres, description/overview
        # If artist not found, returns minimal info

        # 3) Compose documents and send to LangChain pipeline
        inputs = {
            "user": user.dict(),
            "wiki_signo": wiki_signo,
            "wiki_time": wiki_time,
            "wiki_cidade": wiki_cidade,
            "artist_info": artist_info,
        }

        result = oracle.run_pipeline(inputs)

        # 4) Return structured result to frontend
        return {
            "status": "ok",
            "inputs": inputs,
            "pipeline": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Backend do Oráculo Estocástico está funcionando!"}