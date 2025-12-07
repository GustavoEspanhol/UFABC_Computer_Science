# backend/schemas.py
from pydantic import BaseModel
from typing import Any, Dict

class UserInput(BaseModel):
    nome: str
    idade: int
    signo: str
    genero_musical: str
    artista_favorito: str
    time_futebol: str
    cidade: str

class OracleOutput(BaseModel):
    combined_doc: str
    ner_json: str
    keywords_json: str
    classification_json: str
    prediction_json: str
    spacy_ner: Any = None
