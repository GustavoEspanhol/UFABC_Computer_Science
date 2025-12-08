# frontend/streamlit_app.py
import streamlit as st
import requests
import os

BACKEND_URL = os.getenv("ORACLE_BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Oráculo AI", page_icon="🔮", layout="centered")

st.markdown(
    """
    <div style='text-align:center;'>
        <h1 style='font-family:serif; color: #5b2c6f;'>🔮 Oráculo AI</h1>
        <p style='color:#6c757d;'>Previsor Fictício do Futuro — entretenimento baseado em textos reais</p>
    </div>
    """,
    unsafe_allow_html=True
)

with st.form("oracle_form"):
    st.subheader("Diga-me sobre você")
    nome = st.text_input("Nome", value="Ari")
    idade = st.number_input("Idade", min_value=6, max_value=120, value=30)
    signo = st.text_input("Signo", value="Leão")
    genero = st.text_input("Gênero musical", value="MPB")
    artista = st.text_input("Artista favorito (Spotify)", value="Chico Buarque")
    time = st.text_input("Time de futebol", value="Flamengo")
    cidade = st.text_input("Cidade", value="Rio de Janeiro")
    submitted = st.form_submit_button("Gerar Meu Futuro 🔮")

if submitted:
    payload = {
        "nome": nome,
        "idade": idade,
        "signo": signo,
        "genero_musical": genero,
        "artista_favorito": artista,
        "time_futebol": time,
        "cidade": cidade
    }
    with st.spinner("Consultando o Oráculo..."):
        try:
            resp = requests.post(f"{BACKEND_URL}/generate", json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            pipeline = data.get("pipeline", {})
            st.success("Oráculo conjurado! ✨")
        except Exception as e:
            st.error(f"Erro ao contatar o backend: {e}")
            st.stop()

    st.markdown("## Perfil combinado (texto usado para análise)")
    st.write(pipeline.get("combined_doc", "—"))

    st.markdown("## Entidades nomeadas (extraídas pelo LLM)")
    st.code(pipeline.get("ner_json", ""), language="json")

    st.markdown("## Palavras-chave extraídas")
    st.code(pipeline.get("keywords_json", ""), language="json")

    st.markdown("## Classificação (taxonomia criativa)")
    st.code(pipeline.get("classification_json", ""), language="json")

    st.markdown("## Previsão Final (Oráculo Estocástico)")
    st.write(pipeline.get("prediction_json", ""))

    if pipeline.get("spacy_ner"):
        st.markdown("## NER extra (spaCy)")
        st.json(pipeline["spacy_ner"])

    st.markdown("---")
    st.info("Lembrete: esta é uma previsão **imaginária** e apenas para entretenimento.")
