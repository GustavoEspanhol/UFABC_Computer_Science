#!/bin/bash

# Dá permissão de execução ao script (não é necessário incluir isso no próprio script)
# chmod +x start.sh

# Inicia o backend (FastAPI) com uvicorn em segundo plano
uvicorn app:app --reload --host 0.0.0.0 --port 8080 &

# Navega para o diretório do frontend
cd ../frontend

# Inicia o frontend (Streamlit)
streamlit run streamlit_app.py