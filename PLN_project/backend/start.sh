#!/bin/bash

# Dá permissão de execução ao script (não é necessário incluir isso no próprio script)
# chmod +x start.sh

# Inicia o backend (FastAPI) com uvicorn em segundo plano
python3 app.py &

# Navega para o diretório do frontend
cd ../frontend

# Inicia o frontend (Streamlit)
streamlit run streamlit_app.py