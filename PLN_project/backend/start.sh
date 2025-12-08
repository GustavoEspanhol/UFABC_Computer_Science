#!/bin/bash

uvicorn app:app --reload --host 0.0.0.0 --port 8000
cd ../frontend
streamlit run streamlit_app.py  
