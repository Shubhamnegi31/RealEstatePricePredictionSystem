#!/bin/bash
# Script to launch the Real Estate Price Prediction Web Application

echo "Starting Indian Real Estate Price Prediction Web Application..."
cd "$(dirname "$0")"

# Activate Virtual Environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Virtual environment not found. Creating..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt plotly
fi

# Run Streamlit Application
echo "Launching Streamlit server on http://localhost:8501..."
streamlit run app.py
