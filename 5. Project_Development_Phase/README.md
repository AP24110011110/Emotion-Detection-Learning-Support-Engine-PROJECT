# AI-Driven Emotion Detection & Personalized Learning Support Platform

This project builds a student support system that accepts free-text learning problems, detects emotion with BiLSTM and BERT, enhances predictions with keyword context, and generates empathetic responses via Google Gemini.

## Structure
- `app.py`: Streamlit application entrypoint
- `requirements.txt`: Python dependencies
- `.env`: API keys and path configuration
- `models/`: exported model artifacts and tokenizer
- `training/`: training scripts and dataset utilities
- `pipeline/`: inference pipeline, preprocessing, and prediction schema
- `gemini/`: prompt builder and Gemini API wrapper
- `storage/`: CSV logging and session persistence
- `dashboard/`: analytics and chart rendering
- `utils/`: configuration, constants, helpers
- `data/`: raw and processed datasets, logs

## Setup
1. Create a virtual environment: `python -m venv .venv`
2. Activate it: `\.venv\Scripts\Activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Add your Gemini API key in `.env`

> If your current `transformers` installation does not include TensorFlow-backed model support, the app will still start and fall back to BiLSTM-only inference for the BERT analysis path.

## Next steps
- Implement training scripts in `training/`
- Build the prediction pipeline in `pipeline/`
- Create the Streamlit frontend in `app.py`
