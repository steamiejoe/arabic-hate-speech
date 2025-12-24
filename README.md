# Yaqeen-AI: Arabic Hate Speech Detection with Multi-Task Learning

## Overview
This project presents **Yaqeen-AI**, a robust system for detecting and classifying Arabic hate speech. It addresses the scarcity of fine-grained Arabic datasets by harmonizing six major corpora (OSACT, MLMA, SoHateful, ArMI, Brothers, Egyptian-5-Way).

## Key Features
- **Data Harmonization:** Unified ~40k samples from diverse sources into a standard schema.
- **Advanced Modeling:** Compares a **Multi-Task Learning (MTL)** MARBERTv2 architecture against a **Cascade** approach and classic ML baselines.
- **Granular Classification:** Detects specific hate types:
  - **GH:** Gender-Based Hate
  - **OH:** Origin/Race-Based Hate
  - **RH:** Religious Hate
- **Deployment:** Includes an interactive Streamlit dashboard for real-time inference.

## Project Structure
- `notebooks/`: Step-by-step processing, training, and evaluation.
- `src/`: Core preprocessing and model definitions.
- `app/`: Streamlit web application.

## Installation
```bash
pip install -r requirements.txt
streamlit run app/app.py