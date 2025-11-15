# arabic-hate-speech
Arabic hate speech # Arabic Hate Speech Detection (MARBERT Fine-Tuning)

This repository contains a full, research-grade pipeline for Arabic hate speech detection using the MARBERT transformer model. The project includes:

- Multi-source dataset integration (OSACT, MLMA, SoHateful, and hydrated Twitter datasets)
- Text preprocessing & normalization for Modern Standard Arabic and dialects
- Robust training using HuggingFace Transformers
- Weighted loss / threshold tuning to optimize recall
- Error analysis, interpretability, and fairness checks
- (Optional) Deployment as a browser extension for real-time hate-speech detection

---

## 📂 Repository Structure

```
arabic-hate-speech/
│
├── data/
│   ├── raw/                # Unmodified datasets (input only)
│   ├── interim/            # After hydration / merging
│   └── processed/          # Cleaned & preprocessed datasets
│
├── notebooks/              # Jupyter notebooks (EDA, training, evaluation)
│
├── src/                    # Python modules (preprocessing, training, eval)
│
├── reports/                # Visualization, analysis, model card
│   └── figures/            # Saved plots
│
├── models/                 # MARBERT checkpoints & exported models
│
├── requirements.txt        # Python dependencies
├── environment.yml         # Conda environment
├── DATASET_CARD.md         # Documentation of dataset sources & licenses
└── README.md               # This file
```

---

## 🚀 Getting Started

### Install dependencies

```bash
pip install -r requirements.txt
```

### Dataset

Place your merged dataset into:

```
data/raw/unified_hate_raw.csv
```

Run:

```
notebooks/0_data_snapshot.ipynb
```

to verify dataset integrity and see dataset-level summary stats.

---

## 📌 Model

The project fine-tunes:

- **MARBERT** (UBC-NLP/MARBERT)

Binary classification:
- 0 → Non-hate  
- 1 → Hate speech / offensive language

---

## 📄 License

This project is for research and academic usage only. Please respect the original dataset licenses and Twitter terms of service for hydrated tweets.

---

## ✨ Author

Mohammad Ali  
Undergraduate Project — Arabic NLP  
detection using MARBERT (NLP research project)
