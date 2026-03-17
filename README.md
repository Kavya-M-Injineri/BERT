# BERT Sentiment Insight 66666666666666666666666

> End-to-end IMDB sentiment analysis powered by a fine-tuned BERT model, with explainable AI and a glassmorphic dashboard.

---

## Overview

BERT Sentiment Insight is a full-stack NLP application that classifies movie reviews as **Positive** or **Negative** using a fine-tuned `bert-base-uncased` model trained on the IMDB dataset. Beyond raw predictions, it features a word-level **Explainable AI (XAI)** system that highlights the tokens most responsible for the model's decision — making the inference process transparent and interpretable.

The frontend is built with a premium glassmorphism aesthetic: backdrop blurs, gradient overlays, and smooth animated transitions, including a real-time confidence gauge.

---

## Features

- **Fine-Tuned BERT** — `bert-base-uncased` fine-tuned on IMDB for high-accuracy binary sentiment classification (target: >85% test accuracy).
- **Explainable AI** — Token-level attribution highlights which words pushed the model toward Positive or Negative sentiment.
- **Interactive Confidence Gauge** — Animated real-time visualization of prediction confidence.
- **Glassmorphism UI** — Backdrop blurs, vibrant gradients, and smooth transitions built in pure CSS3.
- **Flask REST API** — Lightweight Python backend serving model inference and XAI explanation endpoints.

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Framework | PyTorch, Hugging Face Transformers |
| Dataset | IMDB Movie Reviews (via Hugging Face Datasets) |
| Evaluation | Hugging Face Evaluate |
| Backend | Flask (Python) |
| Frontend | HTML5, CSS3, JavaScript (ES6) |

---

## Project Structure

```
bert-sentiment-insight/
├── app.py               # Flask app — inference & XAI endpoints
├── train_bert.py        # BERT fine-tuning script
├── templates/
│   └── index.html       # Dashboard UI
├── static/
│   ├── style.css        # Glassmorphism styles
│   └── script.js        # Frontend logic & API calls
└── models/              # Fine-tuned model weights (git-ignored)
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- GPU recommended for training (CPU works but is slower)

### 1. Install Dependencies

```bash
pip install torch transformers datasets flask pandas evaluate
```

### 2. Fine-Tune the Model

```bash
python train_bert.py
```

This trains `bert-base-uncased` on a subset of IMDB and saves the model to `models/`. Training logs and evaluation metrics are printed to stdout.

> **Note:** Full dataset fine-tuning benefits significantly from a GPU. On CPU, consider reducing the training subset size in `train_bert.py`.

### 3. Start the Server

```bash
python app.py
```

Open your browser at **http://127.0.0.1:5005**

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | `POST` | Returns sentiment label and confidence score |
| `/explain` | `POST` | Returns token-level attribution for XAI highlighting |

Both endpoints accept a JSON body: `{ "text": "Your movie review here." }`

---

## Model Performance

The model targets **≥ 85% accuracy** on the IMDB test set. Training metrics (loss, accuracy per epoch) are logged during the `train_bert.py` run.

---

## Notes

- The `models/` directory is git-ignored. Each collaborator must run `train_bert.py` locally or download a shared checkpoint.
- The XAI system uses gradient-based or attention-based attribution — see `app.py` for implementation details.

---

*Built with focus on technical depth and visual excellence.*
