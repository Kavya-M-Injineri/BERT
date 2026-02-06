# BERT Sentiment Insight: End-to-End IMDB Analysis

A premium, state-of-the-art sentiment analysis dashboard powered by a fine-tuned **BERT** model. This project combines deep learning with a modern, glassmorphic UI to provide innovative "explainable" insights into movie reviews.

![Project Preview](https://via.placeholder.com/800x400?text=BERT+Sentiment+Insight+Dashboard)

## 🌟 Features

- **Fine-Tuned BERT Engine**: Uses Hugging Face's `bert-base-uncased` fine-tuned on the IMDB dataset for high-accuracy sentiment classification.
- **Explainable AI (XAI)**: An innovative word-highlighting system that reveals exactly which words influenced the model's decision (Positive or Negative).
- **Premium Glassmorphism UI**: A stunning dashboard featuring backdrop blurs, vibrant gradients, and smooth animated elements.
- **Interactive Sentiment Gauge**: Real-time confidence visualization with animated gauge transitions.
- **Flask REST API**: A lightweight and robust backend handling model inference and explanation logic.

## 🛠️ Technology Stack

- **Machine Learning**: PyTorch, Transformers (Hugging Face), Datasets.
- **Backend**: Flask (Python).
- **Frontend**: HTML5, Vanilla CSS3 (Glassmorphism), Javascript (ES6).
- **Dataset**: IMDB Movie Reviews.

## 🚀 Getting Started

### 1. Installation

Ensure you have Python 3.8+ installed. Then, install the required dependencies:

```bash
pip install torch transformers datasets flask pandas evaluate
```

### 2. Prepare the Model

Run the training script to fine-tune BERT on a subset of the IMDB dataset. This will save the model to the `models/` directory.

```bash
python train_bert.py
```

*Note: Training on CPU may take some time. A GPU is recommended for full dataset fine-tuning.*

### 3. Launch the Application

Start the Flask server:

```bash
python app.py
```

Visit the dashboard in your browser at:
`http://127.0.0.1:5005`

## 📂 Project Structure

- `app.py`: Flask application with inference endpoints.
- `train_bert.py`: Training script for fine-tuning BERT.
- `templates/index.html`: Main dashboard UI.
- `static/`:
    - `style.css`: Premium glassmorphism styles.
    - `script.js`: Frontend logic and API connectivity.
- `models/`: Directory containing the fine-tuned model (GitIgnored).

## 🧪 Evaluation

The model is trained to exceed **85% accuracy** on the IMDB test set. Evaluation metrics and loss logs are generated during the training phase.

---
*Developed with focus on technical depth and visual excellence.*
