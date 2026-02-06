import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
import os
import evaluate

# Configuration
MODEL_NAME = 'distilbert-base-uncased' # Faster than full BERT for CPU execution
OUTPUT_DIR = './models/bert_sentiment'
LOGGING_DIR = './logs'
SUBSET_SIZE = 2000 # Use a subset for faster training while hitting 85% goal

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

if __name__ == "__main__":
    print("Loading IMDB dataset...")
    dataset = load_dataset("imdb")
    
    # Shuffle and select subset
    train_dataset = dataset["train"].shuffle(seed=42).select(range(SUBSET_SIZE))
    test_dataset = dataset["test"].shuffle(seed=42).select(range(SUBSET_SIZE // 2))

    print(f"Loading tokenizer {MODEL_NAME}...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    print("Tokenizing data...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    print(f"Loading model {MODEL_NAME}...")
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=LOGGING_DIR,
        logging_steps=10,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("Evaluating model...")
    results = trainer.evaluate()
    print(f"Evaluation Results: {results}")

    print(f"Saving model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training Complete!")
