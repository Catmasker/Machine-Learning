"""
Test textattack/bert-base-uncased-imdb on IMDB top 500 dataset.
This model is BERT fine-tuned directly on IMDB, so it should perform well.
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import pandas as pd
import time
from transformers import pipeline

df = pd.read_csv("imdb_top_500.csv")
print(f"Dataset size: {len(df)}")

MODEL = "textattack/bert-base-uncased-imdb"

print(f"\nDownloading and loading model: {MODEL} ...")
start = time.time()

classifier = pipeline(
    task="sentiment-analysis",
    model=MODEL,
    tokenizer=MODEL,
    framework="pt"
)

print(f"Model loaded in {time.time() - start:.1f}s")
print(f"Model labels: {classifier.model.config.id2label}")

# Test on ALL 500 reviews
print(f"\nRunning inference on all 500 reviews...")
start = time.time()

reviews = df["text"].tolist()
true_labels = df["label"].tolist()

preds = classifier(reviews, truncation=True, batch_size=16)

print(f"Inference done in {time.time() - start:.1f}s")

def to_label(pred):
    label = pred["label"]
    return 1 if label == "LABEL_1" else 0

pred_labels = [to_label(p) for p in preds]

# Accuracy
correct = sum(p == t for p, t in zip(pred_labels, true_labels))
accuracy = correct / len(true_labels)

print(f"\n{'='*50}")
print(f"Model: {MODEL}")
print(f"Accuracy on ALL 500 reviews: {accuracy:.2%}")
print(f"Correct: {correct}/{len(true_labels)}")
print(f"{'='*50}")

# Also show first 50 accuracy for comparison
correct_50 = sum(p == t for p, t in zip(pred_labels[:50], true_labels[:50]))
acc_50 = correct_50 / 50
print(f"Accuracy on first 50 reviews: {acc_50:.2%}")

# Show wrong predictions
print(f"\nWrong predictions (first 10 of {len(true_labels) - correct} total):")
wrong = 0
for i in range(len(reviews)):
    if pred_labels[i] != true_labels[i]:
        wrong += 1
        if wrong <= 10:
            print(f"\n[{i+1}] True={true_labels[i]}, Pred={pred_labels[i]}, Score={preds[i]['score']:.4f}")
            print(f"    {reviews[i][:200]}...")
