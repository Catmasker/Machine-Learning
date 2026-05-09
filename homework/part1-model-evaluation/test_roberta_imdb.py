"""
Run textattack/roberta-base-imdb on IMDB top 500 reviews.
This model achieves 100% on first 50 reviews and 98.4% on all 500.
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import pandas as pd
import time
from transformers import pipeline

df = pd.read_csv("imdb_top_500.csv")
print(f"Dataset size: {len(df)}")

MODEL = "textattack/roberta-base-imdb"

print(f"\nLoading model: {MODEL} ...")
start = time.time()

classifier = pipeline(
    task="sentiment-analysis",
    model=MODEL,
    tokenizer=MODEL,
    framework="pt"
)

print(f"Model loaded in {time.time() - start:.1f}s")
print(f"Model labels: {classifier.model.config.id2label}")

# Run on all 500 reviews
print(f"\nRunning inference on all 500 reviews...")
start = time.time()
reviews = df["text"].tolist()
true_labels = df["label"].tolist()
preds = classifier(reviews, truncation=True, batch_size=16)
print(f"Inference done in {time.time() - start:.1f}s\n")

# Convert labels
def to_label(pred):
    return 1 if pred["label"] == "LABEL_1" else 0

pred_labels = [to_label(p) for p in preds]

# Overall accuracy
correct_all = sum(p == t for p, t in zip(pred_labels, true_labels))
acc_all = correct_all / len(true_labels)

# First 50 accuracy
correct_50 = sum(p == t for p, t in zip(pred_labels[:50], true_labels[:50]))
acc_50 = correct_50 / 50

print("=" * 60)
print(f"Model: {MODEL}")
print("=" * 60)
print(f"Accuracy on first 50 reviews: {acc_50:.2%} ({correct_50}/50)")
print(f"Accuracy on ALL 500 reviews: {acc_all:.2%} ({correct_all}/{len(true_labels)})")
print("=" * 60)

# List all wrong predictions
wrong_indices = [i for i in range(len(true_labels)) if pred_labels[i] != true_labels[i]]
print(f"\nTotal wrong predictions: {len(wrong_indices)} out of {len(true_labels)}")
print(f"\nAll wrong predictions:")
for i, idx in enumerate(wrong_indices):
    print(f"\n  [{i+1}] Review #{idx+1}")
    print(f"      True: {true_labels[idx]}, Predicted: {pred_labels[idx]}, Score: {preds[idx]['score']:.4f}")
    print(f"      Text: {reviews[idx][:200]}...")

# Save results to file
with open("roberta_imdb_results.txt", "w") as f:
    f.write(f"Model: {MODEL}\n")
    f.write(f"Accuracy on first 50: {acc_50:.2%} ({correct_50}/50)\n")
    f.write(f"Accuracy on all 500: {acc_all:.2%} ({correct_all}/{len(true_labels)})\n")
    f.write(f"Wrong predictions: {len(wrong_indices)}/500\n\n")
    for idx in wrong_indices:
        f.write(f"Review #{idx+1}: True={true_labels[idx]}, Pred={pred_labels[idx]}, Score={preds[idx]['score']:.4f}\n")

print(f"\n\nResults saved to roberta_imdb_results.txt")
