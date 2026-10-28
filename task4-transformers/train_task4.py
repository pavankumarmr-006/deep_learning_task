
import os
import time
import numpy as np
import pandas as pd
import torch

from datasets import load_dataset
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    pipeline,
)

MODEL_NAME = "distilbert/distilbert-base-uncased"
SAVE_DIR = "saved_distilbert"
os.makedirs(SAVE_DIR, exist_ok=True)

# 1. Load a labelled sentiment dataset
print("\nLoading Rotten Tomatoes dataset...")
dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes")

# Small subset to make local training more manageable
train_data = dataset["train"].shuffle(seed=42).select(range(600))
test_data = dataset["test"].shuffle(seed=42).select(range(200))

train_texts = train_data["text"]
train_labels = np.array(train_data["label"])
test_texts = test_data["text"]
test_labels = np.array(test_data["label"])

print("Training examples:", len(train_texts))
print("Testing examples:", len(test_texts))

# 2. Tokenizer demonstration for two sentences
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

sentences = [
    "This movie was absolutely wonderful.",
    "The story was boring and disappointing.",
]

print("\nTOKENIZER OUTPUT")
for sentence in sentences:
    encoded = tokenizer(sentence)
    print("\nSentence:", sentence)
    print("Tokens:", tokenizer.convert_ids_to_tokens(encoded["input_ids"]))
    print("Input IDs:", encoded["input_ids"])
    print("Attention mask:", encoded["attention_mask"])

# 3. Run sentiment-analysis and NER pipelines
print("\nSENTIMENT-ANALYSIS PIPELINE")
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
)
for sentence in [
    "I really enjoyed this film.",
    "The movie was terrible.",
    "The acting was good, but the plot was weak.",
]:
    print(sentence, "->", sentiment_pipe(sentence))

print("\nNER PIPELINE")
ner_pipe = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple",
)
for sentence in [
    "Sundar Pichai works at Google.",
    "Barack Obama visited London.",
    "Microsoft was founded in the United States.",
]:
    print(sentence, "->", ner_pipe(sentence))

# 4. Train TF-IDF + Logistic Regression baseline
print("\nTraining TF-IDF baseline...")
tfidf_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b\w+\b",
        max_features=20000,
    )),
    ("classifier", LogisticRegression(max_iter=1000)),
])

start = time.perf_counter()
tfidf_model.fit(train_texts, train_labels)
tfidf_train_time = time.perf_counter() - start

start = time.perf_counter()
tfidf_predictions = tfidf_model.predict(test_texts)
tfidf_inference_ms = (
    (time.perf_counter() - start) / len(test_texts) * 1000
)

tfidf_accuracy = accuracy_score(test_labels, tfidf_predictions)
tfidf_f1 = f1_score(test_labels, tfidf_predictions, average="macro")

print("\nTF-IDF accuracy:", round(tfidf_accuracy, 4))
print("TF-IDF macro F1:", round(tfidf_f1, 4))
print("TF-IDF confusion matrix:")
print(confusion_matrix(test_labels, tfidf_predictions))
print("\nTF-IDF classification report:")
print(classification_report(test_labels, tfidf_predictions))

# 5. Prepare data for DistilBERT
print("\nTokenizing data for DistilBERT...")

def tokenize_batch(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=256,
    )

tokenized = dataset.map(tokenize_batch, batched=True)
tokenized["train"] = tokenized["train"].shuffle(seed=42).select(range(600))
tokenized["test"] = tokenized["test"].shuffle(seed=42).select(range(200))

tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"],
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
    }

training_args = TrainingArguments(
    output_dir="training_output",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=25,
    report_to="none",
    seed=42,
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

print("\nFine-tuning DistilBERT...")
start = time.perf_counter()
trainer.train()
bert_train_time = time.perf_counter() - start

# 6. Evaluate fine-tuned model
print("\nEvaluating DistilBERT...")
start = time.perf_counter()
evaluation = trainer.evaluate()
bert_eval_time = time.perf_counter() - start

pred_output = trainer.predict(tokenized["test"])
bert_predictions = np.argmax(pred_output.predictions, axis=-1)

bert_accuracy = accuracy_score(test_labels, bert_predictions)
bert_f1 = f1_score(test_labels, bert_predictions, average="macro")

print("\nDistilBERT accuracy:", round(bert_accuracy, 4))
print("DistilBERT macro F1:", round(bert_f1, 4))
print("DistilBERT confusion matrix:")
print(confusion_matrix(test_labels, bert_predictions))
print("\nDistilBERT classification report:")
print(classification_report(test_labels, bert_predictions))

# 7. Measure inference latency on the test set
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

inference_texts = test_texts[:50]
encoded_batch = tokenizer(
    inference_texts,
    padding=True,
    truncation=True,
    max_length=256,
    return_tensors="pt",
)
encoded_batch = {k: v.to(device) for k, v in encoded_batch.items()}

if device.type == "cuda":
    torch.cuda.synchronize()

start = time.perf_counter()
with torch.no_grad():
    model(**encoded_batch)

if device.type == "cuda":
    torch.cuda.synchronize()

bert_inference_ms = (
    (time.perf_counter() - start) / len(inference_texts) * 1000
)

# 8. Save the fine-tuned model and tokenizer
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print("\nSaved model and tokenizer to:", SAVE_DIR)

# 9. Create comparison table
comparison = pd.DataFrame([
    {
        "Model": "TF-IDF + Logistic Regression",
        "Accuracy": tfidf_accuracy,
        "Macro F1": tfidf_f1,
        "Training time (seconds)": tfidf_train_time,
        "Inference time (ms/text)": tfidf_inference_ms,
    },
    {
        "Model": "Fine-tuned DistilBERT",
        "Accuracy": bert_accuracy,
        "Macro F1": bert_f1,
        "Training time (seconds)": bert_train_time,
        "Inference time (ms/text)": bert_inference_ms,
    },
])

comparison.to_csv("task4_comparison.csv", index=False)

print("\nFINAL COMPARISON TABLE")
print(comparison.round(4).to_string(index=False))
print("\nSaved comparison to task4_comparison.csv")
