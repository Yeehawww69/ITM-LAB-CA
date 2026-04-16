import torch
from datasets import load_dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
# DEVICE SETUP (IMPORTANT for VS Code)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# LOAD DATASET
dataset = load_dataset("imdb")

# Reduce size for faster training
dataset["train"] = dataset["train"].shuffle(seed=42).select(range(10000))
dataset["test"] = dataset["test"].shuffle(seed=42).select(range(5000))

# TOKENIZER
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

dataset = dataset.map(tokenize, batched=True)

dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# MODEL
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

model.to(device)

# METRICS
def compute_metrics(pred):
    logits, labels = pred
    preds = torch.argmax(torch.tensor(logits), axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary'
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

# TRAINING ARGUMENTS
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # UPDATED for transformers v5+
    save_strategy="epoch",
    logging_dir="./logs",
    learning_rate=2e-5,
    per_device_train_batch_size=8,   # smaller for local machine
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    report_to="none"
)

# TRAINER
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    compute_metrics=compute_metrics
)

# TRAIN
trainer.train()
torch.save(model.state_dict(), "distilbert_sentiment.pth")
print("Model saved as distilbert_sentiment.pth")
# EVALUATE
results = trainer.evaluate()
print("\nFinal Results:", results)

# PREDICTION FUNCTION
def predict(text):
    model.eval()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    return "Positive 😊" if predicted_class == 1 else "Negative 😞"

# TEST
print(predict("This movie was absolutely amazing, I loved it!"))
print(predict("Worst movie ever. Total waste of time."))