import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from pathlib import Path

print("\n🚀 Starting LoRA fine-tuning on AG News + UK custom dataset (9 classes)...\n")

# Step 1 - Load and prepare the dataset
# Load AG News
ag_dataset = load_dataset("ag_news", split="train")
ag_df = pd.DataFrame({
  "text": ag_dataset["text"],
  "label": ag_dataset["label"]
})

# Map AG News labels to strings
ag_df["label"] = ag_df["label"].map({
  0: "World",
  1: "Sports",
  2: "Business",
  3: "Sci/Tech"
})

# Load custom dataset
uk_df = pd.read_csv("synthetic_news_uk.csv")
uk_df = uk_df.rename(columns={"label": "label", "text": "text"})

# Combine datasets
combined_df = pd.concat([ag_df, uk_df], ignore_index=True)
dataset = Dataset.from_pandas(combined_df)

# Encode labels
label_encoder = LabelEncoder()
label_encoder.fit(dataset["label"])
dataset = dataset.map(lambda x: {"label": x["label"]})
dataset = dataset.map(lambda x: {"labels": label_encoder.transform([x["label"]])[0]})

# Load tokenizer
lab03_model_path = str(Path("../../03-deep-learning/lab03-model").resolve())
tokenizer = AutoTokenizer.from_pretrained(lab03_model_path, local_files_only=True)

# Tokenise
def tokenize(data):
  return tokenizer(data["text"], padding="max_length", truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
dataset = dataset.train_test_split(test_size=0.2, seed=42, shuffle=True)
train_data = dataset["train"]
eval_data = dataset["test"]

# Load base model
model = AutoModelForSequenceClassification.from_pretrained(
  lab03_model_path,
  num_labels=len(label_encoder.classes_),
  ignore_mismatched_sizes=True,
  local_files_only=True,
  use_safetensors=True
)

# Configure LoRA
lora_config = LoraConfig(
  r=8,
  lora_alpha=32,
  target_modules=["query", "value"],
  lora_dropout=0.1,
  bias="none",
  task_type=TaskType.SEQ_CLS,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Train
training_args = TrainingArguments(
  output_dir="lora-news/",
  per_device_train_batch_size=16,
  per_device_eval_batch_size=16,
  num_train_epochs=5,
  learning_rate=2e-4,
  evaluation_strategy="epoch",
  save_strategy="epoch",
  logging_dir="./logs",
  logging_steps=10,
  report_to="none"
)

trainer = Trainer(
  model=model,
  args=training_args,
  train_dataset=train_data,
  eval_dataset=eval_data,
)

trainer.train()
print("✅ Training complete!")

# Evaluation
predictions = trainer.predict(eval_data)
preds = torch.argmax(torch.tensor(predictions.predictions), dim=1).numpy()
true = predictions.label_ids

pred_labels = label_encoder.inverse_transform(preds)
true_labels = label_encoder.inverse_transform(true)

print("🔍 Evaluation Results")
print(f"Accuracy: {accuracy_score(true_labels, pred_labels):.4f}")
print("\nClassification Report:")
print(classification_report(true_labels, pred_labels, target_names=label_encoder.classes_))

# Confusion matrix
cm = confusion_matrix(true_labels, pred_labels, labels=label_encoder.classes_)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix_9class.png")
plt.close()

# Save merged model
merged_model = model.merge_and_unload()
label2id = {label: i for i, label in enumerate(label_encoder.classes_)}
id2label = {i: label for label, i in label2id.items()}
merged_model.config.label2id = label2id
merged_model.config.id2label = id2label

merged_model.save_pretrained("lora-news/full-model")
tokenizer.save_pretrained("lora-news/full-model")
model.save_pretrained("lora-news/adapter")

print(f"\n✅ Model and tokenizer saved to 'lora-news/'")
