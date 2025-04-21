from datasets import load_dataset # step 1
from transformers import AutoTokenizer # step 2
import torch # step 3
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoModelForSequenceClassification # step 4
from torch.optim import AdamW # step 5
from sklearn.metrics import accuracy_score # step 6
import matplotlib.pyplot as plt # step 7
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Step 1 - load the ag_news dataset from huggingface
dataset = load_dataset("ag_news")
train_texts = dataset['train']['text']
train_labels = dataset['train']['label']

# Step 2 - load the tokenizer for the distilbert-base-uncased model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer(train_texts, padding=True, truncation=True, return_tensors="pt")

# Step 3 - convert the tokens to a tensor dataset
labels = torch.tensor(train_labels)
dataset = TensorDataset(tokens['input_ids'], tokens['attention_mask'], labels)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

# Step 4 - Define a model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)

# Step 5 - Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

scaler = GradScaler()

model.train()
for epoch in range(5):
  epoch_loss = 0
  print(f"\n🚀 Epoch {epoch+1}")
  for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
    input_ids, attention_mask, labels = [x.to(device) for x in batch]

    optimizer.zero_grad()

    with autocast(device_type=device.type):
      outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
      loss = outputs.loss

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    epoch_loss += loss.item()
  avg_loss = epoch_loss / len(train_loader)
  print(f"✅ Epoch {epoch+1} complete — Avg Loss: {avg_loss:.4f}")

# Step 6 - Evaluate the model
# Load test data
test_texts = dataset = load_dataset("ag_news")['test']['text']
test_labels = dataset = load_dataset("ag_news")['test']['label']

# Tokenise test data
test_tokens = tokenizer(test_texts, padding=True, truncation=True, return_tensors="pt")
test_input_ids = test_tokens['input_ids']
test_attention_mask = test_tokens['attention_mask']
test_labels = torch.tensor(test_labels)

# Create DataLoader for test data
test_dataset = TensorDataset(test_input_ids, test_attention_mask, test_labels)
test_loader = DataLoader(test_dataset, batch_size=64)

# Evaluate
model.eval()
all_preds, all_true = [], []
with torch.no_grad():
  for batch in test_loader:
    input_ids, attention_mask, labels = [x.to(device) for x in batch]
    outputs = model(input_ids, attention_mask=attention_mask)
    preds = torch.argmax(outputs.logits, dim=1)
    all_preds.extend(preds.cpu().numpy())
    all_true.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_true, all_preds)
print(f"\n✅ Evaluation Accuracy: {accuracy:.4f}")

# Step 7 - Confusion Matrix
cm = confusion_matrix(all_true, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["World", "Sports", "Business", "Sci/Tech"])

plt.figure(figsize=(8, 6))
disp.plot(cmap='Blues', xticks_rotation=45, values_format='d')
plt.title("Confusion Matrix for AG News Classification")
plt.savefig('confusion-matrix.png')
plt.close()

# Step 8 - Actually save the model, tokenizer and label mappings
output_dir = "lab03-model"

label2id = {
    "World": 0,
    "Sports": 1,
    "Business": 2,
    "Sci/Tech": 3
}
id2label = {v: k for k, v in label2id.items()}

model.config.label2id = label2id
model.config.id2label = id2label

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"\n✅ Model and tokenizer saved to '{output_dir}/'")
