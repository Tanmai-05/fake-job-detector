import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

# 1. Load and clean dataset
df = pd.read_csv("data/fake_job_postings.csv")
df = df[['title', 'description', 'fraudulent']].dropna()
df["text"] = df["title"] + " " + df["description"]
df = df[["text", "fraudulent"]]

# 2. Split data
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"].tolist(), df["fraudulent"].tolist(), test_size=0.2, random_state=42
)

# 3. Tokenization
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

# 4. Create Hugging Face datasets
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': train_labels
})

test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': test_labels
})

# 5. Load DistilBERT classification model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# 6. Define training arguments (compatible version)
training_args = TrainingArguments(
    output_dir="./models/bert_model",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=500,
    logging_dir="./logs"
)

# 7. Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# 8. Train model
print("🔁 Training BERT model...")
trainer.train()

# 9. Save model and tokenizer
model.save_pretrained("models/bert_model")
tokenizer.save_pretrained("models/bert_model")

print("✅ BERT model training complete and saved to models/bert_model")
