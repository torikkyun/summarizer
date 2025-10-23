from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
import json

# Load dữ liệu
with open("./vietnamese_news_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Chuyển dữ liệu sang dạng HuggingFace Dataset
train_data = [
    {"text": item["text"], "summary": item["summary"]} for item in data
]
dataset = Dataset.from_list(train_data)

# Load tokenizer và model
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)


# Tiền xử lý dữ liệu
def preprocess(example):
    inputs = tokenizer(
        example["text"], max_length=512, truncation=True, padding="max_length"
    )
    targets = tokenizer(
        example["summary"],
        max_length=128,
        truncation=True,
        padding="max_length",
    )
    inputs["labels"] = targets["input_ids"]
    return inputs


tokenized_dataset = dataset.map(preprocess, remove_columns=["text", "summary"])

# Thiết lập tham số huấn luyện
training_args = TrainingArguments(
    output_dir="./bart_finetuned",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    fp16=True,
    report_to="none",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Huấn luyện
trainer.train()

tokenizer.save_pretrained("./bart_finetuned")
model.save_pretrained("./bart_finetuned")
