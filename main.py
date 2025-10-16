from transformers import BartForConditionalGeneration, BartTokenizer

# Load lại mô hình và tokenizer đã fine-tune
model_name_or_path = "./bart_finetuned"
tokenizer = BartTokenizer.from_pretrained(model_name_or_path)
model = BartForConditionalGeneration.from_pretrained(model_name_or_path)


def summarize(text, max_length=128, min_length=30):
    inputs = tokenizer(
        [text], max_length=512, truncation=True, return_tensors="pt"
    )
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        num_beams=4,
        length_penalty=2.0,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    text = "Nhập văn bản cần tóm tắt ở đây."
    print("Tóm tắt:", summarize(text))
