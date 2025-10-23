import json

from dotenv import load_dotenv
from src.abstractive import AbstractiveSummarizer
from src.text_rank import TextRankSummarizer
from google.generativeai.client import configure
from google.generativeai.generative_models import GenerativeModel
from rouge_score import rouge_scorer
import os

load_dotenv()

# Load dataset
with open("vietnamese_news_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Khởi tạo các mô hình
abstractive = AbstractiveSummarizer(model_name="facebook/bart-large-cnn")
textrank = TextRankSummarizer()
configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini = GenerativeModel("gemini-2.0-flash")

scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)


def gemini_summarize(text):
    prompt = f"Tóm tắt văn bản sau bằng tiếng Việt:\n{text}"
    response = gemini.generate_content(prompt)
    return response.text.strip()


results = []
for item in data[:10]:
    text = item.get("text")
    reference = item.get("summary")

    if text:
        bart_input = text[:1000]
        abs_sum = abstractive.summarize(bart_input)
    else:
        abs_sum = ""
    textrank_sum = textrank.summarize(text)
    gemini_sum = gemini_summarize(text)

    abs_score = scorer.score(reference, abs_sum)
    textrank_score = scorer.score(reference, textrank_sum)
    gemini_score = scorer.score(reference, gemini_sum)

    results.append(
        {
            "title": item.get("title"),
            "reference": reference,
            "abstractive": abs_sum,
            "textrank": textrank_sum,
            "gemini": gemini_sum,
            "rouge1": {
                "abstractive": abs_score["rouge1"].fmeasure,
                "textrank": textrank_score["rouge1"].fmeasure,
                "gemini": gemini_score["rouge1"].fmeasure,
            },
            "rougeL": {
                "abstractive": abs_score["rougeL"].fmeasure,
                "textrank": textrank_score["rougeL"].fmeasure,
                "gemini": gemini_score["rougeL"].fmeasure,
            },
        }
    )

# In kết quả
for r in results:
    print(f"Title: {r['title']}")
    print(f"Reference: {r['reference']}")
    print(f"Abstractive: {r['abstractive']}")
    print(f"TextRank: {r['textrank']}")
    print(f"Gemini: {r['gemini']}")
    print(f"ROUGE-1: {r['rouge1']}")
    print(f"ROUGE-L: {r['rougeL']}")
    print("-" * 50)
