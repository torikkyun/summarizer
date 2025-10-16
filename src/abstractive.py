from transformers import pipeline


class AbstractiveSummarizer:
    """
    Tóm tắt văn bản dạng Abstractive sử dụng mô hình Transformer hiện đại (BART)
    """

    def __init__(self, model_name="facebook/bart-large-cnn"):
        self.summarizer = pipeline("summarization", model=model_name)

    def summarize(self, text, max_length=130, min_length=30):
        """
        Tóm tắt văn bản sử dụng mô hình Abstractive

        Args:
            text: Văn bản cần tóm tắt
            max_length: Độ dài tối đa của bản tóm tắt
            min_length: Độ dài tối thiểu của bản tóm tắt

        Returns:
            Bản tóm tắt
        """
        summary = self.summarizer(
            text, max_length=max_length, min_length=min_length, do_sample=False
        )
        return summary[0]["summary_text"]
