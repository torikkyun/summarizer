from src.abstractive import AbstractiveSummarizer


def main():
    text = (
        "Hôm nay, thời tiết tại Hà Nội rất đẹp với nắng vàng rực rỡ và nhiệt độ dễ chịu. "
        "Nhiều người dân đã tận dụng dịp cuối tuần để đi dạo quanh các công viên và thưởng thức không khí trong lành. "
        "Các quán cà phê ngoài trời cũng trở nên đông đúc hơn bao giờ hết khi mọi người tìm đến để thư giãn và trò chuyện cùng bạn bè."
    )
    """
    facebook/bart-large-cnn
    google/pegasus-xsum
    google-t5/t5-base
    VietAI/vietbart-news
    VietAI/viett5-base
    """
    summarizer = AbstractiveSummarizer(model_name="facebook/bart-large-cnn")
    summary = summarizer.summarize(text)
    print("Summary: ")
    print(summary)


if __name__ == "__main__":
    main()
