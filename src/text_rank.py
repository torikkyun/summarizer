class TextRankSummarizer:
    """
    Extractive Summarization sử dụng thuật toán TextRank
    Dựa trên PageRank để xếp hạng các câu quan trọng
    """

    def __init__(self, similarity_threshold=0.1):
        self.similarity_threshold = similarity_threshold

    def _build_similarity_matrix(self, sentences):
        """Xây dựng ma trận độ tương đồng giữa các câu"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            return similarity_matrix
        except (ValueError, AttributeError):
            n = len(sentences)
            return np.eye(n)

    def _create_graph(self, similarity_matrix):
        """Tạo đồ thị từ ma trận độ tương đồng"""
        import networkx as nx

        graph = nx.Graph()
        n = len(similarity_matrix)

        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i][j] > self.similarity_threshold:
                    graph.add_edge(i, j, weight=similarity_matrix[i][j])

        return graph

    def summarize(self, text, num_sentences=3, ratio=0.3):
        """
        Tóm tắt văn bản sử dụng TextRank

        Args:
            text: Văn bản cần tóm tắt
            num_sentences: Số câu trong bản tóm tắt
            ratio: Tỷ lệ câu giữ lại (nếu không chỉ định num_sentences)

        Returns:
            Bản tóm tắt
        """
        import re

        def sent_tokenize_simple(text):
            sentences = re.split(r"[.!?]+", text)
            return [s.strip() for s in sentences if s.strip()]

        sentences = sent_tokenize_simple(text)

        if len(sentences) <= num_sentences:
            return " ".join(sentences)

        similarity_matrix = self._build_similarity_matrix(sentences)
        graph = self._create_graph(similarity_matrix)

        import networkx as nx

        try:
            scores = nx.pagerank(graph, max_iter=100)
        except (nx.NetworkXError, nx.PowerIterationFailedConvergence):
            scores = {i: 1.0 / len(sentences) for i in range(len(sentences))}

        ranked_sentences = sorted(
            ((scores.get(i, 0), i, s) for i, s in enumerate(sentences)),
            reverse=True,
        )

        if num_sentences:
            top_sentences = ranked_sentences[:num_sentences]
        else:
            num_to_select = max(1, int(len(sentences) * ratio))
            top_sentences = ranked_sentences[:num_to_select]

        top_sentences = sorted(top_sentences, key=lambda x: x[1])
        summary = " ".join([s[2] for s in top_sentences])
        return summary
