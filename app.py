import streamlit as st
import google.generativeai as genai
from underthesea import sent_tokenize, word_tokenize
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# Cấu hình trang
st.set_page_config(
    page_title="Hệ thống tóm tắt văn bản tiếng Việt",
    page_icon="📰",
    layout="wide"
)

# CSS tùy chỉnh
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .summary-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: rgba(240, 242, 246, 0.1);
        border: 1px solid rgba(250, 250, 250, 0.1);
        margin-top: 0.5rem;
    }
    .method-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-weight: bold;
        margin-bottom: 0;
    }
    .extractive {
        background-color: #90EE90;
        color: #006400;
    }
    .abstractive {
        background-color: #FFB6C1;
        color: #8B0000;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">📰 Hệ thống Tóm tắt Văn bản Tiếng Việt</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Cấu hình")
    
    # API Key
    api_key = st.text_input("Gemini API Key", type="password", help="Nhập API key của Google Gemini")
    
    # Chọn phương pháp
    st.subheader("Phương pháp tóm tắt")
    method = st.radio(
        "Chọn phương pháp:",
        ["Extractive (TextRank)", "Abstractive (Gemini AI)"],
        help="Extractive: Trích xuất câu quan trọng | Abstractive: Tạo tóm tắt mới"
    )
    
    # Tham số
    st.subheader("Tham số")
    if "Extractive" in method:
        num_sentences = st.slider("Số câu tóm tắt", 2, 10, 3)
    else:
        summary_length = st.select_slider(
            "Độ dài tóm tắt",
            options=["Ngắn", "Trung bình", "Dài"],
            value="Trung bình"
        )
    
    st.markdown("---")
    st.markdown("### 📖 Hướng dẫn")
    st.markdown("""
    1. Nhập API key Gemini
    2. Chọn phương pháp tóm tắt
    3. Dán văn bản cần tóm tắt
    4. Nhấn nút "Tóm tắt"
    """)

# Hàm tóm tắt Extractive với TextRank
def textrank_summarize(text, num_sentences=3):
    try:
        # Tách câu
        sentences = sent_tokenize(text)
        
        if len(sentences) <= num_sentences:
            return text
        
        # Tiền xử lý: word tokenize cho mỗi câu
        processed_sentences = []
        for sent in sentences:
            words = word_tokenize(sent, format="text")
            processed_sentences.append(words.lower())
        
        # Tính TF-IDF
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(processed_sentences)
        
        # Tính độ tương đồng cosine
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # Tạo đồ thị và tính PageRank
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)
        
        # Sắp xếp câu theo điểm số
        ranked_sentences = sorted(
            ((scores[i], i, s) for i, s in enumerate(sentences)),
            reverse=True
        )
        
        # Lấy top câu và sắp xếp lại theo thứ tự xuất hiện
        top_sentences = sorted(ranked_sentences[:num_sentences], key=lambda x: x[1])
        summary = ' '.join([s[2] for s in top_sentences])
        
        return summary
    except Exception as e:
        return f"Lỗi khi tóm tắt: {str(e)}"

# Hàm tóm tắt Abstractive với Gemini
def gemini_summarize(text, api_key, length="Trung bình"):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        length_prompts = {
            "Ngắn": "Tóm tắt văn bản sau thành 2-3 câu ngắn gọn nhất:",
            "Trung bình": "Tóm tắt văn bản sau thành một đoạn văn vừa phải (4-6 câu):",
            "Dài": "Tóm tắt chi tiết văn bản sau (7-10 câu):"
        }
        
        prompt = f"""{length_prompts[length]}

Văn bản: {text}

Yêu cầu:
- Tóm tắt bằng tiếng Việt
- Giữ nguyên ý chính
- Viết mạch lạc, dễ hiểu
- Không thêm thông tin ngoài văn bản gốc
"""
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Lỗi khi tóm tắt: {str(e)}"

# Layout chính
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Văn bản gốc")
    input_text = st.text_area(
        "Nhập văn bản cần tóm tắt:",
        height=400,
        placeholder="Dán văn bản báo chí, tin tức tiếng Việt vào đây...",
        help="Nhập văn bản tiếng Việt để tóm tắt"
    )
    
    # Hiển thị thống kê
    if input_text:
        word_count = len(input_text.split())
        sent_count = len(sent_tokenize(input_text))
        st.caption(f"📊 Thống kê: {word_count} từ, {sent_count} câu")

with col2:
    st.subheader("✨ Kết quả tóm tắt")
    
    # Nút tóm tắt
    if st.button("🚀 Tóm tắt văn bản", type="primary", use_container_width=True):
        if not input_text.strip():
            st.error("⚠️ Vui lòng nhập văn bản cần tóm tắt!")
        elif "Abstractive" in method and not api_key:
            st.error("⚠️ Vui lòng nhập Gemini API Key để sử dụng phương pháp Abstractive!")
        else:
            with st.spinner("⏳ Đang xử lý..."):
                if "Extractive" in method:
                    summary = textrank_summarize(input_text, num_sentences)
                    method_label = "EXTRACTIVE (TextRank)"
                    method_class = "extractive"
                else:
                    summary = gemini_summarize(input_text, api_key, summary_length)
                    method_label = "ABSTRACTIVE (Gemini AI)"
                    method_class = "abstractive"
                
                # Hiển thị kết quả
                st.markdown(f'<span class="method-badge {method_class}">{method_label}</span>', unsafe_allow_html=True)
                st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)
                
                # Thống kê tóm tắt
                summary_word_count = len(summary.split())
                summary_sent_count = len(sent_tokenize(summary))
                reduction = round((1 - summary_word_count / len(input_text.split())) * 100, 1)
                
                st.caption(f"📊 Tóm tắt: {summary_word_count} từ, {summary_sent_count} câu | Giảm {reduction}%")
                
                # Nút copy
                st.download_button(
                    label="📥 Tải xuống tóm tắt",
                    data=summary,
                    file_name="tom_tat.txt",
                    mime="text/plain"
                )

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🎓 Đồ án môn học AI - Hệ thống tóm tắt văn bản tiếng Việt</p>
        <p style='font-size: 0.9rem;'>Sử dụng TextRank và Google Gemini API</p>
    </div>
""", unsafe_allow_html=True)
