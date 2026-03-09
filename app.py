
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import nltk
import fitz
import re

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

st.set_page_config(
    page_title="Intelligent Document Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Intelligent Document Assistant")
st.markdown("Upload a PDF and ask questions about it!")
st.markdown("---")

def extract_text(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    doc.close()
    return text

def clean_text(text):
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^\w\s\.\,\!\?\-]", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()

def chunk_text(text, chunk_size=200, overlap=30):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks

def build_tfidf(chunks):
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    matrix = tfidf.fit_transform(chunks)
    return tfidf, matrix

def answer_question(question, chunks, tfidf, matrix, top_k=3):
    q_vec = tfidf.transform([question])
    scores = cosine_similarity(q_vec, matrix)[0]
    top_indices = scores.argsort()[::-1][:top_k]
    return [{"text": chunks[i], "similarity": float(scores[i])}
            for i in top_indices]

def summarize(text, tfidf, matrix, num_sentences=5):
    sentences = sent_tokenize(text)
    sentences = [s for s in sentences if len(s.split()) > 8]
    if not sentences:
        return "Could not generate summary."
    sent_matrix = tfidf.transform(sentences)
    doc_vec     = tfidf.transform([text[:2000]])
    scores      = cosine_similarity(doc_vec, sent_matrix)[0]
    top_idx     = sorted(scores.argsort()[::-1][:num_sentences])
    return " ".join([sentences[i] for i in top_idx])

# ── File Uploader on MAIN PAGE ──
st.subheader("Step 1: Upload Your PDF")
uploaded_file = st.file_uploader(
    "Choose a PDF file", type=["pdf"]
)

if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        raw_text      = extract_text(uploaded_file)
        cleaned       = clean_text(raw_text)
        chunks        = chunk_text(cleaned)
        tfidf, matrix = build_tfidf(chunks)

    st.success("PDF processed successfully!")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Document Info")
        st.info(f"Total characters: {len(raw_text):,}")
        st.info(f"Total chunks: {len(chunks)}")
        st.info(f"Total words: {len(cleaned.split()):,}")

    with col2:
        st.subheader("Document Summary")
        if st.button("Generate Summary"):
            with st.spinner("Summarizing..."):
                summary = summarize(cleaned, tfidf, matrix)
            st.success("Done!")
            st.write(summary)

    st.markdown("---")
    st.subheader("Ask a Question")
    question = st.text_input("Type your question:")

    if st.button("Get Answer"):
        if not question.strip():
            st.warning("Please type a question!")
        else:
            with st.spinner("Searching..."):
                results = answer_question(
                    question, chunks, tfidf, matrix
                )
            for i, r in enumerate(results):
                with st.expander(
                    f"Result {i+1} — Similarity: {r['similarity']:.1%}"
                ):
                    st.write(r["text"])

    if st.checkbox("Show raw text preview"):
        st.text_area("Raw Text", raw_text[:2000], height=200)

else:
    st.info("Please upload a PDF file above to get started!")
    st.markdown("### How to use:")
    st.markdown("1. Upload a PDF using the uploader above")
    st.markdown("2. Click Generate Summary")
    st.markdown("3. Type a question and click Get Answer")
