import streamlit as st
import pdfplumber
import docx2txt
import os
import tempfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Screener", layout="wide")

st.title("🧠 AI Resume Screener")
st.markdown("Upload multiple resumes and a job description to rank candidates based on relevance.")

# Upload resumes
st.subheader("📄 Upload multiple resumes")
uploaded_resumes = st.file_uploader("Upload Resumes", type=['pdf', 'docx', 'txt'], accept_multiple_files=True)

# Upload/paste job description
st.subheader("📝 Job Description")
job_description = st.text_area("Paste the job description here:")

# Function to extract text from PDF
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        return "\n".join([page.extract_text() or "" for page in pdf.pages])

# Function to extract text from DOCX
def extract_text_from_docx(file):
    temp_path = os.path.join(tempfile.gettempdir(), file.name)
    with open(temp_path, 'wb') as f:
        f.write(file.read())
    return docx2txt.process(temp_path)

# Function to extract text based on file type
def extract_text(file):
    if file.type == "application/pdf":
        return extract_text_from_pdf(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return extract_text_from_docx(file)
    elif file.type == "text/plain":
        return str(file.read(), "utf-8")
    else:
        return ""

# When user clicks Match Resumes
if st.button("🏆 Rank Resumes"):
    if not uploaded_resumes or not job_description.strip():
        st.error("Please upload at least one resume and enter a job description.")
    else:
        resume_texts = []
        resume_names = []

        for file in uploaded_resumes:
            text = extract_text(file)
            resume_texts.append(text)
            resume_names.append(file.name)

        # Show extracted preview
        st.subheader("🧾 Extracted Resume Content (Preview)")
        for name, text in zip(resume_names, resume_texts):
            st.markdown(f"**{name}**")
            st.text(text[:500] + ("..." if len(text) > 500 else ""))

        # Combine job description with resumes for vectorization
        all_docs = [job_description] + resume_texts
        vectorizer = TfidfVectorizer(stop_words='english')
        vectors = vectorizer.fit_transform(all_docs)

        # Cosine similarity between JD and each resume
        scores = cosine_similarity(vectors[0:1], vectors[1:])[0]

        # Sort resumes based on score
        scored_resumes = sorted(zip(resume_names, scores), key=lambda x: x[1], reverse=True)

        st.subheader("🏆 Ranked Resumes:")
        for name, score in scored_resumes:
            st.write(f"📄 {name} — Match Score: {score*100:.2f}%")










