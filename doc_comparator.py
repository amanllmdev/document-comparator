import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from dotenv import load_dotenv
import PyPDF2
import os

load_dotenv()

# Initialize Gemini Flash
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

st.set_page_config(page_title="📄 Document Comparator", layout="centered")
st.title("📄 Document Comparator using Gemini Flash")

def read_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        pdf = PyPDF2.PdfReader(uploaded_file)
        return "\n".join(page.extract_text() for page in pdf.pages)
    else:
        return uploaded_file.read().decode("utf-8")

file1 = st.file_uploader("Upload Document 1", type=["pdf", "txt"])
file2 = st.file_uploader("Upload Document 2", type=["pdf", "txt"])

if file1 and file2:
    text1 = read_file(file1)
    text2 = read_file(file2)

    if st.button("🔍 Compare Documents"):
        with st.spinner("Analyzing with Gemini Flash..."):
            prompt = f"""
            Compare the following two documents and return:
            - A summary of each
            - Key differences in tone, facts, and focus
            - Any inconsistencies or mismatched data

            Document 1:
            {text1}

            Document 2:
            {text2}
            """

            result = llm.invoke(prompt)
            st.success("✅ Comparison Complete!")

            st.subheader("🧾 Result")
            st.text_area("Gemini Comparison Output", result, height=400)

st.markdown("---")
st.caption("Powered by Gemini 1.5 Flash + Streamlit 🚀")
