import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="StudyMate AI", page_icon="🎓", layout="wide")
st.title("🎓 StudyMate AI")
st.markdown("### Your Personalized AI Study Companion")

# -----------------------------
# GROQ API KEY
# -----------------------------
# Tries Streamlit secrets first (cloud), falls back to sidebar input (local)
groq_api_key = None

try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

if not groq_api_key:
    groq_api_key = st.sidebar.text_input("🔑 Enter Groq API Key", type="password")
    if not groq_api_key:
        st.sidebar.warning("API key required. Get one free at https://console.groq.com")

# -----------------------------
# SYSTEM PERSONAS
# -----------------------------
PERSONAS = {
    "Summarize": "Summarize the following context into concise bullet points.",
    "Professor": "Explain the context like a friendly college professor using analogies and simple examples.",
    "Study Guide": "Extract key definitions and important concepts to create a structured study guide.",
    "Socratic Tutor": "Do NOT give the final answer. Guide the student using hints and thoughtful questions."
}

# -----------------------------
# SIDEBAR SETTINGS
# -----------------------------
st.sidebar.header("⚙️ Settings")
mode = st.sidebar.radio("Choose Study Mode:", list(PERSONAS.keys()))
temperature = st.sidebar.slider("Creativity Level", 0.0, 1.0, 0.3)
model_choice = st.sidebar.selectbox("Choose Model:", [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
    "deepseek-r1-distill-llama-70b"
])

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("📄 Upload your study PDF", type="pdf")
user_query = st.text_input("❓ What do you want to learn?")

if not groq_api_key:
    st.info("👈 Please enter your Groq API key in the sidebar to get started.")
    st.stop()

if uploaded_file and user_query:
    file_key = uploaded_file.name

    if "vectorstore" not in st.session_state or st.session_state.get("file_key") != file_key:
        with st.spinner("🔄 Processing your document..."):
            # Save temp file
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getvalue())

            # Load PDF
            loader = PyPDFLoader("temp.pdf")
            pages = loader.load()

            # Split text
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
            docs = text_splitter.split_documents(pages)

            # Embeddings (free, runs locally, no API needed)
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

            # Vector store
            vectorstore = FAISS.from_documents(docs, embeddings)

            st.session_state.vectorstore = vectorstore
            st.session_state.file_key = file_key
            os.remove("temp.pdf")
    else:
        vectorstore = st.session_state.vectorstore

    # -----------------------------
    # PROMPT TEMPLATE
    # -----------------------------
    template = f"""
    {PERSONAS[mode]}
    Use ONLY the context below to answer the question.
    If the answer is not in the context, say "The answer is not found in the document."

    Context:
    {{context}}

    Question:
    {{question}}

    Answer:
    """

    QA_PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

    # -----------------------------
    # GROQ LLM
    # -----------------------------
    llm = ChatGroq(
        model=model_choice,
        temperature=temperature,
        api_key=groq_api_key
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | QA_PROMPT
        | llm
        | StrOutputParser()
    )

    with st.spinner("🧠 Generating response..."):
        response = qa_chain.invoke(user_query)

    st.success("✅ Done!")
    st.markdown("## 📚 StudyMate Response")
    st.write(response)

elif uploaded_file and not user_query:
    st.info("💬 PDF uploaded! Now type your question above.")
elif not uploaded_file:
    st.info("📄 Please upload a PDF to get started.")