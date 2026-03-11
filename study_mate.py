import streamlit as st
import io
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
import warnings
import pypdf
warnings.filterwarnings("ignore")

st.set_page_config(page_title="StudyMate AI", page_icon="🎓", layout="wide")
st.title("🎓 StudyMate AI")
st.markdown("### Your Personalized AI Study Companion")

# -----------------------------
# GROQ API KEY
# -----------------------------
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    st.error("⚠️ Service unavailable. Please contact the administrator.")
    st.stop()

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

if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("📄 Upload your study PDF", type="pdf")

if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# PROCESS PDF (in-memory, no disk writing)
# -----------------------------
if uploaded_file:
    file_key = uploaded_file.name

    if "vectorstore" not in st.session_state or st.session_state.get("file_key") != file_key:
        with st.spinner("🔄 Processing your document..."):

            # Read PDF directly from memory
            pdf_bytes = uploaded_file.read()
            pdf_reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))

            # Extract text into Document objects
            pages = []
            for i, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    pages.append(Document(
                        page_content=text,
                        metadata={"page": i + 1, "source": uploaded_file.name}
                    ))

            if not pages:
                st.error("❌ Could not extract text from this PDF. It may be scanned or image-based.")
                st.stop()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
            docs = text_splitter.split_documents(pages)

            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(docs, embeddings)

            st.session_state.vectorstore = vectorstore
            st.session_state.file_key = file_key
            st.session_state.messages = []

        st.success("✅ Document ready! Ask your first question below.")

    # -----------------------------
    # DISPLAY CHAT HISTORY
    # -----------------------------
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # -----------------------------
    # CHAT INPUT
    # -----------------------------
    user_query = st.chat_input("❓ Ask a question about your document...")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Build conversation history for context
        conversation_history = ""
        if len(st.session_state.messages) > 1:
            for msg in st.session_state.messages[:-1]:
                role = "Student" if msg["role"] == "user" else "Assistant"
                conversation_history += f"{role}: {msg['content']}\n"

        template = f"""
        {PERSONAS[mode]}
        Use ONLY the context below to answer the question.
        If the answer is not in the context, say "The answer is not found in the document."

        Context:
        {{context}}

        Previous conversation:
        {conversation_history}

        Current Question:
        {{question}}

        Answer:
        """

        QA_PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
        llm = ChatGroq(model=model_choice, temperature=temperature, api_key=groq_api_key)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})

        qa_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | QA_PROMPT
            | llm
            | StrOutputParser()
        )

        with st.chat_message("assistant"):
            with st.spinner("🧠 Thinking..."):
                response = qa_chain.invoke(user_query)
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.info("📄 Please upload a PDF to get started.")