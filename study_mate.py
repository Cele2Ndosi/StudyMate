import streamlit as st
import io
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
    "Summarize": "You summarize content into concise bullet points. Build on previous summaries if the student asks follow-up questions.",
    "Professor": "You are a friendly college professor who explains concepts using analogies and simple examples. Remember what you have already taught the student and build on it.",
    "Study Guide": "You extract key definitions and important concepts to create structured study guides. Remember previously covered topics and expand on them when asked.",
    "Socratic Tutor": "You are a Socratic tutor. Do NOT give final answers. Guide the student using hints and thoughtful questions. Remember the student's previous responses and build your next question on them."
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
    st.session_state.chat_history = []
    st.rerun()

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("📄 Upload your study PDF", type="pdf")

# -----------------------------
# INITIALIZE SESSION STATE
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []       # For displaying in UI
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # LangChain message objects for memory

# -----------------------------
# PROCESS PDF
# -----------------------------
if uploaded_file:
    file_key = uploaded_file.name

    if "vectorstore" not in st.session_state or st.session_state.get("file_key") != file_key:
        with st.spinner("🔄 Processing your document..."):
            pdf_bytes = uploaded_file.read()
            pdf_reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))

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
            st.session_state.chat_history = []

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
        # Show user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # -----------------------------
        # BUILD CHAIN WITH MEMORY
        # -----------------------------
        llm = ChatGroq(model=model_choice, temperature=temperature, api_key=groq_api_key)
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})

        # Prompt includes system persona, retrieved context, full chat history, and new question
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""{PERSONAS[mode]}

You have access to the following document context to answer questions.
Always use this context to ground your answers.
If the answer is not in the context, say "The answer is not found in the document."

Document Context:
{{context}}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Retrieve relevant docs
        retrieved_docs = retriever.invoke(user_query)
        context = format_docs(retrieved_docs)

        # Build the chain
        chain = prompt | llm | StrOutputParser()

        with st.chat_message("assistant"):
            with st.spinner("🧠 Thinking..."):
                response = chain.invoke({
                    "context": context,
                    "chat_history": st.session_state.chat_history,
                    "question": user_query
                })
            st.markdown(response)

        # Update both histories
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

else:
    st.info("📄 Please upload a PDF to get started.")