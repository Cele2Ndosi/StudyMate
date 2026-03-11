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

st.set_page_config(page_title="StudyMate AI", page_icon="🎓", layout="wide")
st.title("🎓 StudyMate AI")
st.markdown("### Your Personalized AI Study Companion")

try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    st.error("⚠️ Service unavailable. Please contact the administrator.")
    st.stop()

PERSONAS = {
    "Summarize": "Summarize the following context into concise bullet points.",
    "Professor": "Explain the context like a friendly college professor using analogies and simple examples.",
    "Study Guide": "Extract key definitions and important concepts to create a structured study guide.",
    "Socratic Tutor": "Do NOT give the final answer. Guide the student using hints and thoughtful questions."
}

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

uploaded_file = st.file_uploader("📄 Upload your study PDF", type="pdf")

if "messages" not in st.session_state:
    st.session_state.messages = []

if uploaded_file:
    file_key = uploaded_file.name

    if "vectorstore" not in st.session_state or st.session_state.get("file_key") != file_key:
        with st.spinner("🔄 Processing your document..."):
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getvalue())

            loader = PyPDFLoader("temp.pdf")
            pages = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
            docs = text_splitter.split_documents(pages)

            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(docs, embeddings)

            st.session_state.vectorstore = vectorstore
            st.session_state.file_key = file_key
            st.session_state.messages = []
            os.remove("temp.pdf")
        st.success("✅ Document ready! Ask your first question below.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_query = st.chat_input("❓ Ask a question about your document...")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

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