import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="StudyMate AI", page_icon="🎓", layout="wide")
st.title("🎓 StudyMate AI")
st.markdown("### Your Personalized Offline AI Study Companion")

PERSONAS = {
    "Summarize": "Summarize the following context into concise bullet points.",
    "Professor": "Explain the context like a friendly college professor using analogies and simple examples.",
    "Study Guide": "Extract key definitions and important concepts to create a structured study guide.",
    "Socratic Tutor": "Do NOT give the final answer. Guide the student using hints and thoughtful questions."
}

st.sidebar.header("⚙️ Settings")
mode = st.sidebar.radio("Choose Study Mode:", list(PERSONAS.keys()))
temperature = st.sidebar.slider("Creativity Level", 0.0, 1.0, 0.3)

uploaded_file = st.file_uploader("📄 Upload your study PDF", type="pdf")
user_query = st.text_input("❓ What do you want to learn?")

if uploaded_file and user_query:
    file_key = uploaded_file.name

    if "vectorstore" not in st.session_state or st.session_state.get("file_key") != file_key:
        with st.spinner("🔄 Processing your document..."):
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getvalue())

            loader = PyPDFLoader("temp.pdf")
            pages = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
            docs = text_splitter.split_documents(pages)

            embeddings = OllamaEmbeddings(model="mistral")
            vectorstore = FAISS.from_documents(docs, embeddings)

            st.session_state.vectorstore = vectorstore
            st.session_state.file_key = file_key
            os.remove("temp.pdf")
    else:
        vectorstore = st.session_state.vectorstore

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
    llm = ChatOllama(model="mistral", temperature=temperature)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

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