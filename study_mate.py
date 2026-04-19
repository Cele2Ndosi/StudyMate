import streamlit as st
import io
import base64
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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
    "Socratic Tutor": "You are a Socratic tutor. Do NOT give final answers. Guide the student using hints and thoughtful questions. Remember the student's previous responses and build your next question on them.",
    "Quiz Master": (
        "You are an engaging, encouraging, and sharp-witted quiz master. Your goal is to test knowledge, reinforce learning, and keep the user motivated. "
        "You ask clear, concise questions based on previous topics, provide immediate, constructive feedback on answers, and keep score if requested. "
        "You tailor the difficulty level to the student's progress and explain why an answer is correct or incorrect to deepen understanding."
    ),
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
    st.session_state.pop("vectorstore", None)
    st.session_state.pop("file_key", None)
    st.session_state.pop("image_data", None)
    st.session_state.pop("image_media_type", None)
    st.session_state.pop("free_topic", None)
    st.rerun()

# -----------------------------
# INITIALIZE SESSION STATE
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# INPUT MODE SELECTION
# -----------------------------
st.markdown("---")
input_mode = st.radio(
    "📚 How would you like to study today?",
    ["📄 Upload PDF", "🖼️ Upload Image", "📷 Take a Photo", "✏️ Enter a Topic"],
    horizontal=True
)

# ==============================
# MODE 1: PDF Upload
# ==============================
if input_mode == "📄 Upload PDF":
    uploaded_file = st.file_uploader("Upload your study PDF", type="pdf", label_visibility="collapsed")

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
                st.session_state.pop("image_data", None)
                st.session_state.pop("free_topic", None)

            st.success("✅ Document ready! Ask your first question below.")

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_query = st.chat_input("❓ Ask a question about your document...")

        if user_query:
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            llm = ChatGroq(model=model_choice, temperature=temperature, api_key=groq_api_key)
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})

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

            retrieved_docs = retriever.invoke(user_query)
            context = "\n\n".join(doc.page_content for doc in retrieved_docs)
            chain = prompt | llm | StrOutputParser()

            with st.chat_message("assistant"):
                with st.spinner("🧠 Thinking..."):
                    response = chain.invoke({
                        "context": context,
                        "chat_history": st.session_state.chat_history,
                        "question": user_query
                    })
                st.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=response))
    else:
        st.info("📄 Please upload a PDF to get started.")

# ==============================
# MODE 2 & 3: Image Upload / Camera
# ==============================
elif input_mode in ["🖼️ Upload Image", "📷 Take a Photo"]:
    image_data = None
    media_type = "image/jpeg"

    if input_mode == "🖼️ Upload Image":
        uploaded_img = st.file_uploader(
            "Upload an image (photo of notes, diagrams, textbook pages, etc.)",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed"
        )
        if uploaded_img:
            image_data = uploaded_img.read()
            media_type = uploaded_img.type or "image/jpeg"
            st.image(image_data, caption="Uploaded Image", use_container_width=True)

    else:  # Camera
        camera_image = st.camera_input("📷 Point your camera at anything you want to learn about!")
        if camera_image:
            image_data = camera_image.read()
            media_type = "image/jpeg"

    if image_data:
        # Store image in session if new
        img_hash = hash(image_data)
        if st.session_state.get("image_hash") != img_hash:
            st.session_state.image_data = base64.standard_b64encode(image_data).decode("utf-8")
            st.session_state.image_media_type = media_type
            st.session_state.image_hash = img_hash
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.pop("vectorstore", None)
            st.session_state.pop("free_topic", None)
            st.success("✅ Image ready! Ask anything about what's in the image.")

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_query = st.chat_input("❓ Ask a question about the image...")

        if user_query:
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            # Build multimodal message history
            # For the first turn, attach the image. For subsequent turns, rely on chat history text.
            llm = ChatGroq(model=model_choice, temperature=temperature, api_key=groq_api_key)

            system_prompt = (
                f"{PERSONAS[mode]}\n\n"
                "The student has shared an image. Analyze it thoroughly and answer questions about it. "
                "Use the visual content (text, diagrams, equations, labels, etc.) as your primary reference. "
                "If something is unclear in the image, say so honestly."
            )

            # Construct messages manually for vision support
            history_messages = st.session_state.chat_history.copy()

            # First message includes the image
            if not history_messages:
                first_content = [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{st.session_state.image_media_type};base64,{st.session_state.image_data}"
                        }
                    },
                    {"type": "text", "text": user_query}
                ]
                messages_to_send = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": first_content}
                ]
            else:
                # Subsequent messages: include image in system context reminder + new question
                messages_to_send = [{"role": "system", "content": system_prompt}]
                for msg in history_messages:
                    if isinstance(msg, HumanMessage):
                        messages_to_send.append({"role": "user", "content": msg.content})
                    elif isinstance(msg, AIMessage):
                        messages_to_send.append({"role": "assistant", "content": msg.content})
                messages_to_send.append({"role": "user", "content": user_query})

            with st.chat_message("assistant"):
                with st.spinner("🧠 Analyzing image..."):
                    from groq import Groq
                    client = Groq(api_key=groq_api_key)
                    completion = client.chat.completions.create(
                        model="meta-llama/llama-4-scout-17b-16e-instruct",  # vision-capable model
                        messages=messages_to_send,
                        temperature=temperature,
                        max_tokens=1024,
                    )
                    response = completion.choices[0].message.content
                st.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=response))

    else:
        if input_mode == "🖼️ Upload Image":
            st.info("🖼️ Upload an image of your notes, a diagram, a textbook page — anything you want to study!")
        else:
            st.info("📷 Use the camera above to capture anything you want to learn about.")

# ==============================
# MODE 4: Free Topic (No File)
# ==============================
elif input_mode == "✏️ Enter a Topic":
    st.markdown("#### 💡 What do you want to learn about today?")

    col1, col2 = st.columns([3, 1])
    with col1:
        topic_input = st.text_input(
            "Enter any topic, concept, or question:",
            placeholder="e.g. Photosynthesis, The French Revolution, Newton's Laws, Python decorators...",
            label_visibility="collapsed"
        )
    with col2:
        set_topic = st.button("🚀 Start Studying", use_container_width=True)

    if set_topic and topic_input.strip():
        st.session_state.free_topic = topic_input.strip()
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.pop("vectorstore", None)
        st.session_state.pop("image_data", None)
        st.rerun()

    current_topic = st.session_state.get("free_topic")

    if current_topic:
        st.success(f"📖 Currently studying: **{current_topic}**")

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        default_prompt = f"Let's start! Help me learn about: {current_topic}" if not st.session_state.messages else None
        user_query = st.chat_input("❓ Ask anything about this topic...") or (
            default_prompt if not st.session_state.messages else None
        )

        if user_query:
            if user_query != default_prompt:
                st.session_state.messages.append({"role": "user", "content": user_query})
                with st.chat_message("user"):
                    st.markdown(user_query)

            llm = ChatGroq(model=model_choice, temperature=temperature, api_key=groq_api_key)

            prompt = ChatPromptTemplate.from_messages([
                ("system", f"""{PERSONAS[mode]}

The student wants to learn about: {current_topic}

Use your broad knowledge to teach this topic thoroughly. 
If the student asks something off-topic, gently guide them back to {current_topic} unless it's a natural extension."""),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}")
            ])

            chain = prompt | llm | StrOutputParser()

            with st.chat_message("assistant"):
                with st.spinner("🧠 Thinking..."):
                    response = chain.invoke({
                        "chat_history": st.session_state.chat_history,
                        "question": user_query
                    })
                st.markdown(response)

            if user_query != default_prompt:
                st.session_state.chat_history.append(HumanMessage(content=user_query))
            else:
                st.session_state.messages.append({"role": "user", "content": user_query})
                with st.chat_message("user"):
                    st.markdown(user_query)

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.chat_history.append(AIMessage(content=response))
            st.rerun()

    else:
        st.info("✏️ Type a topic above and click **Start Studying** to begin — no files needed!")