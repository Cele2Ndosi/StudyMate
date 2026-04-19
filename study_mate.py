import streamlit as st
import io
import base64
import os
import json
import warnings
import pypdf
import pandas as pd

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
    "ELI5 Simplifier": (
        "You are an expert at explaining complex topics in the simplest way possible — as if explaining to a curious 10-year-old. "
        "Use everyday analogies, short sentences, and familiar examples. Avoid jargon entirely; if a technical term is unavoidable, immediately explain it in plain language. "
        "Break everything down into small, digestible pieces. Use comparisons to things from daily life (food, games, sports, toys) to make abstract ideas concrete. "
        "Keep a warm, encouraging, and enthusiastic tone — learning should feel easy and fun, never overwhelming. "
        "If the student seems confused, try a completely different analogy. Never say 'it's complicated' — everything can be made simple."
    ),
}

# -----------------------------
# SUPPORTED FILE TYPES
# -----------------------------
SUPPORTED_EXTENSIONS = [
    # Documents
    "pdf", "docx", "doc", "odt", "rtf",
    # Presentations
    "pptx", "ppt",
    # Spreadsheets
    "xlsx", "xls", "xlsm", "ods", "csv", "tsv",
    # Text / Code / Data
    "txt", "md", "py", "js", "ts", "html", "css", "java",
    "c", "cpp", "cs", "go", "rs", "rb", "php", "swift",
    "json", "jsonl", "xml", "yaml", "yml", "toml", "ini", "log",
    # Notebooks
    "ipynb",
]

FILE_TYPE_LABELS = {
    "pdf": "📄 PDF", "docx": "📝 Word", "doc": "📝 Word (legacy)",
    "odt": "📝 OpenDocument Text", "rtf": "📝 Rich Text",
    "pptx": "📊 PowerPoint", "ppt": "📊 PowerPoint (legacy)",
    "xlsx": "📈 Excel", "xls": "📈 Excel (legacy)",
    "xlsm": "📈 Excel (macro)", "ods": "📈 OpenDocument Sheet",
    "csv": "📋 CSV", "tsv": "📋 TSV",
    "txt": "📃 Text", "md": "📃 Markdown", "ipynb": "📓 Jupyter Notebook",
    "json": "🔧 JSON", "jsonl": "🔧 JSONL", "xml": "🔧 XML",
    "yaml": "🔧 YAML", "yml": "🔧 YAML", "toml": "🔧 TOML",
    "ini": "🔧 Config", "log": "📋 Log",
    "py": "🐍 Python", "js": "⚡ JavaScript", "ts": "⚡ TypeScript",
    "html": "🌐 HTML", "css": "🎨 CSS", "java": "☕ Java",
    "c": "⚙️ C", "cpp": "⚙️ C++", "cs": "⚙️ C#",
    "go": "🐹 Go", "rs": "🦀 Rust", "rb": "💎 Ruby",
    "php": "🐘 PHP", "swift": "🍎 Swift",
}


# -----------------------------------------------
# TEXT EXTRACTION — handles all supported formats
# -----------------------------------------------
def extract_text_from_file(uploaded_file) -> str:
    """Extract plain text from any supported file type."""
    filename = uploaded_file.name
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    raw = uploaded_file.read()
    buf = io.BytesIO(raw)

    # PDF
    if ext == "pdf":
        reader = pypdf.PdfReader(buf)
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append(f"[Page {i+1}]\n{text.strip()}")
        if not pages:
            raise ValueError("Could not extract text from this PDF. It may be scanned or image-based.")
        return "\n\n".join(pages)

    # Word / ODT / RTF / EPUB — try docx2txt then pandoc fallback
    elif ext in ("docx", "odt", "rtf", "epub"):
        tmp = f"/tmp/_studymate_upload.{ext}"
        with open(tmp, "wb") as f:
            f.write(raw)
        try:
            import docx2txt
            text = docx2txt.process(tmp)
            if text and text.strip():
                return text.strip()
        except Exception:
            pass
        try:
            import subprocess
            result = subprocess.run(
                ["pandoc", tmp, "-t", "plain"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass
        raise ValueError(f"Could not extract text from {ext.upper()} file. Try converting to PDF or TXT first.")

    # PowerPoint
    elif ext in ("pptx", "ppt"):
        try:
            from pptx import Presentation
            prs = Presentation(buf)
            slides = []
            for i, slide in enumerate(prs.slides):
                texts = [
                    shape.text.strip()
                    for shape in slide.shapes
                    if hasattr(shape, "text") and shape.text.strip()
                ]
                if texts:
                    slides.append(f"[Slide {i+1}]\n" + "\n".join(texts))
            if not slides:
                raise ValueError("No text found in presentation.")
            return "\n\n".join(slides)
        except ImportError:
            raise ValueError("python-pptx is required. Run: pip install python-pptx")

    # Excel / ODS
    elif ext in ("xlsx", "xlsm"):
        df_dict = pd.read_excel(buf, sheet_name=None, engine="openpyxl")
        return "\n\n".join(f"[Sheet: {name}]\n{df.to_string(index=False)}" for name, df in df_dict.items())

    elif ext == "ods":
        df_dict = pd.read_excel(buf, sheet_name=None, engine="odf")
        return "\n\n".join(f"[Sheet: {name}]\n{df.to_string(index=False)}" for name, df in df_dict.items())

    elif ext == "xls":
        df_dict = pd.read_excel(buf, sheet_name=None, engine="xlrd")
        return "\n\n".join(f"[Sheet: {name}]\n{df.to_string(index=False)}" for name, df in df_dict.items())

    # CSV / TSV
    elif ext in ("csv", "tsv"):
        sep = "\t" if ext == "tsv" else ","
        df = pd.read_csv(buf, sep=sep)
        return df.to_string(index=False)

    # JSON
    elif ext == "json":
        data = json.load(buf)
        return json.dumps(data, indent=2)

    # JSONL
    elif ext == "jsonl":
        lines = raw.decode("utf-8", errors="replace").strip().split("\n")
        parsed = []
        for line in lines[:500]:
            try:
                parsed.append(json.dumps(json.loads(line)))
            except Exception:
                parsed.append(line)
        return "\n".join(parsed)

    # Jupyter Notebook
    elif ext == "ipynb":
        nb = json.loads(raw.decode("utf-8", errors="replace"))
        cells = []
        for cell in nb.get("cells", []):
            ct = cell.get("cell_type", "")
            src = "".join(cell.get("source", []))
            if src.strip():
                label = "# CODE CELL" if ct == "code" else "# MARKDOWN"
                cells.append(f"{label}\n{src.strip()}")
        return "\n\n".join(cells)

    # Plain text, code, config, markup
    elif ext in (
        "txt", "md", "py", "js", "ts", "html", "css", "java",
        "c", "cpp", "cs", "go", "rs", "rb", "php", "swift",
        "xml", "yaml", "yml", "toml", "ini", "log",
    ):
        return raw.decode("utf-8", errors="replace")

    else:
        # Last-ditch UTF-8 decode
        decoded = raw.decode("utf-8", errors="replace")
        if decoded.strip():
            return decoded
        raise ValueError(f"Unsupported or unreadable file type: .{ext}")


def build_vectorstore(text: str, source_name: str):
    """Chunk text and build a FAISS vectorstore."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    docs = text_splitter.create_documents([text], metadatas=[{"source": source_name}])
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)


def run_rag_chat(user_query: str, retriever, llm, persona: str) -> str:
    """Retrieve relevant chunks and run the LLM chain."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""{persona}

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
    return chain.invoke({
        "context": context,
        "chat_history": st.session_state.chat_history,
        "question": user_query
    })


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
    for key in ["messages", "chat_history", "vectorstore", "file_key",
                "image_data", "image_media_type", "image_hash", "free_topic"]:
        st.session_state.pop(key, None)
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.rerun()

with st.sidebar.expander("📂 Supported File Types"):
    st.markdown("""
**Documents:** PDF, Word (.docx/.doc), ODT, RTF  
**Presentations:** PowerPoint (.pptx/.ppt)  
**Spreadsheets:** Excel (.xlsx/.xls/.xlsm), ODS, CSV, TSV  
**Text & Code:** TXT, MD, Python, JS, TS, HTML, CSS, Java, C/C++, C#, Go, Rust, Ruby, PHP, Swift  
**Data:** JSON, JSONL, XML, YAML, TOML, INI, LOG  
**Notebooks:** Jupyter (.ipynb)
""")

# -----------------------------
# SESSION STATE INIT
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
    ["📁 Upload a File", "🖼️ Upload Image", "📷 Take a Photo", "✏️ Enter a Topic"],
    horizontal=True
)

# ==============================
# MODE 1: Universal File Upload
# ==============================
if input_mode == "📁 Upload a File":
    uploaded_file = st.file_uploader(
        "Upload any document, spreadsheet, code file, or data file",
        type=SUPPORTED_EXTENSIONS,
        label_visibility="collapsed",
    )

    if uploaded_file:
        ext = uploaded_file.name.rsplit(".", 1)[-1].lower() if "." in uploaded_file.name else "?"
        file_label = FILE_TYPE_LABELS.get(ext, f"📄 .{ext.upper()}")
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"

        if "vectorstore" not in st.session_state or st.session_state.get("file_key") != file_key:
            with st.spinner(f"🔄 Processing your {file_label} file..."):
                try:
                    raw_text = extract_text_from_file(uploaded_file)
                except ValueError as e:
                    st.error(f"❌ {e}")
                    st.stop()
                except Exception as e:
                    st.error(f"❌ Unexpected error reading file: {e}")
                    st.stop()

                if not raw_text or not raw_text.strip():
                    st.error("❌ No readable text could be extracted from this file.")
                    st.stop()

                vectorstore = build_vectorstore(raw_text, uploaded_file.name)

                st.session_state.vectorstore = vectorstore
                st.session_state.file_key = file_key
                st.session_state.messages = []
                st.session_state.chat_history = []
                st.session_state.pop("image_data", None)
                st.session_state.pop("free_topic", None)

            word_count = len(raw_text.split())
            st.success(f"✅ {file_label} ready! (~{word_count:,} words extracted). Ask your first question below.")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_query = st.chat_input("❓ Ask a question about your file...")

        if user_query:
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            llm = ChatGroq(model=model_choice, temperature=temperature, api_key=groq_api_key)
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})

            with st.chat_message("assistant"):
                with st.spinner("🧠 Thinking..."):
                    response = run_rag_chat(user_query, retriever, llm, PERSONAS[mode])
                st.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=response))

    else:
        st.info("📁 Upload any file to get started. See **Supported File Types** in the sidebar for the full list.")

# ==============================
# MODE 2 & 3: Image / Camera
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
    else:
        camera_image = st.camera_input("📷 Point your camera at anything you want to learn about!")
        if camera_image:
            image_data = camera_image.read()
            media_type = "image/jpeg"

    if image_data:
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

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_query = st.chat_input("❓ Ask a question about the image...")

        if user_query:
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            system_prompt = (
                f"{PERSONAS[mode]}\n\n"
                "The student has shared an image. Analyze it thoroughly and answer questions about it. "
                "Use the visual content (text, diagrams, equations, labels, etc.) as your primary reference. "
                "If something is unclear in the image, say so honestly."
            )

            history_messages = st.session_state.chat_history.copy()

            if not history_messages:
                first_content = [
                    {"type": "image_url", "image_url": {
                        "url": f"data:{st.session_state.image_media_type};base64,{st.session_state.image_data}"
                    }},
                    {"type": "text", "text": user_query}
                ]
                messages_to_send = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": first_content}
                ]
            else:
                messages_to_send = [{"role": "system", "content": system_prompt}]
                for msg in history_messages:
                    role = "user" if isinstance(msg, HumanMessage) else "assistant"
                    messages_to_send.append({"role": role, "content": msg.content})
                messages_to_send.append({"role": "user", "content": user_query})

            with st.chat_message("assistant"):
                with st.spinner("🧠 Analyzing image..."):
                    from groq import Groq
                    client = Groq(api_key=groq_api_key)
                    completion = client.chat.completions.create(
                        model="meta-llama/llama-4-scout-17b-16e-instruct",
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
# MODE 4: Free Topic
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

            if user_query == default_prompt:
                st.session_state.messages.append({"role": "user", "content": user_query})

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=response))
            st.rerun()

    else:
        st.info("✏️ Type a topic above and click **Start Studying** to begin — no files needed!")