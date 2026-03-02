# 🎓 StudyMate AI

> **Your Personalized Offline AI Study Companion** — powered by local LLMs via Ollama, LangChain, and Streamlit.

---

## 📖 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the App](#running-the-app)
- [Usage Guide](#usage-guide)
- [Study Modes](#study-modes)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## 🧠 Overview

StudyMate AI is a fully **offline**, privacy-first AI-powered study tool. Upload any PDF — textbook, research paper, lecture notes — ask questions about it, and get intelligent, context-aware responses using your local machine's computing power.

No internet. No API keys. No data leaves your computer.

It uses **Mistral** (via Ollama) as the LLM backbone, **FAISS** for fast vector similarity search, and **Streamlit** for a clean, interactive UI.

---

## ✨ Features

| Feature | Description |
|--------|-------------|
| 📄 **PDF Upload** | Upload any study material in PDF format |
| 🤖 **Local LLM** | Runs entirely offline using Ollama + Mistral |
| 🔍 **RAG Pipeline** | Retrieval-Augmented Generation for accurate, grounded answers |
| 🎭 **Study Modes** | 4 persona-based modes tailored to different learning styles |
| 🌡️ **Creativity Slider** | Adjust the LLM temperature for more factual or creative responses |
| ⚡ **Session Caching** | Avoids reprocessing the same PDF on every query |
| 🔒 **Privacy First** | All processing happens locally — no data sent to the cloud |

---

## 🖥️ Demo

```
1. Upload a PDF (e.g., a biology textbook chapter)
2. Select "Professor" mode
3. Ask: "Explain how mitosis works"
4. Get a clear, analogy-driven explanation based only on your document
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **UI** | [Streamlit](https://streamlit.io/) |
| **LLM** | [Ollama](https://ollama.com/) + [Mistral 7B](https://mistral.ai/) |
| **Embeddings** | Ollama Embeddings (`mistral`) |
| **Vector Store** | [FAISS](https://github.com/facebookresearch/faiss) |
| **RAG Framework** | [LangChain](https://www.langchain.com/) |
| **PDF Loader** | LangChain `PyPDFLoader` |
| **Text Splitter** | `RecursiveCharacterTextSplitter` |

---

## ✅ Prerequisites

Before running this app, make sure you have the following installed:

### 1. Python 3.9+
Download from [python.org](https://www.python.org/downloads/)

### 2. Ollama
Download and install from [ollama.com](https://ollama.com/download)

### 3. Mistral Model
After installing Ollama, pull the Mistral model:
```bash
ollama pull mistral
```

---

## 📦 Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/studymate-ai.git
cd studymate-ai
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```
streamlit
langchain
langchain-community
langchain-ollama
langchain-text-splitters
langchain-core
faiss-cpu
pypdf
```

---

## 🚀 Running the App

### Step 1: Start Ollama (if not already running)
```bash
ollama serve
```

> ⚠️ If you see `Error: bind: Only one usage of each socket address`, Ollama is **already running** — skip this step.

You can verify Ollama is running by visiting: [http://localhost:11434](http://localhost:11434)

### Step 2: Launch the Streamlit App
```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

---

## 📚 Usage Guide

1. **Upload a PDF** — Click the file uploader and select your study material
2. **Choose a Study Mode** — Select a persona from the sidebar (see below)
3. **Adjust Creativity** — Use the slider to control response style (0 = factual, 1 = creative)
4. **Ask a Question** — Type your question in the input box
5. **Get Your Answer** — StudyMate will retrieve relevant context and generate a response

> 💡 **Tip:** The app caches your PDF per session, so you can ask multiple questions without re-uploading or reprocessing.

---

## 🎭 Study Modes

| Mode | Best For | Behavior |
|------|----------|----------|
| 📝 **Summarize** | Quick review | Returns concise bullet-point summaries |
| 👨‍🏫 **Professor** | Deep understanding | Explains concepts with analogies and simple language |
| 📖 **Study Guide** | Exam prep | Extracts key terms, definitions, and structured notes |
| 🤔 **Socratic Tutor** | Active learning | Guides you with hints and questions instead of direct answers |

---

## 📁 Project Structure

```
studymate-ai/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── temp.pdf                # Temporary file (auto-deleted after processing)
```

---

## ⚙️ Configuration

You can tweak the following parameters directly in `app.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | `400` | Size of each text chunk for splitting |
| `chunk_overlap` | `50` | Overlap between chunks to preserve context |
| `k` (retriever) | `2` | Number of document chunks retrieved per query |
| `model` | `"mistral"` | Ollama model used for both embeddings and chat |
| `temperature` | `0.3` | Default creativity level (adjustable via slider) |

### Using a Different Model

To swap Mistral for another model (e.g., LLaMA 3, Gemma):

```bash
ollama pull llama3
```

Then update `app.py`:
```python
embeddings = OllamaEmbeddings(model="llama3")
llm = ChatOllama(model="llama3", temperature=temperature)
```

---

## 🐛 Troubleshooting

### ❌ `ollama serve` — Address already in use
Ollama is already running. Just proceed to `streamlit run app.py`.

### ❌ `ModuleNotFoundError: No module named 'langchain_ollama'`
```bash
pip install langchain-ollama
```

### ❌ `ModuleNotFoundError: No module named 'faiss'`
```bash
pip install faiss-cpu
```

### ❌ App is very slow on large PDFs
- Reduce `chunk_size` to `200–300`
- Use a smaller/faster model like `phi3` or `gemma:2b`
- Ensure no other heavy processes are running

### ❌ Response says "answer not found in document"
- The query may not match the document content closely enough
- Try rephrasing your question using terms directly from the PDF
- Increase `k` from `2` to `4` in the retriever settings

### ❌ Ollama model not found
```bash
ollama list          # Check installed models
ollama pull mistral  # Re-pull if missing
```

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Make your changes and commit: `git commit -m "Add your feature"`
4. Push to your branch: `git push origin feature/your-feature-name`
5. Open a Pull Request

### Ideas for Contributions
- [ ] Support for multiple PDF uploads
- [ ] Chat history with memory
- [ ] Export study notes to `.docx` or `.pdf`
- [ ] Support for `.txt` and `.epub` files
- [ ] Model selector dropdown in UI
- [ ] Dark/light theme toggle

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Ollama](https://ollama.com/) — for making local LLMs easy to run
- [Mistral AI](https://mistral.ai/) — for the open-weight Mistral model
- [LangChain](https://www.langchain.com/) — for the RAG framework
- [Streamlit](https://streamlit.io/) — for the rapid UI framework
- [FAISS](https://github.com/facebookresearch/faiss) — for fast vector search

---

<p align="center">Made with ❤️ for students, by students.</p>
