# 🎓 StudyMate AI

**Your Personalized AI Study Companion**

> 🚀 **Live App:** [https://prostudymate.streamlit.app/](https://prostudymate.streamlit.app/)

StudyMate AI is an intelligent study assistant that lets you learn from your own files, images, or any topic you choose — powered by Groq's fast LLMs and LangChain's RAG pipeline.

---

## ✨ Features

### 📚 Four Ways to Study

| Mode | Description |
|---|---|
| **📁 Upload a File** | Upload any document and chat with its contents using RAG |
| **🖼️ Upload Image** | Upload a photo of notes, diagrams, or textbook pages for AI analysis |
| **📷 Take a Photo** | Use your camera to snap anything and ask questions about it |
| **✏️ Enter a Topic** | No file needed — just type any topic and start learning |

### 🧠 Six AI Study Personas

| Persona | What it does |
|---|---|
| **Summarize** | Condenses content into concise bullet points |
| **Professor** | Explains concepts using analogies and simple examples |
| **Study Guide** | Extracts key definitions and builds structured study guides |
| **Socratic Tutor** | Guides you with hints and questions — never gives away the answer |
| **Quiz Master** | Tests your knowledge with interactive quizzes and score tracking |
| **ELI5 Simplifier** | Breaks down any topic as simply as possible — no jargon, just clarity |

### 📂 Supported File Types (30+)

| Category | Formats |
|---|---|
| **Documents** | PDF, Word (.docx/.doc), ODT, RTF |
| **Presentations** | PowerPoint (.pptx/.ppt) |
| **Spreadsheets** | Excel (.xlsx/.xls/.xlsm), ODS, CSV, TSV |
| **Code & Text** | .py, .js, .ts, .html, .css, .java, .c, .cpp, .go, .rs, .rb, .php, .swift, .md, .txt, .log |
| **Data & Config** | JSON, JSONL, XML, YAML, TOML, INI |
| **Notebooks** | Jupyter (.ipynb) |

### ⚙️ Additional Settings
- **Model selection** — choose from Llama 3.3 70B, Llama 3.1 8B, Gemma2 9B, or DeepSeek R1
- **Creativity slider** — adjust LLM temperature from precise (0.0) to creative (1.0)
- **Full conversation memory** — all personas remember your chat history within a session
- **Clear chat** — reset everything with one click from the sidebar

---

## 🛠️ Tech Stack

- **[Streamlit](https://streamlit.io/)** — UI framework
- **[Groq](https://groq.com/)** — LLM inference (fast!)
- **[LangChain](https://www.langchain.com/)** — RAG pipeline, prompt management, memory
- **[FAISS](https://github.com/facebookresearch/faiss)** — vector store for document retrieval
- **[HuggingFace Embeddings](https://huggingface.co/)** — `all-MiniLM-L6-v2` for text embeddings
- **[pypdf](https://pypdf.readthedocs.io/)** — PDF text extraction
- **[python-pptx](https://python-pptx.readthedocs.io/)** — PowerPoint extraction
- **[pandas](https://pandas.pydata.org/)** — spreadsheet parsing
- **[docx2txt](https://github.com/ankushshah89/python-docx2txt)** — Word document extraction

---

## 🚀 Running Locally

### 1. Clone the repo
```bash
git clone https://github.com/your-username/studymate-ai.git
cd studymate-ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your Groq API key

Create a `.streamlit/secrets.toml` file:
```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

Get a free API key at [console.groq.com](https://console.groq.com).

### 4. Run the app
```bash
streamlit run app.py
```

---

## 📦 requirements.txt

```
streamlit
langchain
langchain-community
langchain-groq
langchain-text-splitters
faiss-cpu
sentence-transformers
pypdf
pandas
openpyxl
xlrd
odfpy
python-pptx
docx2txt
groq
```

---

## 🌐 Deploying to Streamlit Cloud

1. Push your code to a public GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. Add `GROQ_API_KEY` under **App Settings → Secrets**
4. Deploy — that's it!

---

## 📄 License

MIT License — free to use, modify, and distribute.
