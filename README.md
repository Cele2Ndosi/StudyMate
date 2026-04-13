🎓 StudyMate AI

Your Personalized AI Study Companion powered by LLMs and Retrieval-Augmented Generation (RAG)

🚀 Overview

StudyMate AI is an intelligent study assistant that lets you upload PDFs and interact with your study material using AI.

It uses Retrieval-Augmented Generation (RAG) to provide accurate, context-based answers and supports multiple learning modes like summarization, tutoring, and quizzes.

✨ Features
Upload PDFs and chat with your notes
Multiple study modes:
Summarize – concise bullet summaries
Professor – detailed explanations with examples
Study Guide – structured notes and key concepts
Socratic Tutor – guided learning through questions
Quiz Master – interactive quizzes with feedback
Context-aware answers using FAISS vector search
Conversation memory for better responses
Fast responses powered by Groq LLMs
🛠️ Tech Stack
Streamlit
LangChain
Groq (LLaMA, Gemma, DeepSeek models)
HuggingFace Embeddings (all-MiniLM-L6-v2)
FAISS (Vector Database)
PyPDF
🧩 How It Works
Upload a PDF
Text is extracted and split into chunks
Chunks are converted into embeddings
Stored in a FAISS vector database
Relevant content is retrieved for each query
AI generates a response based on the context
📦 Installation

Clone the repository:
git clone https://github.com/Cele2Ndosi/StudyMate.git

Navigate into the project:
cd StudyMate

Create virtual environment:
python -m venv venv

Activate environment:
Windows: venv\Scripts\activate
Mac/Linux: source venv/bin/activate

Install dependencies:
pip install -r requirements.txt

⚙️ Environment Setup

Create a file at:
.streamlit/secrets.toml

Add:
GROQ_API_KEY = "your_api_key_here"

▶️ Run the App

streamlit run app.py

🧪 Usage
Upload your study PDF
Select a study mode from the sidebar
Ask questions about the document
Get AI-powered answers based on your content
Continue chatting with memory
⚠️ Limitations
Works best with text-based PDFs
Scanned/image PDFs may not work properly
🔮 Future Improvements
OCR support for scanned PDFs
Voice-based interaction
Mobile optimization
Saved user sessions
👨‍💻 Author

Menzi Prosper Cele
https://github.com/Cele2Ndosi

⭐ Support

If you like this project:

Star the repo
Fork it
Share it