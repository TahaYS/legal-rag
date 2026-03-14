# ⚖️ PPC Legal Assistant

An AI-powered legal assistant that answers questions about the **Pakistan Penal Code (PPC), 1860** using Retrieval-Augmented Generation (RAG). Ask questions in plain English and get accurate, cited answers backed by actual legal text.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![Groq](https://img.shields.io/badge/Groq-Llama_3.1-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-teal)

🔗 **[Try the Live Demo](https://legal-rag-phsvdctnkabk3su5qyf7ew.streamlit.app/)**

---

## 🎯 What it does

Type a legal question like *"What is the punishment for theft?"* and the system:

1. **Searches** 1,499 chunks of the Pakistan Penal Code using semantic similarity
2. **Retrieves** the most relevant legal sections
3. **Generates** a clear, cited answer using an LLM — grounded only in actual PPC text

> **Example:**  
> **Q:** *What is the punishment for murder in Pakistan?*  
> **A:** According to Section 302 of the PPC, the punishment for qatl-e-amd (murder) can be: death as qisas if proof is available, death or imprisonment for life as ta'zir if proof is not available, or imprisonment up to twenty-five years where qisas is not applicable.

---

## 🏗️ Architecture

```
User Question
    │
    ▼
┌─────────────────┐
│  Query Rewriter  │ ── Expands casual language → legal terminology
└────────┬────────┘    ("murder" → "qatl-i-amd, homicide, causing death")
         │
         ▼
┌─────────────────┐
│    Retriever     │ ── Semantic search over 1,499 PPC chunks
│   (ChromaDB)     │    using sentence-transformers embeddings
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLM Generator   │ ── Groq API (Llama 3.1 8B)
│  (LangChain)     │    Generates cited answer from retrieved context
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Streamlit UI   │ ── Chat interface with source citations
└─────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| PDF Parsing | PyMuPDF | Extract text from legal PDFs |
| Text Splitting | LangChain RecursiveCharacterTextSplitter | Chunk documents for retrieval |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Local, free vector embeddings |
| Vector Store | ChromaDB | Persistent local vector database |
| LLM | Groq API (Llama 3.1 8B Instant) | Fast, free-tier LLM inference |
| RAG Framework | LangChain + LCEL | Chain retrieval with generation |
| Backend API | FastAPI | REST API with auto-generated docs |
| Frontend | Streamlit | Chat interface with citations |

---

## 📁 Project Structure

```
legal-rag/
├── data/
│   ├── pdfs/                  # Source legal PDFs
│   └── processed/             # Parsed sections and chunks (JSON)
├── src/
│   ├── ingestion/
│   │   ├── pdf_parser.py      # PDF text extraction + section parsing
│   │   ├── chunker.py         # Text chunking with overlap
│   │   ├── embedder.py        # Embedding + ChromaDB storage
│   │   └── run_pipeline.py    # One-command ingestion pipeline
│   ├── rag/
│   │   └── rag_pipeline.py    # Query rewriting, retrieval, LLM generation
│   ├── api/
│   │   └── app.py             # FastAPI REST endpoints
│   └── frontend/
│       └── streamlit_app.py   # Streamlit chat interface
├── chroma_db/                 # Vector database (auto-generated)
├── requirements.txt
├── .env                       # API keys (not committed)
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- [Groq API key](https://console.groq.com) (free)

### 1. Clone and install

```bash
git clone https://github.com/TahaYS/legal-rag.git
cd legal-rag
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set up environment

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Download the Pakistan Penal Code PDF

```bash
powershell -Command "Invoke-WebRequest -Uri 'https://www.unodc.org/cld/uploads/res/document/pak/1860/pakistan_penal_code_1860_html/Pakistan_Penal_Code_1860_incorporating_amendments_to_16_February_2017.pdf' -OutFile 'data/pdfs/pakistan_penal_code.pdf'"
```

### 4. Run the ingestion pipeline

```bash
python src/ingestion/run_pipeline.py
```

This parses the PDF, chunks the text, generates embeddings, and stores everything in ChromaDB. Takes 2-3 minutes on first run.

### 5. Launch the app

**Streamlit UI (recommended):**
```bash
streamlit run src/frontend/streamlit_app.py
```

**FastAPI backend (for API access):**
```bash
python src/api/app.py
# API docs at http://localhost:8000/docs
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/query` | Ask a legal question |
| GET | `/health` | Health check |
| GET | `/stats` | Knowledge base statistics |

**Example request:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the punishment for theft?"}'
```

---

## 🔍 How RAG Works in This Project

**The Problem:** LLMs can hallucinate legal information. You can't trust a generic chatbot to give accurate legal answers.

**The Solution:** Instead of relying on the LLM's training data, we:
1. **Parse** the actual Pakistan Penal Code PDF into 1,499 searchable chunks
2. **Embed** each chunk as a vector using sentence-transformers
3. **Retrieve** the most relevant chunks when a user asks a question
4. **Generate** an answer using only the retrieved legal text as context
5. **Cite** specific PPC sections so the user can verify the answer

This ensures every answer is grounded in real legal text, not hallucinated.

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Knowledge base | 1,499 chunks from 177 pages |
| Embedding model | all-MiniLM-L6-v2 (384 dimensions) |
| Average response time | ~1-2 seconds |
| LLM | Llama 3.1 8B via Groq (free tier) |
| Total cost | $0 (all free/open-source) |

---

## ⚠️ Disclaimer

This tool is for **informational purposes only** and does not constitute legal advice. Always consult a qualified legal professional for specific legal guidance. The system's answers are based on the Pakistan Penal Code, 1860 (with amendments up to 2017) and may not reflect the most recent legal changes.

---

## 📄 License

MIT License — feel free to use, modify, and distribute.