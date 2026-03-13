"""
Legal RAG — Streamlit Frontend
Chat interface for asking questions about the Pakistan Penal Code.

Usage (local):
    streamlit run src/frontend/streamlit_app.py

On Streamlit Cloud, the app auto-downloads the PDF and runs
the ingestion pipeline on first launch.
"""

import os
import sys
import time
import urllib.request

import streamlit as st

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


# ── Page Configuration ──────────────────────────────────────────

st.set_page_config(
    page_title="PPC Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom Styling ──────────────────────────────────────────────

st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .main-header h1 {
        font-size: 2rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    .main-header p {
        color: #888;
        font-size: 0.95rem;
    }
    
    /* Source citation cards */
    .source-card {
        background: var(--secondary-background-color);
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
        border-left: 3px solid #4A90D9;
    }
    .source-card .section-num {
        font-weight: 600;
        color: #4A90D9;
        font-size: 0.9rem;
    }
    .source-card .section-title {
        font-size: 0.85rem;
        color: #aaa;
    }
    .source-card .similarity {
        font-size: 0.75rem;
        color: #888;
    }
    
    /* Chat message styling */
    .stChatMessage {
        padding: 1rem;
    }
    
    /* Sidebar styling */
    .sidebar-info {
        background: var(--secondary-background-color);
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ── Auto-initialization for Streamlit Cloud ─────────────────────

PDF_URL = "https://www.unodc.org/cld/uploads/res/document/pak/1860/pakistan_penal_code_1860_html/Pakistan_Penal_Code_1860_incorporating_amendments_to_16_February_2017.pdf"
PDF_PATH = "data/pdfs/pakistan_penal_code.pdf"
CHROMA_DB_PATH = "chroma_db"


def initialize_knowledge_base():
    """
    Auto-download PDF and run the ingestion pipeline if ChromaDB
    doesn't exist yet. This handles first-run on Streamlit Cloud.
    """
    import chromadb
    
    # Check if ChromaDB already has data
    if os.path.exists(CHROMA_DB_PATH):
        try:
            client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            collection = client.get_collection("pakistan_penal_code")
            if collection.count() > 0:
                return  # Already initialized
        except Exception:
            pass  # Collection doesn't exist, need to initialize
    
    st.info("🔄 First run — setting up the legal knowledge base. This takes 2-3 minutes...")
    progress = st.progress(0, text="Starting...")
    
    # Step 1: Download PDF if not present
    if not os.path.exists(PDF_PATH):
        progress.progress(10, text="📥 Downloading Pakistan Penal Code PDF...")
        os.makedirs(os.path.dirname(PDF_PATH), exist_ok=True)
        try:
            urllib.request.urlretrieve(PDF_URL, PDF_PATH)
        except Exception as e:
            st.error(f"Failed to download PDF: {e}")
            st.stop()
    
    # Step 2: Parse PDF
    progress.progress(25, text="📄 Parsing PDF...")
    from src.ingestion.pdf_parser import extract_text_from_pdf, extract_sections, save_extracted_data
    
    pages = extract_text_from_pdf(PDF_PATH)
    os.makedirs("data/processed", exist_ok=True)
    save_extracted_data(pages, "data/processed/raw_pages.json")
    
    # Step 3: Extract sections
    progress.progress(40, text="📑 Extracting legal sections...")
    sections = extract_sections(pages)
    save_extracted_data(sections, "data/processed/sections.json")
    
    # Step 4: Chunk sections
    progress.progress(55, text="✂️ Chunking sections...")
    from src.ingestion.chunker import chunk_sections, save_chunks
    
    chunks = chunk_sections(sections)
    save_chunks(chunks, "data/processed/chunks.json")
    
    # Step 5: Embed and store
    progress.progress(70, text="🧠 Generating embeddings (this takes a minute)...")
    from src.ingestion.embedder import create_embedder, create_vector_store, embed_and_store
    
    model = create_embedder()
    client = create_vector_store(CHROMA_DB_PATH)
    embed_and_store(chunks, model, client)
    
    progress.progress(100, text="✅ Knowledge base ready!")
    time.sleep(1)
    progress.empty()


# Run initialization
initialize_knowledge_base()


# ── Load Pipeline (cached so it only loads once) ────────────────

@st.cache_resource(show_spinner=False)
def load_pipeline():
    """
    Load the RAG pipeline once and cache it.
    Subsequent reruns of the Streamlit app reuse the same pipeline.
    """
    from src.rag.rag_pipeline import LegalRAGPipeline
    return LegalRAGPipeline()


# Show a loading message while pipeline initializes
with st.spinner("Loading legal knowledge base... (this takes a few seconds on first run)"):
    pipeline = load_pipeline()


# ── Sidebar ─────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚖️ About")
    st.markdown(
        "This AI assistant answers questions about the "
        "**Pakistan Penal Code (PPC), 1860** using Retrieval-Augmented "
        "Generation (RAG)."
    )
    
    st.markdown("---")
    
    st.markdown("### How it works")
    st.markdown(
        "1. Your question is matched against **1,499 chunks** "
        "of the PPC using semantic search\n"
        "2. The most relevant sections are sent to an LLM\n"
        "3. The LLM generates a cited answer based only on "
        "the retrieved legal text"
    )
    
    st.markdown("---")
    
    # Settings
    st.markdown("### Settings")
    top_k = st.slider(
        "Number of sources to retrieve",
        min_value=1,
        max_value=10,
        value=5,
        help="More sources = more comprehensive but slower answers",
    )
    
    show_sources = st.toggle("Show source details", value=True)
    show_similarity = st.toggle("Show similarity scores", value=False)
    
    st.markdown("---")
    
    st.markdown("### Sample questions")
    sample_questions = [
        "What is the punishment for murder?",
        "What are the penalties for theft?",
        "Is blasphemy a crime in Pakistan?",
        "What happens if someone kidnaps a child?",
        "What if a public servant takes a bribe?",
        "Can someone go to jail for defamation?",
        "What are offences related to marriage?",
        "What is the punishment for forgery?",
    ]
    
    for q in sample_questions:
        if st.button(q, key=f"sample_{q}", use_container_width=True):
            st.session_state.sample_question = q
    
    st.markdown("---")
    
    # Stats
    chunk_count = pipeline.retriever.collection.count()
    st.markdown(
        f"**Knowledge base:** {chunk_count} chunks  \n"
        f"**Embedding model:** all-MiniLM-L6-v2  \n"
        f"**LLM:** Llama 3.1 8B (Groq)  \n"
        f"**Source:** Pakistan Penal Code, 1860"
    )


# ── Main Chat Area ──────────────────────────────────────────────

st.markdown(
    '<div class="main-header">'
    '<h1>⚖️ PPC Legal Assistant</h1>'
    '<p>Ask anything about the Pakistan Penal Code</p>'
    '</div>',
    unsafe_allow_html=True,
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "sample_question" not in st.session_state:
    st.session_state.sample_question = None

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show sources for assistant messages
        if message["role"] == "assistant" and "sources" in message and show_sources:
            with st.expander(f"📚 Sources ({len(message['sources'])} sections referenced)"):
                for source in message["sources"]:
                    similarity_text = ""
                    if show_similarity:
                        similarity_text = f" — Similarity: {source['similarity']}"
                    
                    st.markdown(
                        f'<div class="source-card">'
                        f'<span class="section-num">Section {source["section"]}</span>'
                        f'<span class="similarity">{similarity_text}</span><br>'
                        f'<span class="section-title">{source["title"][:80]}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )


# ── Handle Input ────────────────────────────────────────────────

# Check if a sample question was clicked
if st.session_state.sample_question:
    prompt = st.session_state.sample_question
    st.session_state.sample_question = None
else:
    prompt = st.chat_input("Ask a question about Pakistani criminal law...")

if prompt:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching the Pakistan Penal Code..."):
            start = time.time()
            result = pipeline.query(question=prompt, top_k=top_k)
            elapsed = round(time.time() - start, 2)
        
        # Display the answer
        st.markdown(result["answer"])
        
        # Display response time
        st.caption(f"⏱️ Response generated in {elapsed}s")
        
        # Display sources
        if show_sources and result["sources"]:
            with st.expander(f"📚 Sources ({len(result['sources'])} sections referenced)"):
                for source in result["sources"]:
                    similarity_text = ""
                    if show_similarity:
                        similarity_text = f" — Similarity: {source['similarity']}"
                    
                    st.markdown(
                        f'<div class="source-card">'
                        f'<span class="section-num">Section {source["section"]}</span>'
                        f'<span class="similarity">{similarity_text}</span><br>'
                        f'<span class="section-title">{source["title"][:80]}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
    
    # Save assistant message with sources to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })
