"""
Legal RAG — Streamlit Frontend
Chat interface for asking questions about the Pakistan Penal Code.

Usage:
    streamlit run src/frontend/streamlit_app.py

Note: This connects directly to the RAG pipeline (not through FastAPI)
so you don't need to run the API server separately. For deployment,
everything runs in one process.
"""

import os
import sys
import time

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
