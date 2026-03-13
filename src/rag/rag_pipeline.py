"""
Legal RAG Pipeline
Connects ChromaDB retrieval with Groq LLM to answer legal questions
about the Pakistan Penal Code with citations.

Components:
1. Query Rewriter — expands casual language into legal terminology
2. Retriever — fetches relevant chunks from ChromaDB
3. Generator — sends context + question to Groq LLM for a cited answer
"""

import os
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# ── Configuration ───────────────────────────────────────────────

def get_groq_api_key():
    """
    Get the Groq API key from either:
    1. Streamlit Cloud secrets (when deployed)
    2. .env file (when running locally)
    """
    # Try Streamlit secrets first (for Streamlit Cloud deployment)
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    
    # Fall back to .env file
    return os.getenv("GROQ_API_KEY")


GROQ_API_KEY = get_groq_api_key()
GROQ_MODEL = "llama-3.1-8b-instant"    # Fast, free-tier friendly
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "pakistan_penal_code"
TOP_K = 5                               # Number of chunks to retrieve


# ── Query Rewriter ──────────────────────────────────────────────

# Maps common English terms to their PPC legal equivalents
# This fixes the "murder" → "qatl-i-amd" problem we saw in testing
LEGAL_TERM_MAP = {
    "murder": "murder qatl-i-amd qatl homicide causing death",
    "kill": "murder qatl-i-amd causing death",
    "stealing": "theft stolen property dishonest misappropriation",
    "steal": "theft stolen property dishonest misappropriation",
    "rape": "rape zina sexual assault",
    "bribe": "bribery gratification corruption public servant",
    "bribery": "bribery gratification corruption public servant",
    "drug": "intoxicating substance drugs narcotic",
    "drugs": "intoxicating substance drugs narcotic",
    "assault": "hurt grievous hurt assault force criminal",
    "beating": "hurt grievous hurt assault voluntarily causing",
    "fraud": "cheating dishonesty fraudulent deception",
    "forgery": "forgery forged document false",
    "defamation": "defamation imputation reputation",
    "kidnap": "kidnapping abduction abducting wrongful confinement",
    "kidnapping": "kidnapping abduction abducting wrongful confinement",
    "trespass": "trespass criminal house-breaking lurking",
    "robbery": "robbery extortion dacoity putting in fear",
    "arson": "mischief fire burning damage",
    "blasphemy": "blasphemy religion religious insult defilement",
    "treason": "waging war against Pakistan sedition",
    "terrorism": "terrorism criminal intimidation threat",
    "dowry": "dowry cruelty woman husband",
    "marriage": "marriage nikah fraudulent ceremony woman",
    "divorce": "marriage dissolution wife husband",
    "bail": "bail bond surety recognizance",
    "death penalty": "punishment death sentence qatl-i-amd",
    "fine": "fine punishment penalty sentence",
    "imprisonment": "imprisonment rigorous simple punishment sentence",
}


def rewrite_query(query: str) -> str:
    """
    Expand a user's casual question with legal terminology.
    
    Example:
        "what happens if someone kills another person?"
        → "what happens if someone kills another person? 
           murder qatl-i-amd qatl homicide causing death"
    
    This helps the embedding model find sections that use
    formal legal language instead of everyday words.
    """
    query_lower = query.lower()
    expansions = []

    for term, legal_terms in LEGAL_TERM_MAP.items():
        if term in query_lower:
            expansions.append(legal_terms)

    if expansions:
        expanded = query + " " + " ".join(expansions)
        return expanded

    return query


# ── Retriever ───────────────────────────────────────────────────

class LegalRetriever:
    """
    Retrieves relevant legal text chunks from ChromaDB.
    Uses the same embedding model that was used to store the chunks.
    """

    def __init__(
        self,
        db_path: str = CHROMA_DB_PATH,
        collection_name: str = COLLECTION_NAME,
        model_name: str = EMBEDDING_MODEL,
    ):
        print("Loading retriever...")
        self.model = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(collection_name)
        print(f"Retriever ready. Collection has {self.collection.count()} chunks.")

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """
        Find the most relevant chunks for a legal question.
        
        Args:
            query: User's question (already rewritten with legal terms)
            top_k: Number of results to return
            
        Returns:
            List of dicts with text, metadata, and similarity score
        """
        # Embed the query
        query_embedding = self.model.encode([query]).tolist()

        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
        )

        # Format results
        retrieved = []
        for i in range(len(results["ids"][0])):
            retrieved.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "similarity": round(1 - results["distances"][0][i], 3),
            })

        return retrieved


# ── Context Builder ─────────────────────────────────────────────

def build_context(retrieved_chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a context string for the LLM.
    
    Each chunk is wrapped with its source info so the LLM
    can cite specific sections in its answer.
    """
    context_parts = []

    for i, chunk in enumerate(retrieved_chunks, 1):
        meta = chunk["metadata"]
        section = meta["section_number"]
        title = meta["section_title"]
        page = meta["page_number"]

        context_parts.append(
            f"[Source {i}: Section {section} — {title} (Page {page})]\n"
            f"{chunk['text']}\n"
        )

    return "\n---\n".join(context_parts)


# ── LLM Generator ──────────────────────────────────────────────

# System prompt that instructs the LLM how to behave
SYSTEM_PROMPT = """You are a legal assistant specializing in the Pakistan Penal Code (PPC), 1860.

Your role:
- Answer questions about Pakistani criminal law based ONLY on the provided context
- Always cite specific PPC sections in your answer (e.g., "According to Section 302...")
- If the context doesn't contain enough information, say so honestly
- Use clear, simple language — the user may not be a lawyer
- When mentioning punishments, be specific about imprisonment terms and fine amounts
- If a section uses Urdu/Arabic legal terms (like qatl-i-amd), explain them in English

Rules:
- NEVER make up legal information. Only use what's in the provided context.
- NEVER give personal legal advice. You provide information, not counsel.
- Always end with a disclaimer that this is for informational purposes only.

Format your response as:
1. A clear answer to the question
2. Relevant section citations with brief explanations
3. A short disclaimer"""

HUMAN_PROMPT = """Based on the following excerpts from the Pakistan Penal Code, answer the user's question.

CONTEXT:
{context}

USER'S QUESTION:
{question}

Provide a clear, well-cited answer:"""


def create_generator():
    """
    Create the LLM chain using Groq (Llama 3.1).
    
    Chain: prompt template → Groq LLM → string output
    """
    api_key = get_groq_api_key()
    
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. "
            "Create a .env file with: GROQ_API_KEY=your_key_here "
            "or set it in Streamlit Cloud secrets."
        )

    llm = ChatGroq(
        api_key=api_key,
        model=GROQ_MODEL,
        temperature=0.1,        # Low temperature = more factual, less creative
        max_tokens=1024,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ])

    # LangChain Expression Language (LCEL) chain
    chain = prompt | llm | StrOutputParser()

    print(f"Generator ready. Model: {GROQ_MODEL}")
    return chain


# ── RAG Pipeline (ties everything together) ─────────────────────

class LegalRAGPipeline:
    """
    Complete RAG pipeline for Pakistan Penal Code queries.
    
    Flow:
        User question
        → Query rewriting (expand legal terms)
        → Retrieval (find relevant PPC sections)
        → Context building (format for LLM)
        → Generation (Groq LLM produces cited answer)
        → Response with sources
    """

    def __init__(self):
        print("\n=== Initializing Legal RAG Pipeline ===")
        self.retriever = LegalRetriever()
        self.generator = create_generator()
        print("Pipeline ready!\n")

    def query(self, question: str, top_k: int = TOP_K) -> dict:
        """
        Answer a legal question using RAG.
        
        Args:
            question: User's legal question in plain language
            top_k: Number of context chunks to retrieve
            
        Returns:
            Dict with answer, sources, and the rewritten query
        """
        # Step 1: Rewrite query with legal terms
        rewritten = rewrite_query(question)

        # Step 2: Retrieve relevant chunks
        retrieved = self.retriever.retrieve(rewritten, top_k=top_k)

        # Step 3: Build context from retrieved chunks
        context = build_context(retrieved)

        # Step 4: Generate answer with LLM
        answer = self.generator.invoke({
            "context": context,
            "question": question,  # Send original question, not rewritten
        })

        # Step 5: Format sources for citation
        sources = [
            {
                "section": chunk["metadata"]["section_number"],
                "title": chunk["metadata"]["section_title"],
                "page": chunk["metadata"]["page_number"],
                "similarity": chunk["similarity"],
                "preview": chunk["text"][:150],
            }
            for chunk in retrieved
        ]

        return {
            "question": question,
            "rewritten_query": rewritten,
            "answer": answer,
            "sources": sources,
        }


# ── Main: Interactive testing ───────────────────────────────────

if __name__ == "__main__":
    # Initialize pipeline
    pipeline = LegalRAGPipeline()

    # Test queries
    test_questions = [
        "What is the punishment for murder in Pakistan?",
        "What are the penalties for theft?",
        "What happens if someone kidnaps a child?",
    ]

    for question in test_questions:
        print("\n" + "=" * 60)
        print(f"QUESTION: {question}")
        print("=" * 60)

        result = pipeline.query(question)

        print(f"\nANSWER:\n{result['answer']}")

        print(f"\nSOURCES:")
        for s in result["sources"]:
            print(f"  - Section {s['section']}: {s['title']} "
                  f"(similarity: {s['similarity']})")

        print()

    # Interactive mode
    print("\n" + "=" * 60)
    print("  INTERACTIVE MODE — Ask anything about the PPC")
    print("  Type 'quit' to exit")
    print("=" * 60)

    while True:
        question = input("\nYour question: ").strip()

        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not question:
            continue

        result = pipeline.query(question)

        print(f"\n{result['answer']}")
        print(f"\n📚 Sources:")
        for s in result["sources"]:
            print(f"  - Section {s['section']}: {s['title']}")
