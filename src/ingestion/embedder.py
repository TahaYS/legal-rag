"""
Embedder + Vector Store for Legal RAG
Converts text chunks into vectors using sentence-transformers
and stores them in ChromaDB for semantic search.

Model: all-MiniLM-L6-v2
- 384 dimensions (small and fast)
- Runs locally, no API needed
- Good quality for English legal text
- ~80MB download on first run
"""

import json
import os
import chromadb
from sentence_transformers import SentenceTransformer


# ── Configuration ───────────────────────────────────────────────

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_DB_PATH = "chroma_db"           # Where the vector database is stored
COLLECTION_NAME = "pakistan_penal_code"
CHUNKS_PATH = "data/processed/chunks.json"
BATCH_SIZE = 64                         # How many chunks to embed at once


def load_chunks(path: str) -> list[dict]:
    """Load chunks from the chunker output."""
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks from {path}")
    return chunks


def create_embedder(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    """
    Load the sentence-transformers model.
    First run downloads ~80MB. Subsequent runs load from cache.
    """
    print(f"Loading embedding model: {model_name}")
    print("(First run will download ~80MB, subsequent runs use cache)")
    model = SentenceTransformer(model_name)
    print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


def create_vector_store(db_path: str = CHROMA_DB_PATH) -> chromadb.PersistentClient:
    """
    Create a persistent ChromaDB client.
    Data is stored on disk so it survives between runs.
    """
    client = chromadb.PersistentClient(path=db_path)
    print(f"ChromaDB initialized at: {db_path}")
    return client


def embed_and_store(
    chunks: list[dict],
    model: SentenceTransformer,
    client: chromadb.PersistentClient,
    collection_name: str = COLLECTION_NAME,
):
    """
    Embed all chunks and store them in ChromaDB.
    
    ChromaDB stores three things per entry:
    1. The embedding vector (for similarity search)
    2. The document text (returned with search results)
    3. Metadata (section number, title, page — for citations)
    """
    # Delete existing collection if it exists (fresh start)
    try:
        client.delete_collection(collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "Pakistan Penal Code 1860 — chunked sections"},
    )
    print(f"Created collection: {collection_name}")

    # Process in batches to avoid memory issues
    total = len(chunks)
    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        batch_end = min(i + BATCH_SIZE, total)

        # Extract texts for embedding
        texts = [chunk["text"] for chunk in batch]

        # Generate embeddings
        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        # Prepare data for ChromaDB
        ids = [chunk["chunk_id"] for chunk in batch]
        metadatas = [
            {
                "section_number": chunk["section_number"],
                "section_title": chunk["section_title"],
                "page_number": chunk["page_number"],
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
                "total_chunks": chunk["total_chunks"],
                "text_length": len(chunk["text"]),
            }
            for chunk in batch
        ]

        # Store in ChromaDB
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        print(f"  Embedded and stored: {batch_end}/{total} chunks")

    print(f"\nDone! {total} chunks stored in ChromaDB.")
    return collection


def test_search(collection, model: SentenceTransformer, query: str, n_results: int = 5):
    """
    Test the vector search with a legal question.
    This is what the RAG pipeline will do — find relevant chunks
    for a user's question.
    """
    print(f"\n{'='*60}")
    print(f"Query: \"{query}\"")
    print(f"{'='*60}")

    # Embed the query
    query_embedding = model.encode([query]).tolist()

    # Search ChromaDB
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
    )

    # Display results
    for i in range(len(results["ids"][0])):
        doc = results["documents"][0][i]
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        
        # Lower distance = more similar
        similarity = round(1 - distance, 3)

        print(f"\n--- Result {i+1} (similarity: {similarity}) ---")
        print(f"Section {meta['section_number']}: {meta['section_title']}")
        print(f"Page: {meta['page_number']}")
        print(f"Preview: {doc[:200]}...")


# ── Main execution ──────────────────────────────────────────────

if __name__ == "__main__":

    # Step 1: Load chunks
    print("=== Loading chunks ===")
    chunks = load_chunks(CHUNKS_PATH)

    # Step 2: Load embedding model
    print("\n=== Loading embedding model ===")
    model = create_embedder()

    # Step 3: Create vector store and embed everything
    print("\n=== Embedding and storing in ChromaDB ===")
    client = create_vector_store()
    collection = embed_and_store(chunks, model, client)

    # Step 4: Test with sample legal queries
    print("\n\n" + "="*60)
    print("  TESTING SEMANTIC SEARCH ON PAKISTAN PENAL CODE")
    print("="*60)

    test_queries = [
        "What is the punishment for murder?",
        "What are the penalties for theft?",
        "What constitutes fraud under Pakistani law?",
        "What is the punishment for kidnapping?",
        "What are offences related to marriage?",
    ]

    for query in test_queries:
        test_search(collection, model, query, n_results=3)

    print("\n\n=== Phase 1 Complete! ===")
    print(f"Vector database saved at: {CHROMA_DB_PATH}/")
    print(f"Total chunks searchable: {collection.count()}")
    print("You can now search the Pakistan Penal Code by meaning!")
