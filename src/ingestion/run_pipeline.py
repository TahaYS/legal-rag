"""
Legal RAG — Ingestion Pipeline Runner
Runs the complete Phase 1 pipeline:
  PDF → Parse → Chunk → Embed → Store in ChromaDB

Usage:
    python src/ingestion/run_pipeline.py
"""

import os
import sys
import time

# Add project root to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.ingestion.pdf_parser import extract_text_from_pdf, extract_sections, save_extracted_data
from src.ingestion.chunker import chunk_sections, print_chunk_stats, save_chunks
from src.ingestion.embedder import create_embedder, create_vector_store, embed_and_store, test_search


def run_pipeline():
    """Run the complete ingestion pipeline."""
    
    start_time = time.time()
    
    # ── Configuration ───────────────────────────────────────────
    PDF_PATH = "data/pdfs/pakistan_penal_code.pdf"
    PROCESSED_DIR = "data/processed"
    CHROMA_DB_PATH = "chroma_db"
    
    print("=" * 60)
    print("  LEGAL RAG — INGESTION PIPELINE")
    print("=" * 60)
    
    # ── Step 1: Parse PDF ───────────────────────────────────────
    print("\n\n📄 STEP 1/4: Parsing PDF")
    print("-" * 40)
    
    if not os.path.exists(PDF_PATH):
        print(f"ERROR: PDF not found at {PDF_PATH}")
        print("Download the Pakistan Penal Code PDF first.")
        sys.exit(1)
    
    pages = extract_text_from_pdf(PDF_PATH)
    save_extracted_data(pages, os.path.join(PROCESSED_DIR, "raw_pages.json"))
    
    # ── Step 2: Extract sections ────────────────────────────────
    print("\n\n📑 STEP 2/4: Extracting legal sections")
    print("-" * 40)
    
    sections = extract_sections(pages)
    save_extracted_data(sections, os.path.join(PROCESSED_DIR, "sections.json"))
    
    # ── Step 3: Chunk sections ──────────────────────────────────
    print("\n\n✂️  STEP 3/4: Chunking sections")
    print("-" * 40)
    
    chunks = chunk_sections(sections)
    print_chunk_stats(chunks)
    save_chunks(chunks, os.path.join(PROCESSED_DIR, "chunks.json"))
    
    # ── Step 4: Embed and store ─────────────────────────────────
    print("\n\n🧠 STEP 4/4: Embedding and storing in ChromaDB")
    print("-" * 40)
    
    model = create_embedder()
    client = create_vector_store(CHROMA_DB_PATH)
    collection = embed_and_store(chunks, model, client)
    
    # ── Summary ─────────────────────────────────────────────────
    elapsed = time.time() - start_time
    
    print("\n\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Time taken:    {elapsed:.1f} seconds")
    print(f"  Pages parsed:  {len(pages)}")
    print(f"  Sections:      {len(sections)}")
    print(f"  Chunks stored: {collection.count()}")
    print(f"  Vector DB:     {CHROMA_DB_PATH}/")
    print("=" * 60)
    
    # ── Quick test ──────────────────────────────────────────────
    print("\n\nRunning quick search test...")
    test_search(collection, model, "What is the punishment for murder?", n_results=3)
    
    print("\n\nPhase 1 is complete! Your legal knowledge base is ready.")
    print("Next step: Phase 2 — building the RAG pipeline.")


if __name__ == "__main__":
    run_pipeline()
