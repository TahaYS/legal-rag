"""
Text Chunker for Legal Documents
Splits parsed legal sections into retrieval-optimized chunks.

Why these settings?
- chunk_size=800: Legal text is dense. 800 chars is roughly 1-2 paragraphs,
  enough to contain a complete legal clause with its explanation.
- chunk_overlap=200: Legal clauses often reference the sentence before them
  ("Whoever commits an offence under the preceding section..."). 
  Overlap ensures these references aren't cut off.
- We chunk within sections, not across them, so a chunk from Section 302
  never bleeds into Section 303.
"""

import json
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_chunker(chunk_size: int = 800, chunk_overlap: int = 200) -> RecursiveCharacterTextSplitter:
    """
    Create a text splitter tuned for legal documents.
    
    RecursiveCharacterTextSplitter tries to split on these separators 
    in order — it prefers splitting on double newlines (paragraphs),
    then single newlines, then sentences, then words. This keeps
    legal paragraphs intact when possible.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",   # Paragraph breaks (preferred split point)
            "\n",      # Line breaks
            ". ",      # Sentence endings
            "; ",      # Semicolons (common in legal lists)
            ", ",      # Commas
            " ",       # Words (last resort)
        ],
        length_function=len,
        is_separator_regex=False,
    )


def chunk_sections(sections: list[dict], chunk_size: int = 800, chunk_overlap: int = 200) -> list[dict]:
    """
    Split legal sections into smaller chunks for embedding.
    
    Each chunk preserves the metadata from its parent section
    (section number, title, source, page number) so the RAG
    system can cite sources accurately.
    
    Args:
        sections: List of section dicts from pdf_parser.py
        chunk_size: Target size of each chunk in characters
        chunk_overlap: Number of overlapping characters between chunks
        
    Returns:
        List of chunk dicts ready for embedding
    """
    splitter = create_chunker(chunk_size, chunk_overlap)
    chunks = []
    chunk_id = 0

    for section in sections:
        section_text = section["text"]
        
        # If the section is already small enough, keep it as one chunk
        if len(section_text) <= chunk_size:
            chunks.append({
                "chunk_id": f"ppc_s{section['section_number']}_{chunk_id}",
                "text": section_text,
                "section_number": section["section_number"],
                "section_title": section["title"],
                "page_number": section["page_number"],
                "source": section["source"],
                "chunk_index": 0,       # First (and only) chunk of this section
                "total_chunks": 1,      # This section produced 1 chunk
            })
            chunk_id += 1
        else:
            # Split the section into multiple chunks
            text_chunks = splitter.split_text(section_text)
            
            for i, text in enumerate(text_chunks):
                # Prepend section header to every chunk so the LLM
                # always knows which section this chunk belongs to
                header = f"[Section {section['section_number']}. {section['title']}]\n"
                
                # Only add header if it's not the first chunk
                # (first chunk already starts with the section heading)
                if i > 0:
                    text = header + text

                chunks.append({
                    "chunk_id": f"ppc_s{section['section_number']}_{chunk_id}",
                    "text": text,
                    "section_number": section["section_number"],
                    "section_title": section["title"],
                    "page_number": section["page_number"],
                    "source": section["source"],
                    "chunk_index": i,
                    "total_chunks": len(text_chunks),
                })
                chunk_id += 1

    print(f"Created {len(chunks)} chunks from {len(sections)} sections")
    return chunks


def print_chunk_stats(chunks: list[dict]):
    """Print useful statistics about the chunking results."""
    
    if not chunks:
        print("No chunks to analyze!")
        return
        
    sizes = [len(c["text"]) for c in chunks]
    
    # How many sections were split vs kept whole
    single_chunks = sum(1 for c in chunks if c["total_chunks"] == 1)
    multi_chunks = len(set(
        c["section_number"] for c in chunks if c["total_chunks"] > 1
    ))
    
    print(f"\n=== Chunking statistics ===")
    print(f"Total chunks: {len(chunks)}")
    print(f"Sections kept whole: {single_chunks}")
    print(f"Sections that were split: {multi_chunks}")
    print(f"Chunk sizes:")
    print(f"  Min:     {min(sizes)} chars")
    print(f"  Max:     {max(sizes)} chars")
    print(f"  Average: {sum(sizes) // len(sizes)} chars")
    print(f"  Median:  {sorted(sizes)[len(sizes) // 2]} chars")


def save_chunks(chunks: list[dict], output_path: str):
    """Save chunks as JSON for the embedding step."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(chunks)} chunks to {output_path}")


# ── Main execution ──────────────────────────────────────────────

if __name__ == "__main__":
    SECTIONS_PATH = "data/processed/sections.json"
    OUTPUT_PATH = "data/processed/chunks.json"

    # Load sections from the parser output
    if not os.path.exists(SECTIONS_PATH):
        print(f"ERROR: Sections file not found at {SECTIONS_PATH}")
        print("Run pdf_parser.py first to extract sections from the PDF.")
        exit(1)

    print("=== Loading parsed sections ===")
    with open(SECTIONS_PATH, "r", encoding="utf-8") as f:
        sections = json.load(f)
    print(f"Loaded {len(sections)} sections")

    # Chunk the sections
    print("\n=== Chunking sections ===")
    chunks = chunk_sections(sections)

    # Show statistics
    print_chunk_stats(chunks)

    # Preview some chunks
    print(f"\n=== Sample chunks ===")
    for chunk in chunks[:3]:
        print(f"\n--- Chunk: {chunk['chunk_id']} ---")
        print(f"Section: {chunk['section_number']} - {chunk['section_title']}")
        print(f"Size: {len(chunk['text'])} chars")
        print(f"Preview: {chunk['text'][:150]}...")

    # Save
    print("\n=== Saving chunks ===")
    save_chunks(chunks, OUTPUT_PATH)
