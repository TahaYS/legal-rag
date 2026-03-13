"""
PDF Parser for Legal Documents
Extracts clean text from Pakistani legal PDFs with metadata.
"""

import fitz  # PyMuPDF
import re
import json
import os


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text from a legal PDF, page by page.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of dicts, each containing:
        - page_number: int
        - raw_text: str (original extracted text)
        - clean_text: str (cleaned version)
    """
    doc = fitz.open(pdf_path)
    pages = []

    print(f"Opened: {pdf_path}")
    print(f"Total pages: {len(doc)}")

    for page_num in range(len(doc)):
        page = doc[page_num]
        raw_text = page.get_text("text")

        # Clean the extracted text
        clean_text = clean_legal_text(raw_text, page_num + 1)

        if clean_text.strip():  # Only keep pages with actual content
            pages.append({
                "page_number": page_num + 1,
                "raw_text": raw_text,
                "clean_text": clean_text,
                "source": os.path.basename(pdf_path),
            })

    doc.close()
    print(f"Extracted text from {len(pages)} pages (skipped empty pages)")
    return pages


def clean_legal_text(text: str, page_number: int) -> str:
    """
    Clean extracted legal text by removing noise common in Pakistani legal PDFs.
    
    Args:
        text: Raw extracted text from a PDF page
        page_number: Current page number (used to remove page headers/footers)
        
    Returns:
        Cleaned text string
    """
    # Remove common PDF artifacts
    # 1. Remove page numbers (standalone numbers on a line)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

    # 2. Remove common headers/footers in legal PDFs
    text = re.sub(r'(?i)^\s*page\s+\d+\s+of\s+\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'(?i)^\s*the pakistan penal code\s*,?\s*1860\s*$', '', text, flags=re.MULTILINE)

    # 3. Remove footnote reference numbers (superscript-style: 1[, 2[, etc.)
    #    but keep the actual content inside brackets
    text = re.sub(r'(\d+)\[', '[', text)

    # 4. Remove excessive whitespace and blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines in a row
    text = re.sub(r' {2,}', ' ', text)       # Max 1 space in a row

    # 5. Remove the mid-dot separators common in legal PDFs
    text = re.sub(r'\s*·\s*', ' ', text)

    # 6. Fix common OCR/extraction artifacts
    text = text.replace('―', '—')    # Fix dash types
    text = text.replace('­', '-')     # Fix soft hyphens
    text = text.replace('\xad', '-')  # Another soft hyphen variant

    # 7. Remove lines that are just dashes or underscores (decorative separators)
    text = re.sub(r'^\s*[-_=]{3,}\s*$', '', text, flags=re.MULTILINE)

    return text.strip()


def extract_sections(pages: list[dict]) -> list[dict]:
    """
    Parse the cleaned text to identify individual legal sections.
    
    Pakistani legal codes follow a pattern:
    - Section numbers (e.g., "Section 302", "302.", "S. 302")
    - Section titles in bold or on their own line
    - Section body text
    
    Args:
        pages: List of page dicts from extract_text_from_pdf()
        
    Returns:
        List of section dicts with section_number, title, text, page_number
    """
    # Combine all pages into one text, keeping track of page boundaries
    full_text = ""
    page_boundaries = []  # (char_index, page_number)

    for page in pages:
        page_boundaries.append((len(full_text), page["page_number"]))
        full_text += page["clean_text"] + "\n\n"

    # Pattern to match section headings in Pakistani legal code
    # Matches: "302.", "302A.", "Section 302.", "S. 302"
    section_pattern = re.compile(
        r'(?:^|\n\n)'                          # Start of text or double newline
        r'\s*'                                   # Optional whitespace
        r'(?:Section\s+|S\.\s*)?'               # Optional "Section" or "S." prefix
        r'(\d+[A-Z]?)'                          # Section number (e.g., 302, 302A)
        r'\.\s*'                                 # Period after number
        r'([^\n]+)',                             # Section title (rest of the line)
        re.MULTILINE
    )

    matches = list(section_pattern.finditer(full_text))
    sections = []

    for i, match in enumerate(matches):
        section_number = match.group(1).strip()
        section_title = match.group(2).strip()

        # Get the text between this section and the next
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        section_body = full_text[start:end].strip()

        # Find which page this section starts on
        section_start_pos = match.start()
        page_number = 1
        for char_idx, pg_num in page_boundaries:
            if char_idx <= section_start_pos:
                page_number = pg_num
            else:
                break

        # Clean up the title (remove trailing periods, colons etc.)
        section_title = re.sub(r'[.:]+$', '', section_title).strip()

        sections.append({
            "section_number": section_number,
            "title": section_title,
            "text": f"Section {section_number}. {section_title}\n{section_body}",
            "page_number": page_number,
            "source": "Pakistan Penal Code, 1860",
        })

    print(f"Found {len(sections)} sections")
    return sections


def save_extracted_data(data: list[dict], output_path: str):
    """Save extracted data as JSON for the next pipeline step."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(data)} items to {output_path}")


# ── Main execution ──────────────────────────────────────────────

if __name__ == "__main__":
    PDF_PATH = "data/pdfs/pakistan_penal_code.pdf"
    OUTPUT_DIR = "data/processed"

    # Check if PDF exists
    if not os.path.exists(PDF_PATH):
        print(f"ERROR: PDF not found at {PDF_PATH}")
        print("Please download the Pakistan Penal Code PDF first.")
        print("See Step 2 in the project guide.")
        exit(1)

    # Step 1: Extract raw text from PDF
    print("\n=== Extracting text from PDF ===")
    pages = extract_text_from_pdf(PDF_PATH)

    # Save raw extracted pages
    save_extracted_data(pages, os.path.join(OUTPUT_DIR, "raw_pages.json"))

    # Step 2: Parse into individual sections
    print("\n=== Parsing legal sections ===")
    sections = extract_sections(pages)

    # Save parsed sections
    save_extracted_data(sections, os.path.join(OUTPUT_DIR, "sections.json"))

    # Print summary
    print("\n=== Summary ===")
    print(f"Total pages processed: {len(pages)}")
    print(f"Total sections found: {len(sections)}")

    if sections:
        print(f"\nFirst 5 sections:")
        for s in sections[:5]:
            preview = s['text'][:80].replace('\n', ' ')
            print(f"  Section {s['section_number']}: {s['title'][:50]}")

        print(f"\nLast 3 sections:")
        for s in sections[-3:]:
            print(f"  Section {s['section_number']}: {s['title'][:50]}")
