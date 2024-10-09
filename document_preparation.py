"""
document_preparation.py

This script prepares the 'Attention is All You Need' paper for embedding.
It extracts text from the PDF, preprocesses it, and splits it into chunks.

Dependencies:
- PyPDF2
- nltk

Usage:
python document_preparation.py input.pdf output.txt
"""

import sys
import re
import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize
from typing import List, Tuple

# Download necessary NLTK data
nltk.download('punkt', quiet=True)

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file.

    Args:
    pdf_path (str): Path to the PDF file.

    Returns:
    str: Extracted text from the PDF.

    Raises:
    FileNotFoundError: If the PDF file is not found.
    PyPDF2.utils.PdfReadError: If there's an error reading the PDF.
    """
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    except FileNotFoundError:
        print(f"Error: PDF file not found at {pdf_path}")
        sys.exit(1)
    except PyPDF2.utils.PdfReadError as e:
        print(f"Error reading PDF: {e}")
        sys.exit(1)

def preprocess_text(text: str) -> str:
    """
    Preprocess the extracted text.

    Args:
    text (str): Raw text extracted from PDF.

    Returns:
    str: Preprocessed text.
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove page numbers
    text = re.sub(r'\n\d+\n', '\n', text)
    
    # Handle mathematical equations (placeholder)
    text = re.sub(r'\$.*?\$', '[EQUATION]', text)
    
    return text

def split_into_chunks(text: str, max_chunk_size: int = 1000) -> List[str]:
    """
    Split the preprocessed text into chunks.

    Args:
    text (str): Preprocessed text.
    max_chunk_size (int): Maximum size of each chunk in characters.

    Returns:
    List[str]: List of text chunks.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def main(input_pdf: str, output_txt: str):
    """
    Main function to process the PDF and save chunks to a text file.

    Args:
    input_pdf (str): Path to the input PDF file.
    output_txt (str): Path to the output text file.
    """
    try:
        # Extract text from PDF
        raw_text = extract_text_from_pdf(input_pdf)

        # Preprocess the text
        processed_text = preprocess_text(raw_text)

        # Split into chunks
        chunks = split_into_chunks(processed_text)

        # Save chunks to file
        with open(output_txt, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks):
                f.write(f"CHUNK {i+1}:\n{chunk}\n\n")

        print(f"Successfully processed {input_pdf} and saved chunks to {output_txt}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python document_preparation.py input.pdf output.txt")
        sys.exit(1)

    input_pdf = sys.argv[1]
    output_txt = sys.argv[2]
    main(input_pdf, output_txt)