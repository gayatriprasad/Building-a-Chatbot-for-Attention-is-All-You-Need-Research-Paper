"""
embedding_generation.py

This script generates embeddings for the preprocessed text chunks
using a pre-trained model from Hugging Face and stores them using FAISS.

Dependencies:
- transformers
- torch
- faiss-cpu (or faiss-gpu for GPU support)

Usage:
python embedding_generation.py input.txt output.index
"""

import sys
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
from typing import List, Tuple

# Check for GPU availability
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_chunks(input_file: str) -> List[str]:
    """
    Load text chunks from the input file.

    Args:
    input_file (str): Path to the input text file.

    Returns:
    List[str]: List of text chunks.

    Raises:
    FileNotFoundError: If the input file is not found.
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        chunks = content.split('CHUNK')[1:]  # Skip the first empty split
        return [chunk.strip() for chunk in chunks]
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        sys.exit(1)

def get_embeddings(chunks: List[str], model_name: str = 'sentence-transformers/all-MiniLM-L6-v2') -> np.ndarray:
    """
    Generate embeddings for the given text chunks.

    Args:
    chunks (List[str]): List of text chunks.
    model_name (str): Name of the pre-trained model to use.

    Returns:
    np.ndarray: Array of embeddings.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE)

    embeddings = []

    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors='pt', truncation=True, max_length=512, padding=True).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        embeddings.append(embedding)

    return np.array(embeddings)

def create_faiss_index(embeddings: np.ndarray, output_file: str):
    """
    Create and save a FAISS index for the embeddings.

    Args:
    embeddings (np.ndarray): Array of embeddings.
    output_file (str): Path to save the FAISS index.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, output_file)
    print(f"FAISS index saved to {output_file}")

def main(input_file: str, output_file: str):
    """
    Main function to generate embeddings and create FAISS index.

    Args:
    input_file (str): Path to the input text file.
    output_file (str): Path to save the FAISS index.
    """
    try:
        # Load chunks
        chunks = load_chunks(input_file)
        print(f"Loaded {len(chunks)} chunks from {input_file}")

        # Generate embeddings
        embeddings = get_embeddings(chunks)
        print(f"Generated embeddings with shape {embeddings.shape}")

        # Create and save FAISS index
        create_faiss_index(embeddings, output_file)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python embedding_generation.py input.txt output.index")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    main(input_file, output_file)