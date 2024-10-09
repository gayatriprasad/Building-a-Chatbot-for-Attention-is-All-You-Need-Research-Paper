"""
rag_system.py

This script implements a Retrieval-Augmented Generation (RAG) system
using Langchain, FAISS, and a pre-trained language model.

Dependencies:
- langchain
- faiss-cpu (or faiss-gpu for GPU support)
- transformers
- torch

Usage:
python rag_system.py faiss_index.index chunks.txt
"""

import sys
import traceback
from typing import List, Dict
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS as LangchainFAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

# Check for GPU availability and set device
if torch.cuda.is_available():
    DEVICE = 0  # Use the first GPU
else:
    DEVICE = -1  # Use CPU

def load_chunks(chunks_file: str) -> List[str]:
    """
    Load text chunks from the input file.

    Args:
    chunks_file (str): Path to the file containing text chunks.

    Returns:
    List[str]: List of text chunks.

    Raises:
    FileNotFoundError: If the chunks file is not found.
    """
    try:
        with open(chunks_file, 'r', encoding='utf-8') as f:
            content = f.read()
        chunks = content.split('CHUNK')[1:]  # Skip the first empty split
        return [chunk.strip() for chunk in chunks]
    except FileNotFoundError:
        print(f"Error: Chunks file not found at {chunks_file}")
        sys.exit(1)

def load_faiss_index(index_file: str) -> faiss.Index:
    """
    Load the FAISS index from a file.

    Args:
    index_file (str): Path to the FAISS index file.

    Returns:
    faiss.Index: Loaded FAISS index.

    Raises:
    FileNotFoundError: If the index file is not found.
    """
    try:
        index = faiss.read_index(index_file)
        return index
    except FileNotFoundError:
        print(f"Error: FAISS index file not found at {index_file}")
        sys.exit(1)


def setup_rag_system(chunks: List[str], model_name: str = 'sentence-transformers/all-MiniLM-L6-v2') -> RetrievalQA:
    """
    Set up the RAG system using Langchain components.

    Args:
    chunks (List[str]): List of text chunks.
    model_name (str): Name of the pre-trained model to use for embeddings.

    Returns:
    RetrievalQA: Langchain RetrievalQA chain.
    """
    # Set up embedding model
    embed_model = HuggingFaceEmbeddings(model_name=model_name)

    # Print debugging information
    print(f"Number of chunks: {len(chunks)}")

    # Create FAISS vectorstore directly from texts
    texts = [chunk for chunk in chunks]
    try:
        vectorstore = LangchainFAISS.from_texts(texts, embed_model)
        print(f"FAISS vectorstore created successfully with {len(texts)} documents")
    except Exception as e:
        print(f"Error creating FAISS vectorstore: {str(e)}")
        raise

    # Set up language model pipeline for generation
    model_id = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to('cuda' if DEVICE >= 0 else 'cpu')
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=1000,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        device=DEVICE
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    # Create RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
    )

    return qa_chain

def main(chunks_file: str):
    """
    Main function to set up and run the RAG system.

    Args:
    chunks_file (str): Path to the file containing text chunks.
    """
    try:
        # Load chunks
        chunks = load_chunks(chunks_file)
        
        # Set up RAG system
        qa_chain = setup_rag_system(chunks)

        print("RAG system is ready. You can now ask questions about the 'Attention is All You Need' paper.")
        print("Type 'exit' to quit.")

        while True:
            question = input("\nEnter your question: ")
            if question.lower() == 'exit':
                break

            try:
                # Generate response
                response = qa_chain({"query": question})
                print("\nAnswer:", response['result'])
            except Exception as e:
                print(f"Error generating response: {str(e)}")
                print("Traceback:")
                traceback.print_exc()

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python rag_system.py chunks.txt")
        sys.exit(1)

    chunks_file = sys.argv[1]
    main(chunks_file)