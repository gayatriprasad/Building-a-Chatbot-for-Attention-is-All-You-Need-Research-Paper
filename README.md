# RAG-powered Chatbot for "Attention is All You Need" Paper

This project implements a Retrieval-Augmented Generation (RAG) system to answer questions about the seminal paper "Attention is All You Need" by Vaswani et al. The chatbot uses a document-based system where the paper is converted into embeddings, which are then queried using Langchain's RAG system to generate responses.

## Features

- Extract and preprocess text from the PDF of the paper
- Generate embeddings using FAISS and HuggingFace models
- Implement a RAG system using Langchain
- Answer user questions about the paper using retrieved context and a language model

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/rag-attention-paper-chatbot.git
   cd rag-attention-paper-chatbot
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Prepare the document:
   ```
   python document_preparation.py attention_paper.pdf chunks.txt
   ```

2. Generate embeddings:
   ```
   python embedding_generation.py chunks.txt faiss_index.index
   ```

3. Run the RAG system:
   ```
   python rag_system.py faiss_index.index chunks.txt
   ```

4. Start asking questions about the "Attention is All You Need" paper!

## Project Structure

- `document_preparation.py`: Extracts and preprocesses text from the PDF
- `embedding_generation.py`: Generates embeddings and creates a FAISS index
- `rag_system.py`: Implements the RAG system for answering questions
- `requirements.txt`: Lists all Python dependencies
- `README.md`: Provides project information and usage instructions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- "Attention is All You Need" paper by Vaswani et al.
- Langchain for providing the RAG implementation
- Hugging Face for pre-trained models and transformers library
- FAISS for efficient similarity search
