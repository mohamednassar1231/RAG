# RAG Assistant

A simple local Retrieval-Augmented Generation (RAG) system that answers questions based on the content of a PDF document.

Built as a learning project to understand RAG pipelines, prompt engineering, and working with local LLMs.

## Features

- Loads and processes a PDF document
- Splits text into chunks and stores embeddings using FAISS
- Retrieves relevant context for user questions
- Uses conversation history for better context
- Friendly and natural responses while staying grounded to the document
- Verbose logging for easy debugging
- Saves FAISS index locally to avoid rebuilding every time

## Tech Stack

- **LangChain** – Document loading, splitting, and orchestration
- **FAISS** – Vector store with local persistence
- **Hugging Face Embeddings** – `intfloat/multilingual-e5-large-instruct` (runs on CPU)
- **Ollama** – Local LLM (`llama3:8b`)
- **Python** – Core implementation

## Project Structure
├── rag_assistant.py          # Main RAG class
├── requirements.txt          # Dependencies
├── faiss_index/              # Auto-generated FAISS vector store (ignored in .gitignore)
├── your_document.pdf         # Your PDF file (add your own)
└── README.md
## How It Works (RAG Pipeline)

1. Load the PDF document
2. Split it into smaller overlapping chunks
3. Create embeddings and store them in a FAISS vector database
4. When a question is asked:
   - Retrieve the most relevant chunks
   - Build a prompt with context + conversation history
   - Send to Llama 3 (via Ollama)
   - Return a natural, document-grounded answer
