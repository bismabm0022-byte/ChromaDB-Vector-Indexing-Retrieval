# Document Indexing with ChromaDB

This repository implements document loading, text chunking, embedding generation, and persistent vector storage using **ChromaDB** and **LangChain**.

## Features
- Multi-format document loading (`.txt`, `.pdf`, `.md`)
- Recursive character text chunking
- Open-source vector embeddings (`all-MiniLM-L6-v2`)
- Persistent storage and retrieval verification via ChromaDB
- Chunk size impact comparison

##  Project Structure

```text
.
├── sample_docs/            # Input document directory (.txt, .pdf, .md)
├── chroma_db/              # Persistent vector database output (ignored by git)
├── index_documents.py      # Main pipeline script
├── requirements.txt        # Python dependency specifications
├── .gitignore              # Keeps heavy local DB files out of version control
└── README.md               # Project documentation

## How to Run

1. **Clone the repo and set up venv:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
