import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Define local persistent directory for ChromaDB
DB_DIR = "./chroma_db"
COLLECTION_NAME = "document_index"

def load_documents_from_folder(folder_path: str) -> List[Document]:
    """Loads TXT, PDF, and Markdown files from a given directory."""
    documents = []
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created sample directory at '{folder_path}'. Please add your files there.")
        return documents

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if file.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
            documents.extend(loader.load())
        elif file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        elif file.endswith(".md"):
            # Fallback to TextLoader if unstructured is not installed
            loader = TextLoader(file_path, encoding="utf-8")
            documents.extend(loader.load())

    print(f"Loaded {len(documents)} raw document(s) from '{folder_path}'.")
    return documents

def build_vector_db(documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> Chroma:
    """Splits documents into chunks, generates embeddings, and indexes them into ChromaDB."""
    # 1. Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks (Chunk Size: {chunk_size}, Overlap: {chunk_overlap}).")

    # 2. Embedding Model (Local HuggingFace model, no API key required)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 3. Store in ChromaDB (Persistent storage)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=DB_DIR
    )
    print(f"Successfully indexed chunks into ChromaDB at '{DB_DIR}'.")
    return vector_store

def verify_and_query(vector_store: Chroma, query: str = "AI Engineering"):
    """Verifies that vectors are stored by checking count and performing a similarity test."""
    # Check stored document count
    collection = vector_store._collection
    count = collection.count()
    print(f"\n--- Storage Verification ---")
    print(f"Total Vectors Stored in Collection '{COLLECTION_NAME}': {count}")

    # Test Similarity Retrieval
    results = vector_store.similarity_search(query, k=2)
    print(f"\n--- Similarity Search Verification for Query: '{query}' ---")
    for idx, doc in enumerate(results, start=1):
        print(f"Result {idx}:")
        print(f"  Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"  Content snippet: {doc.page_content[:150]}...\n")

def bonus_chunk_size_comparison(documents: List[Document]):
    """Bonus: Compare chunk counts across different chunk sizes."""
    print("--- Bonus: Comparing Chunk Sizes ---")
    chunk_sizes = [200, 500, 1000]
    
    for size in chunk_sizes:
        splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=20)
        chunks = splitter.split_documents(documents)
        print(f"Chunk Size: {size:4d} | Total Chunks Created: {len(chunks)}")

if __name__ == "__main__":
    docs_folder = "./sample_docs"
    
    # 1. Load documents
    raw_docs = load_documents_from_folder(docs_folder)
    
    if raw_docs:
        # 2 & 3 & 4. Split, Embed, and Store
        vector_db = build_vector_db(raw_docs, chunk_size=500, chunk_overlap=50)
        
        # 5. Verification
        verify_and_query(vector_db, query="retrieval embeddings")
        
        # Bonus Analysis
        bonus_chunk_size_comparison(raw_docs)
