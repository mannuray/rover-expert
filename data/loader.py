#!/usr/bin/env python3
import os
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader, Docx2txtLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 🔥 Environment variables
EMBEDDINGS_MODEL_NAME = os.environ.get("EMBEDDINGS_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_HOST = os.environ.get("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", 8000))
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "easyeat_collection")
DOCS_FOLDER = os.environ.get("DOCS_FOLDER", "./docs")

# ✅ Initialize ChromaDB client
chroma_client = chromadb.HttpClient(
    host=CHROMA_HOST,
    port=CHROMA_PORT
)

# ✅ Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)

# ✅ Connect to Chroma vector store
vectorstore = Chroma(
    client=chroma_client,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings
)

# ✅ Supported file types and loaders
LOADER_MAPPING = {
    ".txt": TextLoader,
    ".pdf": PyPDFLoader,
    ".csv": CSVLoader,
    ".docx": Docx2txtLoader,
    ".md": UnstructuredMarkdownLoader,
}

# ✅ Function to load and index documents
def load_and_index_documents(folder):
    all_docs = []
    for root, _, files in os.walk(folder):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            if ext in LOADER_MAPPING:
                try:
                    print(f"Loading {file}...")
                    loader = LOADER_MAPPING[ext](file_path)
                    documents = loader.load()
                    
                    # ✅ Split documents into smaller chunks
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                    docs = text_splitter.split_documents(documents)
                    
                    all_docs.extend(docs)
                except Exception as e:
                    print(f"Error loading {file}: {e}")
    
    # ✅ Add documents to Chroma
    if all_docs:
        print(f"Adding {len(all_docs)} documents to Chroma...")
        try:
            vectorstore.add_documents(all_docs)
            print("✅ Documents added successfully!")
        except Exception as e:
            print(f"Error adding documents to Chroma: {e}")
    else:
        print("⚠️ No valid documents found!")

# ✅ Run the script
if __name__ == "__main__":
    load_and_index_documents(DOCS_FOLDER)