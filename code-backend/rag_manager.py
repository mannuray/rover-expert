import os
import json
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, CSVLoader, Docx2txtLoader, UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import *

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

# ✅ Loader mapping
LOADER_CLASSES = {
    ".txt": TextLoader,
    ".pdf": PyPDFLoader,
    ".csv": CSVLoader,
    ".docx": Docx2txtLoader,
    ".md": UnstructuredMarkdownLoader,
}

# ✅ Timestamps handling
TIMESTAMP_FILE = "timestamps.json"

def save_timestamps(timestamps):
    """Save timestamps to JSON file"""
    with open(TIMESTAMP_FILE, "w") as f:
        json.dump(timestamps, f)

def load_timestamps():
    """Load timestamps from JSON file"""
    if os.path.exists(TIMESTAMP_FILE):
        with open(TIMESTAMP_FILE, "r") as f:
            return json.load(f)
    return {}

# ✅ Add or update files in RAG
def add_files_to_rag(file_paths):
    """Add new or modified files to RAG"""
    all_docs = []

    for file_path in file_paths:
        ext = os.path.splitext(file_path)[1].lower()

        if ext in LOADER_CLASSES:
            try:
                loader = LOADER_CLASSES[ext](file_path)
                docs = loader.load()

                # ✅ Split documents into smaller chunks
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = splitter.split_documents(docs)

                all_docs.extend(chunks)
                print(f"✅ Added {len(chunks)} chunks from {file_path}")
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")

    if all_docs:
        vectorstore.add_documents(all_docs)
        print("✅ Documents added successfully!")

# ✅ Detect file updates using Git
def get_updated_files():
    """Get modified, added, and deleted files using Git"""
    modified = os.popen("git diff --name-status HEAD~1").read().splitlines()
    new_files, modified_files, deleted_files = [], [], []

    for line in modified:
        status, file_path = line.split("\t")
        if status == "M":
            modified_files.append(file_path)
        elif status == "A":
            new_files.append(file_path)
        elif status == "D":
            deleted_files.append(file_path)

    return new_files, modified_files, deleted_files

# ✅ Update RAG with file changes
def update_rag():
    """Update RAG with changed files"""
    new_files, modified_files, deleted_files = get_updated_files()

    # Add new and modified files
    files_to_add = new_files + modified_files
    add_files_to_rag(files_to_add)

    # Remove deleted files
    for file in deleted_files:
        print(f"Removing deleted file: {file}")
        vectorstore.delete(file)
