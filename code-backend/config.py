import os

# 🔥 Environment variables
EMBEDDINGS_MODEL_NAME = os.getenv("EMBEDDINGS_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "code_rag")
CODE_FOLDER = os.getenv("CODE_FOLDER", "./codebase")

# ✅ Loader Mapping for different file types
LOADER_MAPPING = {
    ".txt": "TextLoader",
    ".pdf": "PyPDFLoader",
    ".csv": "CSVLoader",
    ".docx": "Docx2txtLoader",
    ".md": "UnstructuredMarkdownLoader",
}

