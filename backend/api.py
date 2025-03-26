#!/usr/bin/env python3
import os
import time
import chromadb
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM

# ✅ Environment Variables
MODEL = os.environ.get("MODEL", "mistral")
EMBEDDINGS_MODEL_NAME = os.environ.get("EMBEDDINGS_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
TARGET_SOURCE_CHUNKS = int(os.environ.get("TARGET_SOURCE_CHUNKS", 4))

# ✅ Chroma Server Connection Info
CHROMA_HOST = os.environ.get("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", 8000))
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "easyeat_collection")

# ✅ Initialize Flask App
app = Flask(__name__)
CORS(app)

# ✅ Initialize ChromaDB Client
chroma_client = chromadb.HttpClient(
    host=CHROMA_HOST,
    port=CHROMA_PORT
)

# ✅ Initialize Embeddings
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)

# ✅ Connect to the Running Chroma Server
print(f"🔌 Connecting to Chroma server at {CHROMA_HOST}:{CHROMA_PORT}...")
vectorstore = Chroma(
    client=chroma_client,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings
)

# ✅ Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": TARGET_SOURCE_CHUNKS})

# ✅ Initialize LLM
llm = OllamaLLM(model=MODEL)

# ✅ Create RAG QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)
print("🚀 RAG system connected to Chroma server!")

# ✅ Utility function for semantic search
def perform_semantic_search(query, k=TARGET_SOURCE_CHUNKS):
    """
    Perform semantic search without generating an answer
    """
    try:
        # Directly use the retriever for semantic search
        docs = retriever.get_relevant_documents(query)
        
        sources = [
            {
                "source": doc.metadata.get("source", "Unknown"),
                "content": doc.page_content
            } for doc in docs
        ]
        
        return sources
    except Exception as e:
        print(f"Error in semantic search: {e}")
        return []

### ✅ **POST `/query`: Query the RAG system**
@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.get_json()
        query = data.get("query", "")
        semantic_only = data.get("semantic_only", False)
        
        if not query:
            return jsonify({"error": "Query cannot be empty"}), 400
        
        # Query Chroma store
        print("\n🔍 Querying Chroma store...")
        start = time.time()
        
        if semantic_only:
            # Only perform semantic search
            sources = perform_semantic_search(query)
            response = {
                "question": query,
                "sources": sources,
                "time_taken": f"{time.time() - start:.2f} seconds"
            }
        else:
            # Full RAG query
            res = qa_chain(query)
            answer = res['result']
            docs = res.get('source_documents', [])
            
            sources = [
                {
                    "source": doc.metadata.get("source", "Unknown"),
                    "content": doc.page_content
                } for doc in docs
            ]
            
            response = {
                "question": query,
                "answer": answer,
                "sources": sources,
                "time_taken": f"{time.time() - start:.2f} seconds"
            }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

### ✅ **GET `/`: Health check**
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "status": "RAG API connected to Chroma!",
        "host": CHROMA_HOST,
        "port": CHROMA_PORT,
        "collection": COLLECTION_NAME
    }), 200

### ✅ **GET `/collections`: List available collections**
@app.route('/collections', methods=['GET'])
def list_collections():
    try:
        collections = [collection.name for collection in chroma_client.list_collections()]
        return jsonify({"collections": collections}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)