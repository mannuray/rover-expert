import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import chromadb

# ✅ Initialize Flask app
app = Flask(__name__)
CORS(app)

# 🔥 Environment Variables
MODEL = os.getenv("MODEL", "mistral")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "code_rag")
EMBEDDINGS_MODEL_NAME = os.getenv("EMBEDDINGS_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# ✅ Initialize Chroma Client
chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

# ✅ Initialize Embeddings Model
embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)

# ✅ Initialize Chroma VectorStore
vectorstore = Chroma(
    client=chroma_client,
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_function
)

# ✅ Create Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ✅ Initialize LLM (e.g., OpenAI GPT-4)
llm = OllamaLLM(model=MODEL)

# ✅ Chain: Retriever + LLM for answering code queries
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

@app.route('/query-rag', methods=['POST'])
def query_rag():
    """ Query the RAG system and get insights """
    try:
        data = request.get_json()
        query = data.get("query", "")

        if not query:
            return jsonify({"error": "Query cannot be empty"}), 400

        # 🔥 Use LLM to generate insights from retrieved documents
        response = qa_chain.run(query)

        return jsonify({
            "query": query,
            "insight": response
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=True)
