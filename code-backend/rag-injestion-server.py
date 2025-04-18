import os
import chromadb
from flask import Flask, request, jsonify
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from tree_sitter import Language, Parser
import tree_sitter_typescript as tstypescript
import tree_sitter_python as tspython

# 🔥 Environment Variables
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "code_rag")
EMBEDDINGS_MODEL_NAME = os.getenv("EMBEDDINGS_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# ✅ Initialize Flask app
app = Flask(__name__)

chroma_client = chromadb.HttpClient(
    host=CHROMA_HOST,
    port=CHROMA_PORT
)

# ✅ Initialize Chroma Vectorstore
vectorstore = Chroma(
    client=chroma_client,
    collection_name=COLLECTION_NAME,
    embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
)

# ✅ Initialize Tree-Sitter for Code Parsing
LANGUAGE_LIB = "/Users/metallonudo/expert_test/new/build/my-languages.so"  # Path to compiled languages
parser = Parser()

# ✅ Supported Languages and Extensions
LANGUAGE_MAPPING = {
    ".ts": "typescript",
    ".js": "javascript",
    ".py": "python",
    ".go": "go",
    ".java": "java"
}

# ✅ Load language grammar
def load_language(language_name):
    language = Language(tstypescript.language())
    parser.set_language(language)

# ✅ Function to parse and extract code chunks
def parse_code(file_path, file_ext):
    """ Parse code into structured chunks using Tree-Sitter """
    
    language_name = LANGUAGE_MAPPING.get(file_ext)
    if not language_name:
        print(f"❌ Unsupported language: {file_ext}")
        return []

    try:
        # ✅ Initialize parser correctly
        language = Language(tstypescript.language_typescript())
        parser = Parser(language)

        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()

        tree = parser.parse(bytes(code, "utf8"))
        root_node = tree.root_node

        chunks = []

        def extract_chunks(node):
            """ Recursively extract code chunks """
            if node.type in ["function_declaration", "method_declaration", "class_declaration"]:
                chunk = code[node.start_byte:node.end_byte]
                metadata = {
                    "file": file_path,
                    "language": language_name,
                    "type": node.type,
                    "start_line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1
                }
                chunks.append({"content": chunk, "metadata": metadata})

            for child in node.children:
                extract_chunks(child)

        extract_chunks(root_node)
        return chunks

    except Exception as e:
        print(f"❌ Error parsing {file_path}: {e}")
        return []

# ✅ Add Code Folder to RAG
@app.route('/add-code-folder-to-rag', methods=['POST'])
def add_code_folder_to_rag():
    """ Add code files from a local folder to RAG """
    try:
        data = request.get_json()
        folder_path = data.get("folder_path", "")

        if not folder_path or not os.path.exists(folder_path):
            return jsonify({"error": "Invalid folder path"}), 400

        all_code_chunks = []

        # ✅ Iterate over files in the folder
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                ext = os.path.splitext(file)[1].lower()

                # ✅ Only process supported languages
                if ext in LANGUAGE_MAPPING:
                    print(f"📄 Parsing {file_path}...")
                    try:
                        # ✅ Use the code parser function
                        chunks = parse_code(file_path, ext)
                        if chunks:
                            all_code_chunks.extend(chunks)
                        else:
                            print(f"⚠️ No code chunks extracted from {file}")
                    except Exception as e:
                        print(f"❌ Error parsing {file}: {e}")

        # ✅ Add parsed code chunks to RAG
        if all_code_chunks:
            print(f"🚀 Adding {len(all_code_chunks)} code chunks to RAG...")
            
            # ✅ Prepare the documents for RAG ingestion
            documents = [
                Document(
                    page_content=chunk["content"],
                    metadata=chunk["metadata"]
                )
                for chunk in all_code_chunks
            ]

            # ✅ Use from_documents() instead of add_documents()
            vectorstore.add_documents(documents)
            
            print("here 2")
            return jsonify({
                "message": f"Added {len(all_code_chunks)} code chunks to RAG!"
            }), 200
        else:
            return jsonify({"message": "No valid code chunks found!"}), 200

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500


# ✅ Health Check
@app.route('/', methods=['GET'])
def health_check():
    """ Health check """
    return jsonify({
        "status": "Code RAG API connected to Chroma!",
        "host": CHROMA_HOST,
        "port": CHROMA_PORT,
        "collection": COLLECTION_NAME
    }), 200

# ✅ Run the service
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004, debug=True)
