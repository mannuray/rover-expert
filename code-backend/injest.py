#!/usr/bin/env python3

import os
import argparse
import requests

# ✅ Constants
API_URL = "http://localhost:5004/add-code-folder-to-rag"

def add_folder_to_rag(folder):
    """ Add folder to RAG using the API """
    if not os.path.exists(folder):
        print(f"❌ Folder '{folder}' does not exist.")
        return

    print(f"📂 Adding folder '{folder}' to RAG...")

    try:
        response = requests.post(API_URL, json={"folder_path": folder})

        if response.status_code == 200:
            print("✅ Folder added successfully!")
            print(response.json())
        else:
            print(f"❌ Failed to add folder: {response.text}")
    
    except Exception as e:
        print(f"🔥 Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add a folder to RAG")
    parser.add_argument("folder", help="Path to the folder to add")
    args = parser.parse_args()

    add_folder_to_rag(args.folder)
