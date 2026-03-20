# 1a is all about splitting text, dividing into chunks, convert into embedding and storing to vector store
# Referring to image high-level-plan-using-RAG.png

import os


from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings


# Define the directory containing the text file and persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "resident_evil2.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")


# Check if the Chroma vector store already exsists
# SO if a embedding already exsists we dont need to rerun all of it, 
# that is what this script is all about covers the part from chunking to embedding to vectorstore
if not os.path.exists(persistent_directory):        
    print("Persistent directory does not exsists. Initializing vector store...")

    # Ensure the text file exsists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Text file not found  at path {file_path}")

    # Read the text content from the file
    loader = TextLoader(file_path)
    documents = loader.load()


    # Split the document into chunks, using CharacterTextSplitter
    text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)
    docs = text_splitter.split_documents(documents)


    # Display information about the chunks
    print("\n Document Chunks Information")
    print(f"Number of document chunks {len(docs)}")
    print(f"Sample chunk: \n {docs[0].page_content}\n")

    # Create embeddings
    print("\n ---  Creatnig embeddings ---")
#   embeddings = OpenAIEmbeddings(model="text-embedding-3-small") # Update accordingly
    embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")
    print("\n --- Finished creating embeddings ---")

    # Create vector store and persist automatically
    print("\n ---  Creating vector store ---")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory) 
        
else:
    print("Persistent directory already exsists. Skipping vector store initialization.")