import os


#from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings


# Define the directory containing the text file and persistent directory
curremt_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(curremt_dir, "books")
db_dir  = os.path.join(curremt_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

print(f"\nBooks directory: {books_dir}")
print(f"Persistent directory: {persistent_directory}\n")

# Check if the Chroma vector store already exsists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exsits. Initializing vector store...")

    # Ensure the books directory exsists
    if not os.path.exists(books_dir):
        raise FileNotFoundError(f"The directory {books_dir} does not exsists")
    

    # List all text files in the directory
    books_files = [f for f in os.listdir(books_dir) if f.endswith(".txt")]

    # Stores txt files as a list of txt files
    print(f"book files are {books_files}")

    # Read the text content from each file and store it with metadata
    documents = []
    for book_file in books_files:
        file_path = os.path.join(books_dir,book_file)
        loader = TextLoader(file_path)
        book_docs = loader.load()
        for doc in book_docs:
            # Add metadata to each document indicating its source
            doc.metadata = {"source": book_file}
            documents.append(doc)


    # Split the documents into chunks

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1500, chunk_overlap = 0)
    docs =  text_splitter.split_documents(documents)

    # Display information about the chunks
    print("\n Document Chunks Information")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk: \n {docs[0].page_content}\n")


    # Create embeddings
    print("\n ---  Creatnig embeddings ---")
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small") # Update accordingly
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    print("\n --- Finished creating embeddings ---")

    print("\n ---  Creating and persisting vector store ---")
    # Create a vector store and persist automatically
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
    print("\n --- Finished creating and persisting vector store ---")


else:
    print("Persistent directory already exsists. Skipping vector store initialization.")