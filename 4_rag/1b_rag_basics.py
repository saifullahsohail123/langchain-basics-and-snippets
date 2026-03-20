import os

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings


# Define the persisten directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_dir = os.path.join(current_dir,"db","chroma_db")

# Define the embedding model
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")


# Load the exsisting vector store with the embedding function
db = Chroma(persist_directory=persistent_dir, embedding_function=embeddings)
# embeddings is used to convert user query into vector and then compare it with the 
# vectors in the vector store to find relevant documents


# Define the user's question
query = "what are the main characters in Resident Evil 2?"

# Retrieve relevant documents based on the query
retriever = db.as_retriever(
    search_type = "similarity_score_threshold", # This search type retrieves documents based on a similarity score threshold, ensuring that only documents with a certain level of relevance to the query are returned.
    search_kwargs = {"k": 3, "score_threshold": 0.3} # k is the number of top documents to retrieve, and score_threshold is the minimum similarity score required for a document to be considered relevant.
)

relevant_docs = retriever.invoke(query)


# Display the relevant results with the metadata
print("\n--- Relevant Documents ---")
for i,doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")


