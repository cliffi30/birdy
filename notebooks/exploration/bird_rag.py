

import os
from typing import List

import chromadb
import pypdf
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter


def extract_pdf_text(pdf_path: str) -> str:
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found at {pdf_path}")
    text = ""
    with open(pdf_path, 'rb') as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text()
    return text

def split_into_chunks(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    return splitter.split_text(text)

def build_chroma_index(chunks: List[str], collection_name="my_collection"):
    # Initialize the Chroma client
    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet",
                                      persist_directory="chromadb_store"))
    # Example embedding function (SentenceTransformer)
    embed_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    # Create or get a collection
    collection = client.get_or_create_collection(name=collection_name,
                                                 embedding_function=embed_func)
    # Add chunks to the collection
    for i, chunk in enumerate(chunks):
        collection.add(documents=[chunk], ids=[str(i)])
    return collection

def retrieve_relevant_chunks(collection, query: str, k=3) -> List[str]:
    results = collection.query(query_texts=[query], n_results=k)
    return results["documents"][0]

# Example usage
def main():
    pdf_path = "data/mypdf.pdf"
    text = extract_pdf_text(pdf_path)
    chunks = split_into_chunks(text)
    collection = build_chroma_index(chunks)

    # Query
    question = "What does this PDF say about climate impact?"
    retrieved = retrieve_relevant_chunks(collection, question)
    context = "\n".join(retrieved)

    # Pass context + question to Llama
    # (Example use with a hypothetical OllamaCompletions class)
    from models.ollama_completions import OllamaCompletions
    ollama = OllamaCompletions()
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    answer = ollama.get_completion(prompt)
    print(answer)

if __name__ == "__main__":
    main()