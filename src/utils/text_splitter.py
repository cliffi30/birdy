import os
import pathlib

from embeddings.embeddings import Embeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader


class LangChainTextSplitter:
    def __init__(self, embedder: Embeddings = None, ):
        # set base Path
        self.base_dir = pathlib.Path(__file__).parent.resolve()
        self.embedder = embedder

        # split documents into text and embeddings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            add_start_index=True)

    def get_simple_chunks_from_pdf(self, file_name):
        # build FilePath
        data_file_path = os.path.join(self.base_dir, file_name)

        # load PdfFile
        loader = PyPDFLoader(data_file_path)
        documents = loader.load()

        split_documents = self.text_splitter.split_documents(documents=documents[20:])

        return split_documents

    def get_recursive_character_chunks_from_pdf(self, file_name):
        # build FilePath
        data_file_path = os.path.join(self.base_dir, file_name)

        # This is a long document we can split up.
        # load PdfFile
        pdf_loader = PyPDFLoader(data_file_path)
        documents = pdf_loader.load()

        # Create documents using LangChain's create_documents method
        # Assuming the documents are already in the correct format
        # If you need to create Document objects, you can do it like this:
        langchain_documents = [Document(page_content=doc.page_content, metadata={"source": doc.metadata}) for doc in
                               documents]

        # Initialize the text splitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

        # Split the loaded documents into chunks
        split_documents = text_splitter.split_documents(langchain_documents)
        return split_documents

    def get_recursive_character_chunks_from_text_file(self, bird_name):
        # build FilePath
        data_file_path = os.path.join("data/raw/BirdFiles/", bird_name + "V1.txt")

        # This is a long document we can split up.
        # load PdfFile
        text_loader = TextLoader(data_file_path)
        documents = text_loader.load()

        # Create documents using LangChain's create_documents method
        # Assuming the documents are already in the correct format
        # If you need to create Document objects, you can do it like this:
        langchain_documents = [
            Document(page_content=doc.page_content, metadata={"source": data_file_path, "bird": bird_name}) for doc in
            documents]

        # Initialize the text splitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

        # Split the loaded documents into chunks
        split_documents = text_splitter.split_documents(langchain_documents)
        return split_documents
