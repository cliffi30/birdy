import os
import pathlib
from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class LangChainTextSplitter():
    def __init__(self):

      # set base Path
      self.base_Dir  = pathlib.Path(__file__).parent.resolve()

      # split documents into text and embeddings
      self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=150,
            add_start_index=True)
      pass
    
    def GetSimpleChunksFromPdf(self, fileName):
      #build FilePath
      dataFilePath = os.path.join(self.base_Dir, fileName)

      #load PdfFile
      loader = PyPDFLoader(dataFilePath)
      documents = loader.load()

      split_documents = self.text_splitter.split_documents(documents=documents[20:])

      return split_documents
    
    def GetRecursiveCharacterChunksFromPdf(self, fileName):
      #build FilePath
      dataFilePath = os.path.join(self.base_Dir, fileName)
      
      # This is a long document we can split up.
      #load PdfFile
      pdf_loader = PyPDFLoader(dataFilePath)
      documents = pdf_loader.load()

      # Create documents using LangChain's create_documents method
      # Assuming the documents are already in the correct format
      # If you need to create Document objects, you can do it like this:
      langchain_documents = [Document(page_content=doc.page_content, metadata={"source": doc.metadata}) for doc in documents]

      # Initialize the text splitter
      text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap=50)

      # Split the loaded documents into chunks
      split_documents = text_splitter.split_documents(langchain_documents)
      return split_documents
    
    def GetRecursiveCharacterChunksFromTextFile(self, bird_name):
        #build FilePath
        #dataFilePath = os.path.join(self.base_Dir, fileName)
        dataFilePath = os.path.join("data/raw/BirdFiles/", bird_name + "V1.txt")
        
        # This is a long document we can split up.
        #load PdfFile
        text_loader = TextLoader(dataFilePath)
        documents = text_loader.load()

        # Create documents using LangChain's create_documents method
        # Assuming the documents are already in the correct format
        # If you need to create Document objects, you can do it like this:
        langchain_documents = [Document(page_content=doc.page_content, metadata={"source": doc.metadata, "Bird" : bird_name}) for doc in documents]

        # Initialize the text splitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap=50)

        # Split the loaded documents into chunks
        split_documents = text_splitter.split_documents(langchain_documents)
        return split_documents