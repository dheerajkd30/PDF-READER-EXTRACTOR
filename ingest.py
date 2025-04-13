import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

def ingest_pdf(file_path):
    # Load PDF
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # Split into smaller chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Embed and store in Chroma
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(chunks, embedding, persist_directory="db")
    vectordb.persist()
    print(f"✅ Ingested {file_path} into Chroma vector store.")

if __name__ == "__main__":
    ingest_pdf("sample.pdf")