from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstore/db_faiss'

# Create vector database
def create_vector_db():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data path '{DATA_PATH}' does not exist.")
    
    print("Loading PDF documents...")
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")
    
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks.")
    
    print("Generating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'}
    )
    
    print("Creating FAISS vector database...")
    db = FAISS.from_documents(texts, embeddings)
    
    print(f"Saving FAISS database to '{DB_FAISS_PATH}'...")
    os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
    db.save_local(DB_FAISS_PATH)
    print("Vector database creation completed successfully!")

if __name__ == "__main__":
    create_vector_db()
