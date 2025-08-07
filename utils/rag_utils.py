from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
import tempfile

# Load embedding model once
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def load_and_split_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)
    return chunks

def embed_chunks(chunks):
    return FAISS.from_documents(chunks, embedding_model)

def retrieve_relevant_chunks(query, faiss_index, k=4):
    return faiss_index.similarity_search(query, k=k)
