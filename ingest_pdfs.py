import os
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from dotenv import load_dotenv
import pinecone

load_dotenv()

DATA_DIR = "./data"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))
index_name = os.getenv("PINECONE_INDEX")

all_text = ""
for file_name in os.listdir(DATA_DIR):
    if file_name.endswith(".pdf"):
        with fitz.open(os.path.join(DATA_DIR, file_name)) as doc:
            for page in doc:
                all_text += page.get_text()

splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks = splitter.split_text(all_text)

embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
vectorstore = Pinecone.from_texts(chunks, embedder, index_name=index_name)

print("✅ Ingestion Complete")
