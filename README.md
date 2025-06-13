# legal-rag-advisor
# 🧑‍⚖️ RAG-based Legal Advisor for Indian Labor Laws

A Streamlit-based AI assistant that uses Retrieval Augmented Generation (RAG) to answer questions about Indian labor laws.

## 🔧 Tech Stack
- Python
- Streamlit
- LangChain
- Pinecone
- HuggingFace (Mistral 7B)
- Prompt Engineering
- PDF Ingestion (PyMuPDF)

## 📁 Setup Instructions

```bash
# 1. Clone repo
$ git clone https://github.com/yourusername/legal-rag-advisor.git
$ cd legal-rag-advisor

# 2. Set up environment
$ python -m venv venv
$ source venv/bin/activate  # On Windows: venv\Scripts\activate
$ pip install -r requirements.txt

# 3. Add Pinecone credentials to .env
PINECONE_API_KEY=your_key
PINECONE_ENV=your_env
PINECONE_INDEX=your_index

# 4. Ingest data
$ python ingest_pdfs.py

# 5. Run app
$ streamlit run app.py
