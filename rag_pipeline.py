import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import pinecone
from dotenv import load_dotenv

load_dotenv()

pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))
index_name = os.getenv("PINECONE_INDEX")

embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectordb = Pinecone.from_existing_index(index_name=index_name, embedding=embed_model)

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a legal advisor. Use the context below to answer the question:

    Context:
    {context}

    Question:
    {question}

    Helpful Answer:"""
)

llm = HuggingFacePipeline.from_model_id(
    model_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    model_kwargs={"max_new_tokens": 300}
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template}
)

def get_answer(question):
    return qa.run(question)
