import streamlit as st
from rag_pipeline import get_answer

st.set_page_config(page_title="🧑‍⚖️ Legal Advisor - Indian Labor Laws", layout="centered")
st.title("🧑‍⚖️ RAG-based Legal Advisor")

st.markdown("""
This tool allows you to ask any question related to Indian labor laws. It uses Retrieval Augmented Generation (RAG) to find the most relevant context and then generates an answer using a language model.
""")

question = st.text_input("Ask your question below:", placeholder="e.g., What are the rights of contract workers?")

if question:
    with st.spinner("Retrieving answer..."):
        response = get_answer(question)
    st.markdown("**Answer:**")
    st.success(response)
