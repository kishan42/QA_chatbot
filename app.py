import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_objectbox.vectorstores import ObjectBox
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import time


groq_api_key = "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"

# Embedding using HuggingFace
HuggingFace_embedding = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-m3",  # sentence-transformers/all-MiniLM-l6-v2
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

st.title("Question Answering Chatbot")

llm = ChatGroq(groq_api_key=groq_api_key,
               model_name="Llama3-8b-8192")


prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions: {input}

    """
)


## Vector Embedding and Objectbox Vectorstore db

def process_documents(uploaded_file):
    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        st.session_state.embeddings = HuggingFace_embedding
        st.session_state.loader = PyPDFLoader(tmp_file_path)  # Load the PDF from the temp file path
        st.session_state.docs = st.session_state.loader.load()  # Documents Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = ObjectBox.from_documents(
            st.session_state.final_documents, st.session_state.embeddings, embedding_dimensions=768
        )
        st.write("ChatBot is ready")


# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    if st.button("Process"):
        process_documents(uploaded_file)

# Question input and response retrieval
input_prompt = st.text_input("Enter Your Question From Documents")

if input_prompt and "vectors" in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()

    response = retrieval_chain.invoke({'input': input_prompt})

    st.write("Response time:", time.process_time() - start)
    st.write(response['answer'])

    # With a Streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")