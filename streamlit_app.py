from openai import OpenAI
import streamlit as st
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from PyPDF2 import PdfReader
import os

# Load and process the textbook
@st.cache_resource
def load_textbook():
    pdf_url = "https://med.mui.ac.ir/sites/med/files/users/jarah-maghz/Handbook%20of%20Neurosurgery%208.pdf"
    response = requests.get(pdf_url)
    with open("textbook.pdf", "wb") as f:
        f.write(response.content)
    
    reader = PdfReader("textbook.pdf")
    text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(text)
    
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(texts, embeddings)
    return vector_store

st.title("🩺 AI Health Assistant (RAG-powered)")
st.write(
    "This AI-powered healthcare assistant provides general medical guidance using Retrieval-Augmented Generation (RAG)."
    "\n⚠️ **Disclaimer:** This is not a substitute for professional medical advice."
)

openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:
    os.environ["OPENAI_API_KEY"] = openai_api_key
    vector_store = load_textbook()
    client = OpenAI(api_key=openai_api_key)

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": "You are a helpful healthcare assistant providing medical insights based on a neurosurgery textbook. Always advise users to consult a licensed medical professional."}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a health-related question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Retrieve relevant information from the textbook
        docs = vector_store.similarity_search(prompt, k=3)
        retrieved_text = "\n".join([doc.page_content for doc in docs])

        # Generate response with context
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Use the retrieved textbook information to answer the user's query."},
                {"role": "user", "content": f"User question: {prompt}\nRelevant textbook info: {retrieved_text}"}
            ]
        )

        response_text = completion.choices[0].message.content

        with st.chat_message("assistant"):
            st.markdown(response_text)
        
        st.session_state.messages.append({"role": "assistant", "content": response_text})

