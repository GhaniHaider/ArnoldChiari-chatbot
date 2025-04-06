import streamlit as st
import os
import json
import requests
from google.colab import files
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Streamlit Page Configuration
st.set_page_config(page_title="Neurosurgery Guide", page_icon="🧠")

# Title and Description
st.title("Neurosurgery Guide 🤯")
st.write(
    "This chatbot provides healthcare-related information based on the Handbook of Neurosurgery."
)

# Gemini API Key Input
gemini_api_key = st.text_input("Enter your Gemini API Key", type="password")

if not gemini_api_key:
    st.warning("Please enter your Gemini API Key to continue.")

else:
    # Check if the PDF is available
    pdf_path = "/content/neurosurgeryHandbook.pdf"

    if not os.path.exists(pdf_path):
        st.warning("The book 'neurosurgeryHandbook.pdf' is not found. Please upload the PDF file.")
        
        uploaded_file = st.file_uploader("Upload the Neurosurgery Handbook PDF", type=["pdf"])
        
        if uploaded_file is not None:
            # Save the uploaded PDF
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                st.success("PDF uploaded successfully!")
    else:
        st.info("PDF file found. Proceeding with processing...")

    # Process PDF if file exists
    if os.path.exists(pdf_path):
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        documents = text_splitter.split_documents(pages)

        # Create embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Create FAISS Vector Store
        db = FAISS.from_documents(documents, embeddings)

        # Prompt Template for Chatbot
        PROMPT_TEMPLATE = """
        You are Neurosurgery Guide 🤯 - my responses are based on the Handbook of Neurosurgery.
        
        Only answer using the book content provided. If you don't know the answer or if the question has no relevant domain words like aneurysm, brain, spine, etc., reply:

        "My responses are limited, you must ask the right question."

        Context: {context}
        Question: {question}
        Answer:
        """

        PROMPT = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )

        # Load Chat Model (Correct Model)
        llm = ChatGoogleGenerativeAI(
            model="models/chat-bison-001",  # Correct model for Chat
            temperature=0
        )

        # Create Retrieval QA Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(),
            chain_type_kwargs={"prompt": PROMPT}
        )

        # Domain Keywords
        domain_keywords = ["aneurysm", "brain", "spine", "tumor", "neurosurgery", "skull", "nerve"]

        # Create chat input field to allow user to enter a message
        user_input = st.chat_input("Ask a healthcare question...")

        # Handle user input
        if user_input:
            # Store and display the user's input message
            st.session_state.messages.append({"role": "user", "content": user_input})

            with st.chat_message("user"):
                st.markdown(user_input)

            # Check for domain keywords in the user's query
            if any(word.lower() in user_input.lower() for word in domain_keywords):
                response = qa_chain.run(user_input)
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                response = "My responses are limited, you must ask the right question."
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

    # Display chat history
    if "messages" in st.session_state:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
