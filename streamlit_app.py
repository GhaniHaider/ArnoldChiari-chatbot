import streamlit as st
import requests
import json
import re
from io import BytesIO
import pdfplumber
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Streamlit Page Configuration
st.set_page_config(page_title="Neurosurgery Chatbot", page_icon="🏥")

# Title and Description
st.title("Neurosurgery Chatbot")
st.write("Ask questions about neurosurgery, based on the Handbook of Neurosurgery.")

# Gemini API Key Input
gemini_api_key = st.text_input("Enter your Gemini API Key", type="password")

if not gemini_api_key:
    st.warning("Please enter your Gemini API Key to continue.")
else:
    # Gemini API Endpoint
    GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_api_key}"

    # URL of the PDF
    pdf_url = "https://med.mui.ac.ir/sites/med/files/users/jarah-maghz/Handbook%20of%20Neurosurgery%208.pdf"

    # Download the PDF from the URL and extract text
    def download_and_extract_text(url):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            with BytesIO(response.content) as pdf_file:
                with pdfplumber.open(pdf_file) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text()
                    return text
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to download the PDF: {e}")
            return None
        except Exception as e:
            st.error(f"Failed to extract text from PDF: {e}")
            return None

    # Preprocess the text (split into chunks)
    def preprocess_text(text, chunk_size=500):
        text = re.sub(r'\s+', ' ', text)
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        return chunks

    # Retrieve the most relevant chunk for a query
    def retrieve_relevant_chunk(chunks, query):
        vectorizer = TfidfVectorizer()
        chunk_vectors = vectorizer.fit_transform(chunks)
        query_vector = vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, chunk_vectors).flatten()
        most_relevant_index = np.argmax(similarities)
        return chunks[most_relevant_index]

    # Generate a response using Gemini
    def generate_response(query, context):
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        try:
            response = requests.post(
                GEMINI_API_URL,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
            )
            response.raise_for_status()
            response_data = response.json()
            if "candidates" in response_data and response_data["candidates"]:
                return response_data["candidates"][0]["content"]["parts"][0]["text"]
            else:
                return "Sorry, I couldn't generate a response."
        except requests.exceptions.RequestException as e:
            st.error(f"Error with Gemini API: {e}")
            return "An error occurred while generating the response."
        except Exception as e:
            st.error(f"Gemini response error: {e}")
            return "An unexpected error occurred."

    # Chatbot interaction
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask a question about neurosurgery...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        text = download_and_extract_text(pdf_url)
        if text:
            chunks = preprocess_text(text)
            relevant_chunk = retrieve_relevant_chunk(chunks, user_input)
            gemini_response = generate_response(user_input, relevant_chunk)
            st.session_state.messages.append({"role": "assistant", "content": gemini_response})
            with st.chat_message("assistant"):
                st.markdown(gemini_response)
        else:
            st.error("Unable to process PDF content.")
