pip install requests pdfplumber openai

import requests
import pdfplumber
import re
import openai
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:

    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)

# Set up OpenAI API key
openai.api_key = "openai.api_key"

# URL of the PDF
pdf_url = "https://api.pageplace.de/preview/DT0400.9781684205059_A46804179/preview-9781684205059_A46804179.pdf"

# Download the PDF from the URL and extract text
def download_and_extract_text(url):
    try:
        # Download the PDF
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an error for bad status codes

        # Use BytesIO to handle the PDF in memory
        with BytesIO(response.content) as pdf_file:
            with pdfplumber.open(pdf_file) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text()
                return text
    except requests.exceptions.RequestException as e:
        print(f"Failed to download the PDF: {e}")
        return None
    except Exception as e:
        print(f"Failed to extract text from PDF: {e}")
        return None

# Preprocess the text (split into chunks)
def preprocess_text(text, chunk_size=500):
    # Remove extra spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    # Split text into chunks
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

# Retrieve the most relevant chunk for a query
def retrieve_relevant_chunk(chunks, query):
    # Use TF-IDF vectorizer to convert text into vectors
    vectorizer = TfidfVectorizer()
    chunk_vectors = vectorizer.fit_transform(chunks)
    query_vector = vectorizer.transform([query])

    # Compute cosine similarity between query and chunks
    similarities = cosine_similarity(query_vector, chunk_vectors).flatten()
    most_relevant_index = np.argmax(similarities)
    return chunks[most_relevant_index]

# Generate a response using OpenAI's GPT
def generate_response(query, context):
    # Create a prompt with the context and query
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    
    # Use the new OpenAI API format
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",  # Use GPT-3.5-turbo or GPT-4
        messages=[
            {"role": "system", "content": "You are a helpful healthcare assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.7
    )
    return response.choices[0].message.content

# Chatbot interaction loop
def chatbot():
    print("Welcome to the Healthcare Assistant Chatbot!")
    print("You can ask me questions about neurosurgery, my resource is the Handbook of Neurosurgery.")
    print("Type 'exit' to quit.")

    # Download and extract text from the PDF
    text = download_and_extract_text(pdf_url)
    if not text:
        print("Chatbot: Unable to download or extract text from the PDF. Exiting.")
        return

    # Preprocess the text into chunks
    chunks = preprocess_text(text)

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break

        # Retrieve the most relevant chunk
        relevant_chunk = retrieve_relevant_chunk(chunks, user_input)

        # Generate a response using OpenAI's GPT
        response = generate_response(user_input, relevant_chunk)
        print(f"Chatbot: {response}")

# Run the chatbot
chatbot()
